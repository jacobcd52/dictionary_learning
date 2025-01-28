import torch as t
import torch.nn as nn
import einops
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
import numpy as np
import json 

from trainers.top_k import AutoEncoderTopK


@dataclass
class SubmoduleConfig:
    """Configuration for a single submodule in the SCAE suite"""
    activation_dim: int
    dict_size: int
    k: int

class SCAESuite(nn.Module):
    """A suite of Sparsely-Connected TopK Autoencoders"""
    
    def __init__(
        self,
        submodule_configs: Dict[str, SubmoduleConfig],
        connections: Optional[Dict[str, Dict[str, t.Tensor]]] = None,
        dtype: t.dtype = t.float32,
        device: Optional[str] = None,
    ):
        """
        Args:
            submodule_configs: Dictionary mapping submodule names to their configs
            connections: Optional dictionary specifying sparse connections:
                {downstream_name: {upstream_name: tensor}}
                where tensor has shape [num_down_features, C] and contains indices
                of connected upstream features, padded with -1.
                C is the maximum number of connections per downstream feature.
            dtype: Data type for the autoencoders
            device: Device to place the autoencoders on
        """
        super().__init__()
        self.device = device or ('cuda' if t.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.is_pretrained = False
        
        # Initialize autoencoders
        self.aes = nn.ModuleDict({
            name: AutoEncoderTopK(
                config.activation_dim,
                config.dict_size,
                config.k
            ).to(device=self.device, dtype=self.dtype)
            for name, config in submodule_configs.items()
        })
        
        # Store configs for reference
        self.configs = submodule_configs
        self.submodule_names = list(submodule_configs.keys())

        # Initialize and validate connections
        self.connections = {}
        if connections is not None:
            for down_name, up_dict in connections.items():
                self.connections[down_name] = {}
                for up_name, conn_tensor in up_dict.items():
                    # Verify shape
                    assert conn_tensor.shape[0] == self.configs[down_name].dict_size, \
                        f"Connection tensor for {down_name}->{up_name} has wrong shape. Expected first dim {self.configs[down_name].dict_size}, got {conn_tensor.shape[0]}"
                    # Verify indices are valid
                    valid_indices = (conn_tensor >= -1) & (conn_tensor < self.configs[up_name].dict_size)
                    assert valid_indices.all(), \
                        f"Invalid indices in connection tensor for {down_name}->{up_name}. All values must be -1 or valid feature indices."
                    
                    self.connections[down_name][up_name] = conn_tensor.to(device=self.device, dtype=t.long)

    def get_initial_contributions(
            self,
            initial_act: t.Tensor, # [batch, d_in]
            down_name: str, # e.g. 'mlp_5'
            down_indices: t.Tensor, # [batch, k_down + n_threshold + n_random]
    ):
        W_enc_FD = self.aes[down_name].encoder.weight
        contributions_BF = initial_act @ W_enc_FD.T
        contributions_BK = contributions_BF.gather(1, down_indices)
        return contributions_BK

    def get_pruned_contributions(
        self,
        up_name: str,
        down_name: str,
        up_indices: t.Tensor,  # [batch, k_up]
        down_indices: t.Tensor,  # [batch, k_down + n_threshold + n_random]
        up_vals: t.Tensor,  # [batch, k_up]
    ) -> t.Tensor:  # [batch, k_down + n_threshold + n_random]
        """
        Memory-efficient version that maintains GPU parallelism.
        """
        batch_size = up_indices.shape[0]
        k_up = up_indices.shape[1]
        total_down_features = down_indices.shape[1]  # k_down + n_threshold + n_random
        device = up_indices.device
        
        # Get connection tensor for this pair of modules
        connections = self.connections[down_name][up_name]  # [num_down, C]
        C = connections.shape[1]
        
        # Get weights for the active features
        up_decoder = self.aes[up_name].decoder.weight  # [d_in, d_up]
        down_encoder = self.aes[down_name].encoder.weight  # [d_down, d_in]
        
        active_up_vectors = up_decoder[:, up_indices.reshape(-1)].T  # [batch*k_up, d_in]
        active_up_vectors = active_up_vectors.view(batch_size, k_up, -1)  # [batch, k_up, d_in]
        
        # Get all relevant downstream vectors
        active_down_vectors = down_encoder[down_indices.reshape(-1)]  # [batch*(k_down + n_threshold + n_random), d_in]
        active_down_vectors = active_down_vectors.view(batch_size, total_down_features, -1)  # [batch, k_down + n_threshold + n_random, d_in]
        
        # Compute virtual weights between active features
        virtual_weights = einops.einsum(
            active_down_vectors, active_up_vectors,
            "batch k_down d_in, batch k_up d_in -> batch k_down k_up"
        )  # [batch, k_down + n_threshold + n_random, k_up]
        
        # Initialize contributions tensor for all selected downstream features
        contributions = t.zeros((batch_size, total_down_features), device=device, dtype=up_vals.dtype)
        
        chunk_size = 256
        num_chunks = (C + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, C)
            
            # Get the current chunk of connections for ALL downstream features we care about
            chunk_connections = connections[down_indices.reshape(-1), start_idx:end_idx]  # [(batch*(k_down + n_threshold + n_random)), chunk_size]
            chunk_connections = chunk_connections.view(batch_size, total_down_features, -1)  # [batch, k_down + n_threshold + n_random, chunk_size]
            
            valid_mask = (chunk_connections != -1)  # [batch, k_down + n_threshold + n_random, chunk_size]
            
            # Match upstream features
            chunk_connections_expanded = chunk_connections.unsqueeze(-1)  # [batch, k_down + n_threshold + n_random, chunk_size, 1]
            up_indices_expanded = up_indices.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, k_up]
            connection_mask = (chunk_connections_expanded == up_indices_expanded) & valid_mask.unsqueeze(-1)
            
            # Apply mask and compute contributions
            chunk_contributions = t.where(
                connection_mask,
                virtual_weights.unsqueeze(2) * up_vals.unsqueeze(1).unsqueeze(2),
                t.zeros_like(virtual_weights[:, :, None, :])
            )
            
            contributions += chunk_contributions.sum(dim=(2, 3))
        
        # Add bias terms for all selected features
        b_dec_contrib = self.aes[down_name].encoder.weight @ self.aes[up_name].b_dec
        contributions = contributions + b_dec_contrib[down_indices]
        
        return contributions

    def vanilla_forward(
        self,
        inputs: Dict[str, t.Tensor],
        return_features: bool = False,
        return_topk: bool = False,
        n_threshold: Optional[int] = None
    ) -> Union[Dict[str, t.Tensor], Tuple[Dict[str, t.Tensor], Dict[str, t.Tensor]]]:
        """
        Modified vanilla forward pass to support additional threshold features.
        
        Args:
            inputs: Dictionary mapping submodule names to input tensors
            return_features: Whether to return encoded features
            return_topk: Whether to return top-k feature indices and values
            n_threshold: Optional number of additional threshold features to return
        """
        results = {}
        features = {}
        topk_info = {}
        
        for name, ae in self.aes.items():
            if name not in inputs:
                continue
                
            if return_topk:
                feat, top_vals, top_idxs = ae.encode(inputs[name], return_topk=True, n_threshold=n_threshold)
                topk_info[name] = (top_idxs, top_vals)
            else:
                feat = ae.encode(inputs[name])
                
            recon = ae.decode(feat)
            
            results[name] = recon
            if return_features:
                features[name] = feat
        
        if return_features and return_topk:
            return results, features, topk_info
        elif return_features:
            return results, features
        elif return_topk:
            return results, topk_info
        else:
            return results

    def pruned_forward_train(
        self,
        initial_acts: t.Tensor,
        inputs: Dict[str, t.Tensor],
        layernorm_scales: Dict[str, t.Tensor],
        n_threshold: int,
        n_random: int
    ) -> Dict[str, Dict[str, t.Tensor]]:
        """
        Run modified forward pass that computes:
        - Top k features
        - Additional n_threshold features
        - n_random randomly selected features
        
        Args:
            initial_acts: Initial activations
            inputs: Dictionary mapping submodule names to input tensors
            layernorm_scales: Dictionary of layernorm scales
            n_threshold: Number of additional threshold features to compute
            n_random: Number of random features to compute
            
        Returns:
            Dictionary with structure:
            {module_name: {
                'topk': tensor of top-k feature activations,
                'threshold': tensor of threshold feature activations,
                'random': tensor of random feature activations,
                'all_indices': tensor of concatenated indices [batch, k_down + n_threshold + n_random]
            }}
        """
        # First get vanilla features and topk info
        results, features, topk_info = self.vanilla_forward(
            inputs, return_features=True, return_topk=True, n_threshold=n_threshold
        )
        
        outputs = {}
        
        # Now compute features for each module
        for down_name, config in self.configs.items():
            if down_name not in self.connections or down_name not in inputs:
                continue
            
            batch_size = inputs[down_name].shape[0]
            device = inputs[down_name].device
                    
            # Get top k + threshold indices and values
            down_idxs_extended = topk_info[down_name][0]  # [batch, k + n_threshold]
            k_down = self.configs[down_name].k
            
            # Split into topk and threshold
            down_idxs_topk = down_idxs_extended[:, :k_down]
            down_idxs_threshold = down_idxs_extended[:, k_down:]
            
            # Generate random indices (excluding those already selected)
            all_indices = set(range(config.dict_size))
            batch_random_indices = []
            
            for b in range(batch_size):
                used_indices = set(down_idxs_extended[b].cpu().numpy())
                available_indices = list(all_indices - used_indices)
                selected_random = t.tensor(
                    np.random.choice(available_indices, n_random, replace=False),
                    device=device
                )
                batch_random_indices.append(selected_random)
                
            down_idxs_random = t.stack(batch_random_indices)  # [batch, n_random]
            
            # Combine all indices for computation
            down_idxs_all = t.cat([
                down_idxs_topk,
                down_idxs_threshold,
                down_idxs_random
            ], dim=1)  # [batch, k_down + n_threshold + n_random]
            
            # Get initial contributions
            approx_acts = self.get_initial_contributions(
                initial_acts, down_name, down_idxs_all
            )
            
            # Get contributions from each upstream module
            for up_name in self.connections[down_name].keys():
                if up_name not in inputs:
                    continue
                    
                up_idxs = topk_info[up_name][0][:, :self.configs[up_name].k]  # [batch, k_up]
                up_vals = topk_info[up_name][1][:, :self.configs[up_name].k]  # [batch, k_up]
                
                contributions = self.get_pruned_contributions(
                    up_name, down_name, up_idxs, down_idxs_all, up_vals
                )
                
                approx_acts = approx_acts + contributions
            
            # Apply layernorm scaling
            approx_acts = approx_acts / layernorm_scales[down_name].unsqueeze(1)
            
            # Add encoder bias and subtract decoder bias contribution
            b_enc = self.aes[down_name].encoder.bias[down_idxs_all]
            approx_acts = approx_acts + b_enc
            
            W_enc = self.aes[down_name].encoder.weight[down_idxs_all]
            approx_acts = approx_acts - W_enc @ self.aes[down_name].b_dec
            
            # Split results and store concatenated indices
            outputs[down_name] = {
                'topk': approx_acts[:, :k_down],
                'threshold': approx_acts[:, k_down:k_down + n_threshold],
                'random': approx_acts[:, k_down + n_threshold:],
                'all_indices': down_idxs_all  # [batch, k_down + n_threshold + n_random]
            }
        
        return outputs

    # TODO: put evaluation code in evaluate.py
    @t.no_grad()
    def evaluate_varexp_batch(
        self,
        initial_acts: t.Tensor,
        input_acts: Dict[str, t.Tensor],
        target_acts: Dict[str, t.Tensor],
        layernorm_scales: Dict[str, t.Tensor],
        use_sparse_connections: bool = False,
        normalize_batch: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate reconstruction quality metrics.
        
        Args:
            input_acts: Dictionary of input activations
            target_acts: Dictionary of target activations
            use_sparse_connections: Whether to use approximate forward pass
            normalize_batch: Whether to normalize inputs by their mean norm
            
        Returns:
            Dictionary of metrics for each SAE
        """
        metrics = {}
        
        # Move inputs to device
        input_acts = {
            k: v.to(device=self.device, dtype=self.dtype) 
            for k, v in input_acts.items()
        }
        target_acts = {
            k: v.to(device=self.device, dtype=self.dtype) 
            for k, v in target_acts.items()
        }
        
        # Get reconstructions based on forward pass type
        if use_sparse_connections:
            results = self.approx_forward(initial_acts, input_acts, layernorm_scales)
        else:
            results = self.vanilla_forward(input_acts)
        
        for name, ae in self.aes.items():
            if name not in input_acts:
                continue
                
            x = input_acts[name]
            tgt = target_acts[name]
            
            if normalize_batch:
                scale = (ae.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
                tgt = tgt * scale
            
            x_hat = results[name]
            if normalize_batch:
                x_hat = x_hat / scale
            
            # Compute metrics
            l2_loss = t.linalg.norm(tgt - x_hat, dim=-1).mean()
            
            total_variance = t.var(tgt, dim=0).sum()
            residual_variance = t.var(tgt - x_hat, dim=0).sum()
            frac_variance_explained = 1 - residual_variance / total_variance
            
            metrics[name] = {
                'l2_loss': l2_loss.item(),
                'FVU': 1 - frac_variance_explained.item()
            }
        
        return metrics

    @t.no_grad()
    def evaluate_ce_batch(
        self,
        model,
        text: Union[str, t.Tensor],
        initial_submodule: 'Module',
        submodules: Dict[str, Tuple['Module', str]],  # Maps SAE name to (submodule, io_type)
        layernorm_submodules: Dict[str, 'Module'],
        use_sparse_connections: bool = False,
        max_len: Optional[int] = None,
        normalize_batch: bool = False,
        device: Optional[str] = None,
        tracer_args: dict = {'use_cache': False, 'output_attentions': False}
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate cross entropy loss when patching in reconstructed activations.
        
        Args:
            model: TransformerLens model to evaluate
            text: Input text or token tensor
            submodules: Dictionary mapping SAE names to (submodule, io_type) pairs
                where io_type is one of ['in', 'out', 'in_and_out']
            use_sparse_connections: Whether to use approximate forward pass
            max_len: Optional maximum sequence length
            normalize_batch: Whether to normalize activations by their mean norm
            device: Device to run evaluation on
            tracer_args: Arguments for model.trace
            
        Returns:
            Dictionary mapping submodule names to metrics:
                - loss_original: CE loss of unmodified model
                - loss_reconstructed: CE loss with reconstructed activations
                - loss_zero: CE loss with zeroed activations
                - frac_recovered: Fraction of performance recovered vs zeroing
        """

        device = device or self.device
        
        # Setup invoker args for length control
        invoker_args = {"truncation": True, "max_length": max_len} if max_len else {}
        
        # Get original model outputs
        with model.trace(text, invoker_args=invoker_args):
            logits_original = model.output.save()
        logits_original = logits_original.value
        
        # Get all activations in one pass TODO: layernorm_scales
        saved_activations = {}
        with model.trace(text, **tracer_args, invoker_args=invoker_args):
            for name, (submodule, io) in list(submodules.items()) + [('initial', (initial_submodule, 'in'))]:
                if io in ['in', 'in_and_out']:
                    x = submodule.input
                elif io == 'out':
                    x = submodule.output
                else:
                    raise ValueError(f"Invalid io type: {io}")
                
                if normalize_batch:
                    scale = (self.aes[name].activation_dim ** 0.5) / x.norm(dim=-1).mean()
                    x = x * scale
                else:
                    scale = 1.0
                
                saved_activations[name] = {
                    'x': x.save(),
                    'io': io,
                    'scale': scale
                }
        
        # Process saved activations
        for name, saved in saved_activations.items():
            if isinstance(saved['x'].value, tuple):
                saved['x'] = saved['x'].value[0]
            else:
                saved['x'] = saved['x'].value
                
        # Get reconstructions
        inputs = {
            name: saved['x'].to(device).view(-1, saved['x'].shape[-1])
            for name, saved in saved_activations.items()
            if name != 'initial'
        }

        initial_acts = saved_activations['initial']['x'].to(device).view(-1, saved_activations['initial']['x'].shape[-1])
        
        # Choose forward pass based on use_sparse_connections
        if use_sparse_connections:
            layernorm_scales = {
                name: t.ones([initial_acts.shape[0]], dtype=self.dtype, device=device)
                for name in submodules
            } # TODO fix
            reconstructions = self.approx_forward(initial_acts, inputs, layernorm_scales)
        else:
            reconstructions = self.vanilla_forward(inputs)
        
        # Reshape reconstructions back to original shapes and apply scaling
        for name, saved in saved_activations.items():
            if name == 'initial':
                continue
            x_shape = saved['x'].shape
            x_hat = reconstructions[name]
            x_hat = x_hat.view(x_shape)
            if normalize_batch:
                x_hat = x_hat / saved['scale']
            reconstructions[name] = x_hat.to(model.dtype)
        
        # Get tokens for loss computation
        if isinstance(text, t.Tensor):
            tokens = text
        else:
            with model.trace(text, **tracer_args, invoker_args=invoker_args):
                model_input = model.input.save()
            try:
                tokens = model_input.value[1]['input_ids']
            except:
                tokens = model_input.value[1]['input']
        
        # Setup loss function
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            loss_kwargs = {'ignore_index': model.tokenizer.pad_token_id}
        else:
            loss_kwargs = {}
        
        # Compute original loss
        try:
            logits = logits_original.logits
        except:
            logits = logits_original
            
        loss_original = t.nn.CrossEntropyLoss(**loss_kwargs)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1)
        ).item()
        
        # Compute per-submodule metrics
        results = {}
        
        for name, (submodule, io) in submodules.items():
            # Test with reconstructed activations
            with model.trace(text, **tracer_args, invoker_args=invoker_args):
                x_hat = reconstructions[name]
                # Patch reconstructions
                submodule.input = x_hat
                if io in ['out', 'in_and_out']:
                    if "attn" in name:
                        submodule.output = (x_hat,)
                    elif "mlp" in name:
                        submodule.output = x_hat
                    else:
                        raise ValueError(f"Invalid submodule name: {name}")
                
                logits_reconstructed = model.output.save()
            
            # Test with zeroed activations
            with model.trace(text, **tracer_args, invoker_args=invoker_args):
                if io in ['in', 'in_and_out']:
                    x = submodule.input
                    submodule.input = t.zeros_like(x)
                if io in ['out', 'in_and_out']:
                    x = submodule.output
                    if "attn" in name:
                        submodule.output = (t.zeros_like(x[0]),)
                    elif "mlp" in name:
                        submodule.output = t.zeros_like(x)
                
                logits_zero = model.output.save()
            
            # Format logits and compute losses
            try:
                logits_reconstructed = logits_reconstructed.value.logits
                logits_zero = logits_zero.value.logits
            except:
                logits_reconstructed = logits_reconstructed.value
                logits_zero = logits_zero.value
            
            loss_reconstructed = t.nn.CrossEntropyLoss(**loss_kwargs)(
                logits_reconstructed[:, :-1, :].reshape(-1, logits_reconstructed.shape[-1]),
                tokens[:, 1:].reshape(-1)
            ).item()
            
            loss_zero = t.nn.CrossEntropyLoss(**loss_kwargs)(
                logits_zero[:, :-1, :].reshape(-1, logits_zero.shape[-1]),
                tokens[:, 1:].reshape(-1)
            ).item()
            
            # Compute recovery fraction
            if loss_original - loss_zero != 0:
                frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
            else:
                frac_recovered = 0.0
            
            results[name] = {
                'loss_original': loss_original,
                'loss_reconstructed': loss_reconstructed,
                'loss_zero': loss_zero,
                'frac_recovered': frac_recovered
            }
        
        return results

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        device: Optional[str] = None,
        dtype: t.dtype = t.float32
    ) -> "SCAESuite":
        """
        Load a pretrained SCAESuite from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID containing the saved model
            device: Device to load the model on
            dtype: Data type for model parameters
            
        Returns:
            Initialized SCAESuite with pretrained weights
            
        Example:
            >>> suite = SCAESuite.from_pretrained("organization/model-name")
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub package is required to load pretrained models. "
                "Install with: pip install huggingface_hub"
            )

        # Download configuration
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract submodule configs
        if config.get("is_pretrained", False):
            # Handle nested pretrained case (though this shouldn't happen in practice)
            raise ValueError(
                f"Model {repo_id} was initialized from pretrained weights. "
                "Please load from the original source."
            )
        
        submodule_configs = {
            name: SubmoduleConfig(**cfg)
            for name, cfg in config["submodule_configs"].items()
        }
        
        # Try to load connections if they exist
        try:
            connections_path = hf_hub_download(repo_id=repo_id, filename="connections.pt")
            connections = t.load(connections_path, map_location='cpu')
        except Exception:
            connections = None
        
        # Initialize suite
        suite = cls(
            submodule_configs=submodule_configs,
            connections=connections,
            dtype=dtype,
            device=device
        )
        
        # Download and load state dict
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pt")
        checkpoint = t.load(checkpoint_path, map_location='cpu')
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'suite_state' in checkpoint:
            state_dict = checkpoint['suite_state']
        else:
            # Handle case where checkpoint is just the state dict
            state_dict = checkpoint
        
        # Load state dict into suite
        missing_keys, unexpected_keys = suite.load_state_dict(state_dict)
        
        if len(missing_keys) > 0:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        suite.is_pretrained = True
        return suite




@dataclass
class TrainerConfig:
    """Configuration for SCAE Suite training"""
    steps: int
    auxk_alpha: float = 1/32  # Weight for auxiliary loss (dead features)
    connection_sparsity_coeff: float = 0.0  # Weight for connection sparsity loss
    lr_decay_start_proportion: int = 0.8  # Fraction of training at which to start decaying LR
    dead_feature_threshold: int = 10_000_000  # Tokens since last activation
    base_lr: float = 2e-4  # Base learning rate (will be scaled by dictionary size)

class TrainerSCAESuite:
    def __init__(
        self,
        suite: SCAESuite,
        config: TrainerConfig,
        seed: Optional[int] = None,
        wandb_name: Optional[str] = None,
    ):
        """
        Trainer for Sparse Connected Autoencoder Suite.
        
        Args:
            suite: SCAESuite object to train
            config: Training configuration
            seed: Random seed for reproducibility
            wandb_name: Optional W&B run name
        """
        self.suite = suite
        self.config = config
        self.wandb_name = wandb_name
        
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)
        
        # Initialize learning rates
        self.lrs = {
            name: config.base_lr / (ae.dict_size / 2**14)**0.5
            for name, ae in suite.aes.items()
        }
        
        # Initialize optimizer with per-module learning rates
        self.optimizer = t.optim.Adam([
            {'params': ae.parameters(), 'lr': self.lrs[name]}
            for name, ae in suite.aes.items()
        ], betas=(0.9, 0.999))
        
        # Learning rate scheduler
        def lr_fn(step):
            if step < config.lr_decay_start_proportion * config.steps:
                return 1.0
            return (config.steps - step) / (config.steps - config.lr_decay_start_proportion * config.steps)
        
        self.scheduler = t.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )
        
        # Initialize feature usage tracking
        self.num_tokens_since_fired = {
            name: t.zeros(ae.dict_size, dtype=t.long, device=suite.device)
            for name, ae in suite.aes.items()
        }
        
        # Initialize metrics
        self.effective_l0s = {name: -1 for name in suite.submodule_names}
        self.dead_features = {name: -1 for name in suite.submodule_names}

    def update(
        self,
        step: int,
        initial_acts: t.Tensor,
        input_acts: Dict[str, t.Tensor],
        target_acts: Dict[str, t.Tensor],
        layernorm_scales: Dict[str, t.Tensor],
        log_metrics: bool = False,
    ) -> float:
        """
        Single training step.
        
        Args:
            step: Current training step
            input_acts: Dictionary of input activations for each SAE
            target_acts: Dictionary of target activations for each SAE
            log_metrics: Whether to log metrics to wandb
                
        Returns:
            Total loss for this step
        """
        # Move inputs to device
        initial_acts = initial_acts.to(device=self.suite.device, dtype=self.suite.dtype)
        input_acts = {
            k: v.to(device=self.suite.device, dtype=self.suite.dtype) 
            for k, v in input_acts.items()
        }
        target_acts = {
            k: v.to(device=self.suite.device, dtype=self.suite.dtype) 
            for k, v in target_acts.items()
        }
        
        # Initialize geometric median at step 0 only for non-pretrained SAEs
        if step == 0 and not self.suite.is_pretrained:
            for name, x in input_acts.items():
                median = geometric_median(x)
                self.suite.aes[name].b_dec.data = median
        
        # Ensure decoder norms are unit
        for ae in self.suite.aes.values():
            ae.set_decoder_norm_to_unit_norm()
        
        self.optimizer.zero_grad()
        total_loss = 0
        
        # Track losses for logging
        losses = {
            'reconstruction': {},
            'auxiliary': {},
            'connection': {},
            'vanilla FVU': {},
            'pruned FVU': {},
            'dead_features': {},
        }
        
        # First get vanilla features and reconstructions
        results, features, topk_info = self.suite.vanilla_forward(
            input_acts, return_features=True, return_topk=True
        )
        
        # Compute L2 reconstruction loss and auxiliary loss for each SAE
        for name, ae in self.suite.aes.items():
            if name not in input_acts:
                continue
                
            x = input_acts[name]
            tgt = target_acts[name]
            x_hat = results[name]
            
            # Update feature usage tracking
            num_tokens_in_step = x.size(0)
            did_fire = t.zeros_like(self.num_tokens_since_fired[name], dtype=t.bool)
            did_fire[topk_info[name][0].flatten()] = True
            self.num_tokens_since_fired[name] += num_tokens_in_step
            self.num_tokens_since_fired[name][did_fire] = 0
            
            # L2 reconstruction loss
            l2_loss = (x_hat - tgt).pow(2).sum(dim=-1).mean()
            total_loss = total_loss + l2_loss
            losses['reconstruction'][name] = l2_loss.item()
            
            # Compute variance explained
            total_variance = t.var(tgt, dim=0).sum()
            residual_variance = t.var(tgt - x_hat, dim=0).sum()
            frac_variance_explained = 1 - residual_variance / total_variance
            losses['vanilla FVU'][name] = 1 - frac_variance_explained.item()
            
            # Auxiliary loss for dead features
            if self.config.auxk_alpha > 0:
                dead_mask = (
                    self.num_tokens_since_fired[name] > 
                    self.config.dead_feature_threshold
                )
                losses['dead_features'][name] = int(dead_mask.sum())
                
                if (num_dead := int(dead_mask.sum())) > 0:
                    k_aux = x.shape[-1] // 2
                    scale = min(num_dead / k_aux, 1.0)
                    k_aux = min(k_aux, num_dead)
                    total_variance = (tgt - tgt.mean(0)).pow(2).sum(0)
                    
                    # Get activations for dead features
                    auxk_latents = t.where(
                        dead_mask[None], features[name], -t.inf
                    )
                    auxk_acts, auxk_indices = auxk_latents.topk(
                        k_aux, sorted=False
                    )
                    auxk_buffer = t.zeros_like(features[name])
                    auxk_acts = auxk_buffer.scatter_(
                        dim=-1, index=auxk_indices, src=auxk_acts
                    )
                    e_hat = ae.decode(auxk_acts)
                    
                    auxk_loss = scale * t.mean(
                        (e_hat - (x_hat - tgt)).pow(2) / total_variance
                    )
                    total_loss = total_loss + self.config.auxk_alpha * auxk_loss
                    losses['auxiliary'][name] = auxk_loss.item()
        
        # Compute connection sparsity loss
        if self.config.connection_sparsity_coeff > 0:
            # Get approximate reconstructions
            approx_results = self.suite.approx_forward(initial_acts, input_acts, layernorm_scales)
            
            for down_name in self.suite.connections:
                if down_name not in input_acts:
                    continue
                
                x_hat_approx = approx_results[down_name]
                tgt = target_acts[down_name]
                
                # L2 loss between approximate and target
                approx_loss = (x_hat_approx - tgt).pow(2).sum(dim=-1).mean()
                total_loss = total_loss + self.config.connection_sparsity_coeff * approx_loss
                losses['connection'][down_name] = approx_loss.item()
                losses['pruned FVU'][down_name] = t.var(tgt - x_hat_approx, dim=0).sum() / t.var(tgt, dim=0).sum()
        
        # Backward pass and optimization
        total_loss.backward()
        
        # Clip gradients and remove parallel components
        for ae in self.suite.aes.values():
            t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae.remove_gradient_parallel_to_decoder_directions()
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Log metrics if requested
        if log_metrics and self.wandb_name is not None:
            import wandb
            
            # Log learning rates
            lrs = {
                f"lr/{name}": param_group['lr']
                for name, param_group in zip(self.suite.aes.keys(), self.optimizer.param_groups)
            }
            
            # Flatten nested loss dict
            log_dict = {}
            for loss_type, loss_dict in losses.items():
                for name, value in loss_dict.items():
                    log_dict[f"{loss_type}/{name}"] = value
            
            log_dict.update(lrs)
            log_dict['loss/total'] = total_loss.item()
            wandb.log(log_dict, step=step)
        
        return total_loss.item()


def geometric_median(x: t.Tensor, num_iterations: int = 20) -> t.Tensor:
    """Compute geometric median of points."""
    median = x.mean(dim=0)
    for _ in range(num_iterations):
        dists = t.norm(x - median, dim=-1, keepdim=True)
        weights = 1 / (dists + 1e-8)
        median = (x * weights).sum(dim=0) / weights.sum(dim=0)
    return median