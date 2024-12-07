import torch as t
import torch.nn as nn
import einops
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import wandb

from trainers.top_k import AutoEncoderTopK

@dataclass
class SubmoduleConfig:
    """Configuration for a single submodule in the SCAE suite"""
    activation_dim: int
    dict_size: int
    k: int
    upstream_connections: List[str] = None  # Names of upstream modules to connect to

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

    def get_virtual_weights(
        self,
        up_name: str,
        down_name: str,
        up_indices: t.Tensor,  # [batch, k_up]
        down_indices: t.Tensor,  # [batch, k_down]
        up_vals: t.Tensor,  # [batch, k_up]
    ) -> t.Tensor:  # [batch, k_down]
        """
        Vectorized computation of contributions from upstream features to downstream features.
        Only computes contributions between connected feature pairs.
        """
        batch_size = up_indices.shape[0]
        k_up = up_indices.shape[1]
        k_down = down_indices.shape[1]
        device = up_indices.device
        
        # Get connection tensor for this pair of modules
        connections = self.connections[down_name][up_name]  # [num_down, C]
        C = connections.shape[1]
        
        # Get weights for the active features
        up_decoder = self.aes[up_name].decoder.weight  # [d_in, d_up]
        down_encoder = self.aes[down_name].encoder.weight  # [d_down, d_in]
        
        # For each batch element and downstream feature,
        # we need to find which of its allowed upstream connections are currently active
        
        # First get the allowed connections for each active downstream feature
        # [batch, k_down, C]
        allowed_up = connections[down_indices]
        
        # Create a mask for valid (non-padding) connections
        # [batch, k_down, C]
        valid_mask = (allowed_up != -1)
        
        # Create a mask of shape [batch, k_down, C, k_up] indicating where
        # allowed_up[b,d,c] == up_indices[b,u]
        allowed_up_expanded = allowed_up.unsqueeze(-1)  # [batch, k_down, C, 1]
        up_indices_expanded = up_indices.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, k_up]
        connection_mask = (allowed_up_expanded == up_indices_expanded) & valid_mask.unsqueeze(-1)
        
        # Get the corresponding upstream values
        # [batch, k_down, C, k_up]
        up_vals_expanded = up_vals.unsqueeze(1).unsqueeze(2).expand(-1, k_down, C, -1)
        connected_vals = t.where(connection_mask, up_vals_expanded, t.zeros_like(up_vals_expanded))
        
        # Get the corresponding virtual weights
        active_up_vectors = up_decoder[:, up_indices.reshape(-1)].T  # [batch*k_up, d_in]
        active_up_vectors = active_up_vectors.view(batch_size, k_up, -1)  # [batch, k_up, d_in]
        
        active_down_vectors = down_encoder[down_indices.reshape(-1)]  # [batch*k_down, d_in]
        active_down_vectors = active_down_vectors.view(batch_size, k_down, -1)  # [batch, k_down, d_in]
        
        # Compute all virtual weights between active features
        # [batch, k_down, k_up]
        virtual_weights = einops.einsum(
            active_down_vectors, active_up_vectors,
            "batch k_down d_in, batch k_up d_in -> batch k_down k_up"
        )
        
        # Expand virtual weights to match connection mask shape
        # [batch, k_down, C, k_up]
        virtual_weights = virtual_weights.unsqueeze(2).expand(-1, -1, C, -1)
        
        # Apply connection mask and sum contributions
        contributions = (virtual_weights * connected_vals).sum(dim=(-1, -2))  # [batch, k_down]
        
        return contributions

    def vanilla_forward(
        self,
        inputs: Dict[str, t.Tensor],
        return_features: bool = False,
        return_topk: bool = False
    ) -> Union[Dict[str, t.Tensor], Tuple[Dict[str, t.Tensor], Dict[str, t.Tensor]]]:
        """
        Run vanilla forward pass through each autoencoder.
        
        Args:
            inputs: Dictionary mapping submodule names to input tensors
            return_features: Whether to return encoded features
            return_topk: Whether to return top-k feature indices and values
            
        Returns:
            Dictionary of reconstructions for each submodule
            (Optional) Dictionary of encoded features for each submodule
            (Optional) Dictionary of (top_indices, top_values) for each submodule
        """
        results = {}
        features = {}
        topk_info = {}
        
        for name, ae in self.aes.items():
            if name not in inputs:
                continue
                
            if return_topk:
                feat, top_vals, top_idxs = ae.encode(inputs[name], return_topk=True)
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

    def approx_forward(
        self,
        inputs: Dict[str, t.Tensor]
    ) -> Dict[str, t.Tensor]:
        """
        Run approximate forward pass using virtual weights.
        
        First runs vanilla forward to get features, then uses virtual weights
        to approximate downstream features.
        
        Args:
            inputs: Dictionary mapping submodule names to input tensors
            
        Returns:
            Dictionary of reconstructions for each submodule
        """
        # First get vanilla features and topk info
        results, features, topk_info = self.vanilla_forward(
            inputs, return_features=True, return_topk=True
        )
        
        # Now compute approximate features for each module
        for down_name, config in self.configs.items():
            if down_name not in self.connections:
                continue
                
            if down_name not in inputs:
                continue
                
            down_idxs = topk_info[down_name][0]  # [batch, k_down]
            batch_size = down_idxs.shape[0]
            
            # Aggregate contributions from connected upstream modules
            approx_acts = t.zeros(
                (batch_size, config.k),
                device=self.device,
                dtype=self.dtype
            )
            
            for up_name in self.connections[down_name].keys():
                if up_name not in inputs:
                    continue
                    
                up_idxs = topk_info[up_name][0]  # [batch, k_up]
                up_vals = topk_info[up_name][1]  # [batch, k_up]
                
                contributions = self.get_virtual_weights(
                    up_name, down_name, up_idxs, down_idxs, up_vals
                )  # [batch, k_down]
                
                approx_acts = approx_acts + contributions
            
            # Create sparse feature tensor
            approx_features = t.zeros(
                (batch_size, config.dict_size),
                device=self.device,
                dtype=self.dtype
            )
            approx_features.scatter_(
                dim=1, index=down_idxs, src=approx_acts
            )
            
            # Update reconstruction
            results[down_name] = self.aes[down_name].decode(approx_features)
        
        return results
    
    @t.no_grad()
    def evaluate_varexp_batch(
        self,
        input_acts: Dict[str, t.Tensor],
        target_acts: Dict[str, t.Tensor],
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
            results = self.approx_forward(input_acts)
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
                'frac_variance_explained': frac_variance_explained.item()
            }
        
        return metrics

    @t.no_grad()
    def evaluate_ce_batch(
        self,
        model,
        text: Union[str, t.Tensor],
        submodules: Dict[str, Tuple['Module', str]],  # Maps SAE name to (submodule, io_type)
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
        
        # Get all activations in one pass
        saved_activations = {}
        with model.trace(text, **tracer_args, invoker_args=invoker_args):
            for name, (submodule, io) in submodules.items():
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
        }
        
        # Choose forward pass based on use_sparse_connections
        if use_sparse_connections:
            reconstructions = self.approx_forward(inputs)
        else:
            reconstructions = self.vanilla_forward(inputs)
        
        # Reshape reconstructions back to original shapes and apply scaling
        for name, saved in saved_activations.items():
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
        pretrained_configs: Dict[str, Dict],
        connections: Optional[Dict[str, Dict[str, t.Tensor]]] = None,
        device: Optional[str] = None
    ) -> "SCAESuite":
        """
        Load pretrained autoencoders from HuggingFace.
        
        Args:
            pretrained_configs: Dictionary mapping submodule names to configs with:
                - repo_id: HuggingFace repo ID
                - filename: Name of weights file
                - k: Number of features to use
                - upstream_connections: List of upstream module names (optional)
            connections: Optional sparse connection dictionary (see __init__)
            device: Device to load models on
                
        Returns:
            Initialized SCAESuite with pretrained weights
        """
        submodule_configs = {}
        
        # First load all state dicts to get dimensions
        state_dicts = {}
        for name, config in pretrained_configs.items():
            weights_path = hf_hub_download(
                repo_id=config['repo_id'],
                filename=config['filename']
            )
            state_dict = t.load(weights_path, map_location='cpu')
            state_dicts[name] = state_dict
            
            # Get dimensions from weights
            dict_size, activation_dim = state_dict['encoder.weight'].shape
            
            submodule_configs[name] = SubmoduleConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=config['k'],
                upstream_connections=config.get('upstream_connections')
            )
        
        # Initialize suite
        suite = cls(submodule_configs, connections=connections, device=device)
        
        # Load weights
        for name, state_dict in state_dicts.items():
            suite.aes[name].load_state_dict(state_dict)

        suite.is_pretrained = True
        return suite





@dataclass
class TrainerConfig:
    """Configuration for SCAE Suite training"""
    steps: int
    auxk_alpha: float = 0.0  # Weight for auxiliary loss (dead features)
    connection_sparsity_coeff: float = 0.0  # Weight for connection sparsity loss
    lr_decay_start: int = 24000  # Step to start learning rate decay
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
            if step < config.lr_decay_start:
                return 1.0
            return (config.steps - step) / (config.steps - config.lr_decay_start)
        
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
        input_acts: Dict[str, t.Tensor],
        target_acts: Dict[str, t.Tensor],
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
            'variance_explained': {},
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
            losses['variance_explained'][name] = frac_variance_explained.item()
            
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
            approx_results = self.suite.approx_forward(input_acts)
            
            for down_name in self.suite.connections:
                if down_name not in input_acts:
                    continue
                
                x_hat_approx = approx_results[down_name]
                tgt = target_acts[down_name]
                
                # L2 loss between approximate and target
                approx_loss = (x_hat_approx - tgt).pow(2).sum(dim=-1).mean()
                total_loss = total_loss + self.config.connection_sparsity_coeff * approx_loss
                losses['connection'][down_name] = approx_loss.item()
        
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