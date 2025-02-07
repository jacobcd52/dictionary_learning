import torch as t
import torch.nn as nn
import einops
from typing import Dict, Tuple, Optional, List, Union, Any
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
        connections: Optional[Union[Dict[str, Dict[str, t.Tensor]], str]] = None,
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
                Can also be the string "all" to use all upstream features.
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
        self.connections = connections
        self.connection_masks = {}  # Store precomputed connection masks
        if isinstance(connections, dict):
            processed_connections = {}
            for down_name, up_dict in connections.items():
                processed_connections[down_name] = {}
                self.connection_masks[down_name] = {}
                for up_name, conn_tensor in up_dict.items():
                    # Verify shape
                    num_down_features = self.configs[down_name].dict_size
                    num_up_features = self.configs[up_name].dict_size
                    assert conn_tensor.shape[0] == num_down_features, \
                        f"Connection tensor for {down_name}->{up_name} has wrong shape. Expected first dim {num_down_features}, got {conn_tensor.shape[0]}"
                    # Verify indices are valid
                    valid_indices = (conn_tensor >= -1) & (conn_tensor < num_up_features)
                    assert valid_indices.all(), \
                        f"Invalid indices in connection tensor for {down_name}->{up_name}. All values must be -1 or valid feature indices."
                    
                    processed_connections[down_name][up_name] = conn_tensor.to(device=self.device, dtype=t.long)
                    
                    # Precompute connection mask
                    connection_mask = t.zeros(num_down_features, num_up_features, device=self.device, dtype=t.bool)
                    valid_connections = conn_tensor != -1  # [num_down, C]
                    
                    # For each downstream feature, scatter its valid connections
                    for i in range(num_down_features):
                        valid_mask = valid_connections[i]  # [C]
                        valid_targets = conn_tensor[i][valid_mask]  # Get valid target indices
                        connection_mask[i].scatter_(0, valid_targets, t.ones(len(valid_targets), device=self.device, dtype=t.bool))
                    
                    self.connection_masks[down_name][up_name] = connection_mask
            self.connections = processed_connections
        elif connections is not None and connections != "all":
            raise ValueError("connections must be either None, 'all', or a dictionary of connection tensors")

    def vanilla_forward(
            self,
            inputs: Dict[str, t.Tensor],
            return_features: bool = False,
        ) -> Union[Dict[str, t.Tensor], Tuple[Dict[str, t.Tensor], Dict[str, t.Tensor]]]:
            """Forward pass that outputs feature activations."""
            results = {}
            features = {}
            
            for name, ae in self.aes.items():
                if name not in inputs:
                    continue
                    
                feat = ae.encode(inputs[name])  # [batch, num_features]
                recon = ae.decode(feat)
                
                results[name] = recon
                if return_features:
                    features[name] = feat
            
            return (results, features) if return_features else results

    def get_initial_contributions(
            self,
            initial_act: t.Tensor, # [batch, d_in]
            down_name: str, # e.g. 'mlp_5'
    ):
        """Compute initial contributions for all downstream features."""
        W_enc_FD = self.aes[down_name].encoder.weight
        contributions_BF = initial_act @ W_enc_FD.T  # [batch, num_down_features]
        return contributions_BF

    def get_pruned_contributions(
        self,
        up_name: str,
        down_name: str,
        upstream_acts: t.Tensor,  # [batch, num_up_features]
    ) -> t.Tensor:  # [batch, num_down_features]
        """Compute contributions for all downstream features using efficient matrix operations."""
        batch_size = upstream_acts.shape[0]
        num_down_features = self.configs[down_name].dict_size
        device = upstream_acts.device
        
        if self.connections == "all":
            # For "all" mode, every downstream feature is connected to every upstream feature
            up_decoder = self.aes[up_name].decoder.weight  # [d_in, d_up]
            down_encoder = self.aes[down_name].encoder.weight  # [d_down, d_in]
            
            # Compute full virtual weight matrix
            virtual_weights = down_encoder @ up_decoder  # [num_down, num_up]
            
            # Compute contributions
            contributions = upstream_acts @ virtual_weights.T  # [batch, num_down]
            
        else:
            # Get connection tensor for this pair of modules
            connections = self.connections[down_name][up_name]  # [num_down, C]
            
            # Get weights for computing virtual weights
            up_decoder = self.aes[up_name].decoder.weight  # [d_in, d_up]
            down_encoder = self.aes[down_name].encoder.weight  # [d_down, d_in]
            
            # Compute full virtual weight matrix
            virtual_weights = down_encoder @ up_decoder  # [num_down, num_up]
            
            # Get precomputed connection mask
            connection_mask = self.connection_masks[down_name][up_name]
            
            # Apply connection mask to virtual weights
            masked_weights = virtual_weights * connection_mask  # [num_down, num_up]
            
            # Compute contributions
            contributions = upstream_acts @ masked_weights.T  # [batch, num_down]
        
        # Add bias terms for all features
        b_dec_contrib = down_encoder @ self.aes[up_name].b_dec  # [num_down]
        contributions = contributions + b_dec_contrib.unsqueeze(0)  # [batch, num_down]
        
        return contributions

    def pruned_forward(
        self,
        initial_acts: t.Tensor,
        inputs: Dict[str, t.Tensor],
        layernorm_scales: Dict[str, t.Tensor],
        return_topk: bool = False
    ) -> Union[Dict[str, t.Tensor], Tuple[Dict[str, t.Tensor], Dict[str, Dict[str, t.Tensor]]]]:
        """Forward pass that computes activations for all features, then applies TopK."""
        reconstructions = {}
        topk_info = {}
        
        # Get vanilla forward features for all modules
        results, features = self.vanilla_forward(inputs, return_features=True)
        
        for down_name, config in self.configs.items():
            if down_name not in inputs:
                continue
                
            if 'attn' in down_name:
                # For attention modules, use vanilla forward pass
                feat = features[down_name]
                recon = self.aes[down_name].decode(feat)
                reconstructions[down_name] = recon
                
                if return_topk:
                    k = self.configs[down_name].k
                    top_vals, top_idx = feat.topk(k, dim=1)
                    topk_info[down_name] = {
                        'indices': top_idx,
                        'values': top_vals
                    }
            else:
                # Handle MLP modules
                if self.connections == 'all' or (isinstance(self.connections, dict) and down_name in self.connections):
                    # Compute pruned reconstruction for connected MLPs
                    batch_size = inputs[down_name].shape[0]
                    device = inputs[down_name].device
                    k_down = self.configs[down_name].k
                    
                    # Get initial contributions for all features
                    approx_acts = self.get_initial_contributions(initial_acts, down_name)
                    
                    # Add contributions from each upstream module
                    for up_name in (self.connections[down_name].keys() if isinstance(self.connections, dict) else [n for n in self.submodule_names if n != down_name]):
                        if up_name not in inputs:
                            continue
                        
                        # Get contributions from this upstream module using full activations
                        contributions = self.get_pruned_contributions(
                            up_name, down_name, features[up_name]
                        )
                        
                        approx_acts = approx_acts + contributions
                    
                    # Apply layernorm scaling
                    approx_acts = approx_acts / layernorm_scales[down_name].unsqueeze(1)
                    
                    # Add encoder bias and subtract decoder bias contribution
                    b_enc = self.aes[down_name].encoder.bias
                    approx_acts = approx_acts + b_enc
                    
                    W_enc = self.aes[down_name].encoder.weight
                    approx_acts = approx_acts - W_enc @ self.aes[down_name].b_dec
                    
                    # Get top k features
                    top_vals, top_idx = approx_acts.topk(k_down, dim=1)  # both [batch, k_down]
                    
                    if return_topk:
                        topk_info[down_name] = {
                            'indices': top_idx,
                            'values': top_vals
                        }
                    
                    # Create sparse feature tensor for reconstruction
                    feat = t.zeros(
                        (batch_size, config.dict_size),
                        device=device,
                        dtype=self.dtype
                    )
                    feat.scatter_(
                        dim=1,
                        index=top_idx,
                        src=top_vals
                    )
                    
                    # Get reconstruction
                    reconstruction = self.aes[down_name].decode(feat)
                    reconstructions[down_name] = reconstruction
                else:
                    # Use vanilla reconstruction for non-connected MLPs
                    reconstructions[down_name] = results[down_name]
        
        if return_topk:
            return reconstructions, topk_info
        return reconstructions
    
    # def get_ce_loss(
    #     self,
    #     initial_acts: t.Tensor,  # [batch, d_in] TODO: need to keep sequence position
    #     pruned_results: Dict[str, Dict[str, t.Tensor]],  # output from pruned_forward_train
    # ) -> t.Tensor:  # [batch, d_in]
    #     """
    #     Compute residual final activation from initial acts and all reconstructions.
        
    #     Args:
    #         initial_acts: Initial activations
    #         pruned_results: Dictionary of results from pruned_forward_train
    #             For MLP modules: contains pruned reconstructions
    #             For attention modules: contains vanilla reconstructions
                
    #     Returns:
    #         resid_final_act: Sum of initial activations and all reconstructions
    #     """
        
    #     # Start with initial acts
    #     resid_final_act = initial_acts
        
    #     # Add all reconstructions
    #     for name in self.submodule_names:
    #         if name in pruned_results:
    #             resid_final_act = resid_final_act + pruned_results[name]['pruned_reconstruction']
        
    #     pass # TODO finish this

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        connections = None,
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
    steps: int
    base_lr: float = 2e-4
    lr_decay_start_proportion: float = 0.8
    log_steps: int = 100

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
        buffer: Optional[Any] = None,
        log_metrics: bool = False,
    ) -> float:
        """Single training step using pruned forward pass with FVU loss."""
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
        
        # Get forward pass results
        reconstructions = self.suite.pruned_forward(
            initial_acts=initial_acts,
            inputs=input_acts,
            layernorm_scales=layernorm_scales,
            return_topk=False
        )
        
        # Compute losses for each module
        losses = {'reconstruction_FVU': {}}
        for name, ae in self.suite.aes.items():
            if name not in input_acts:
                continue
            
            tgt = target_acts[name]
            x_hat = reconstructions[name]
            
            # Compute FVU loss
            total_variance = t.var(tgt, dim=0).sum()
            residual_variance = t.var(tgt - x_hat, dim=0).sum()
            fvu_loss = residual_variance / total_variance
            total_loss = total_loss + fvu_loss
            losses['reconstruction_FVU'][name] = fvu_loss.item()
        
        # Backward pass and optimization
        total_loss.backward()
        
        # Clip gradients and remove parallel components
        for ae in self.suite.aes.values():
            if ae.decoder.weight.grad is not None:
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