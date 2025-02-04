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

    def get_initial_contributions_train(
            self,
            initial_act: t.Tensor, # [batch, d_in]
            down_name: str, # e.g. 'mlp_5'
            down_indices: t.Tensor, # [batch, k_down + n_threshold + n_random]
            return_l1_norm: bool = False # used for logging
    ):
        W_enc_FD = self.aes[down_name].encoder.weight
        contributions_BF = initial_act @ W_enc_FD.T
        contributions_BK = contributions_BF.gather(1, down_indices)
        if return_l1_norm:
            return contributions_BK, t.norm(contributions_BK, p=1, dim=1)
        return contributions_BK

    def get_initial_contributions_test(
            self,
            initial_act: t.Tensor, # [batch, d_in]
            down_name: str, # e.g. 'mlp_5'
    ):
        """Compute initial contributions for all downstream features during test time."""
        W_enc_FD = self.aes[down_name].encoder.weight
        contributions_BF = initial_act @ W_enc_FD.T  # [batch, num_down_features]
        return contributions_BF

    def get_pruned_contributions_train(
        self,
        up_name: str,
        down_name: str,
        up_indices: t.Tensor,  # [batch, k_up]
        down_indices: t.Tensor,  # [batch, k_down + n_threshold + n_random]
        up_vals: t.Tensor,  # [batch, k_up]
        return_l1_norm: bool = False # used for logging
    ) -> t.Tensor:  # [batch, k_down + n_threshold + n_random]
        """Memory-efficient version that maintains GPU parallelism."""
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
        contributions_with_b_dec = contributions + b_dec_contrib[down_indices]
        
        if return_l1_norm:
            return contributions_with_b_dec, t.norm(contributions, p=1, dim=1)
        return contributions_with_b_dec

    def get_pruned_contributions_test(
        self,
        up_name: str,
        down_name: str,
        up_indices: t.Tensor,  # [batch, k_up]
        up_vals: t.Tensor,  # [batch, k_up]
    ) -> t.Tensor:  # [batch, num_down_features]
        """Test-time version that computes contributions for all downstream features."""
        batch_size = up_indices.shape[0]
        k_up = up_indices.shape[1]
        num_down_features = self.configs[down_name].dict_size
        device = up_indices.device
        
        # Get connection tensor for this pair of modules
        connections = self.connections[down_name][up_name]  # [num_down, C]
        
        # Get weights for the active features
        up_decoder = self.aes[up_name].decoder.weight  # [d_in, d_up]
        down_encoder = self.aes[down_name].encoder.weight  # [d_down, d_in]
        
        # Get active upstream vectors
        active_up_vectors = up_decoder[:, up_indices.reshape(-1)].T  # [batch*k_up, d_in]
        active_up_vectors = active_up_vectors.view(batch_size, k_up, -1)  # [batch, k_up, d_in]
        
        # Get all downstream vectors
        down_vectors = down_encoder  # [num_down, d_in]
        
        # Compute virtual weights between all downstream and active upstream features
        virtual_weights = einops.einsum(
            down_vectors, active_up_vectors,
            "num_down d_in, batch k_up d_in -> batch num_down k_up"
        )  # [batch, num_down, k_up]
        
        # Initialize contributions tensor
        contributions = t.zeros((batch_size, num_down_features), device=device, dtype=up_vals.dtype)
        
        # Expand connection tensor for broadcasting
        connections_expanded = connections.unsqueeze(0)  # [1, num_down, C]
        valid_mask = (connections_expanded != -1)  # [1, num_down, C]
        
        # For each valid connection, add contribution if upstream feature is active
        up_indices_expanded = up_indices.unsqueeze(1)  # [batch, 1, k_up]
        connections_match = (connections_expanded.unsqueeze(-1) == up_indices_expanded.unsqueeze(2))  # [batch, num_down, C, k_up]
        connection_mask = connections_match & valid_mask.unsqueeze(-1)  # [batch, num_down, C, k_up]
        
        # Apply mask and compute all contributions at once
        contributions = (connection_mask * virtual_weights.unsqueeze(2)).sum(dim=(2, 3))  # [batch, num_down]
        
        # Add bias terms for all features
        b_dec_contrib = down_encoder @ self.aes[up_name].b_dec  # [num_down]
        contributions = contributions + b_dec_contrib.unsqueeze(0)  # [batch, num_down]
        
        return contributions

    def pruned_forward_test(
        self,
        initial_acts: t.Tensor,
        inputs: Dict[str, t.Tensor],
        layernorm_scales: Dict[str, t.Tensor],
        return_topk: bool = False
    ) -> Union[Dict[str, t.Tensor], Tuple[Dict[str, t.Tensor], Dict[str, Dict[str, t.Tensor]]]]:
        """
        Test-time forward pass that computes activations for all features, then applies TopK.
        
        Args:
            initial_acts: Initial activations
            inputs: Dictionary mapping submodule names to input tensors
            layernorm_scales: Dictionary of layernorm scales
            return_topk: Whether to return top-k feature indices and values
            
        Returns:
            If return_topk is False:
                Dictionary mapping module names to pruned reconstructions
            If return_topk is True:
                Tuple of (reconstructions_dict, topk_info_dict) where topk_info_dict has structure:
                {module_name: {
                    'indices': tensor of top-k feature indices [batch, k],
                    'values': tensor of top-k feature values [batch, k]
                }}
        """
        reconstructions = {}
        topk_info = {}
        
        # Store pruned upstream info for progressive computation
        pruned_upstream_info = {}
        
        for down_name, config in self.configs.items():
            if down_name not in inputs:
                continue
                
            if 'attn' in down_name:
                # For attention modules, use vanilla forward pass
                feat = self.aes[down_name].encode(inputs[down_name])
                recon = self.aes[down_name].decode(feat)
                reconstructions[down_name] = recon
                
                if return_topk:
                    # Get top k features for upstream usage
                    k = self.configs[down_name].k
                    top_vals, top_idx = feat.topk(k, dim=1)
                    topk_info[down_name] = {
                        'indices': top_idx,
                        'values': top_vals
                    }
                    pruned_upstream_info[down_name] = topk_info[down_name]
                continue
                
            if down_name not in self.connections:
                continue
                
            batch_size = inputs[down_name].shape[0]
            device = inputs[down_name].device
            k_down = self.configs[down_name].k
            
            # Get initial contributions for all features
            approx_acts = self.get_initial_contributions_test(initial_acts, down_name)
            
            # Add contributions from each upstream module
            for up_name in self.connections[down_name].keys():
                if up_name not in inputs:
                    continue
                    
                # Get upstream values (only use top k)
                k_up = self.configs[up_name].k
                if up_name not in pruned_upstream_info:
                    # First layer or attention module - use vanilla forward
                    feat = self.aes[up_name].encode(inputs[up_name])
                    top_vals, top_idx = feat.topk(k_up, dim=1)
                    up_indices = top_idx
                    up_vals = top_vals
                else:
                    # Use previously computed pruned values
                    up_indices = pruned_upstream_info[up_name]['indices']
                    up_vals = pruned_upstream_info[up_name]['values']
                
                # Get contributions from this upstream module
                contributions = self.get_pruned_contributions_test(
                    up_name, down_name, up_indices, up_vals
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
                pruned_upstream_info[down_name] = topk_info[down_name]
            
            # Create sparse feature tensor for reconstruction
            features = t.zeros(
                (batch_size, config.dict_size),
                device=device,
                dtype=self.dtype
            )
            features.scatter_(
                dim=1,
                index=top_idx,
                src=top_vals
            )
            
            # Get reconstruction
            reconstruction = self.aes[down_name].decode(features)
            reconstructions[down_name] = reconstruction
        
        if return_topk:
            return reconstructions, topk_info
        return reconstructions

    def vanilla_forward(
        self,
        inputs: Dict[str, t.Tensor],
        return_features: bool = False,
        return_topk: bool = False,
        n_threshold: int = 0
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
        n_threshold: int = 0,
        n_random: int = 0
    ) -> Tuple[Dict[str, Dict[str, t.Tensor]], Dict[str, float]]:
        """
        Run modified forward pass that computes:
        - Top k features
        - Additional n_threshold features
        - n_random randomly selected features
        
        Uses pruned activations of upstream modules when computing downstream activations.
        For attention modules, returns vanilla reconstructions.
        
        Args:
            initial_acts: Initial activations
            inputs: Dictionary mapping submodule names to input tensors
            layernorm_scales: Dictionary of layernorm scales
            n_threshold: Number of additional threshold features to compute
            n_random: Number of random features to compute
            
        Returns:
            Tuple of (outputs_dict, metrics_dict) where:
            outputs_dict: {module_name: {
                'topk': tensor of top-k feature activations,
                'threshold': tensor of threshold feature activations,
                'random': tensor of random feature activations,
                'all_indices': tensor of concatenated indices [batch, k_down + n_threshold + n_random],
                'pruned_reconstruction': reconstruction from pruned features (or vanilla for attention)
            }}
            metrics_dict: Dictionary of L1 norms for logging
        """
        # First get vanilla features and topk info (just for indices)
        results, features, topk_info = self.vanilla_forward(
            inputs, return_features=True, return_topk=True, n_threshold=n_threshold
        )
        
        outputs = {}
        metrics = {}
        # Store pruned upstream info
        pruned_upstream_info = {}
        
        # Now compute features for each module
        for down_name, config in self.configs.items():
            if down_name not in inputs:
                continue
                
            if 'attn' in down_name:
                # For attention modules, just return vanilla reconstruction and empty values
                k_down = self.configs[down_name].k
                batch_size = inputs[down_name].shape[0]
                device = inputs[down_name].device
                
                # For attention modules, just return vanilla reconstruction
                module_outputs = {
                    'pruned_reconstruction': results[down_name]  # Use vanilla reconstruction
                }
                outputs[down_name] = module_outputs
                # Store vanilla topk info for use as upstream module
                pruned_upstream_info[down_name] = {
                    'all_indices': topk_info[down_name][0],  # [batch, k]
                    'topk': topk_info[down_name][1]  # [batch, k]
                }
                continue
                
            if down_name not in self.connections:
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
            
            # Get initial contributions and their L1 norms
            approx_acts, initial_l1_norms = self.get_initial_contributions_train(
                initial_acts, down_name, down_idxs_all, return_l1_norm=True
            )
            metrics[f'contributions/{down_name}/from_initial'] = initial_l1_norms.mean().item()
            
            # Get contributions from each upstream module
            for up_name in self.connections[down_name].keys():
                if up_name not in inputs:
                    continue
                
                # Get upstream values - ensure we only use top k for both vanilla and pruned
                k_up = self.configs[up_name].k
                # For attention modules or vanilla values, use vanilla topk info
                # For MLP modules with pruned results, get the reranked top k
                if up_name not in pruned_upstream_info or 'attn' in up_name:
                    up_idxs = topk_info[up_name][0][:, :k_up]  # [batch, k_up]
                    up_vals = topk_info[up_name][1][:, :k_up]  # [batch, k_up]
                else:
                    # Concatenate topk and threshold from pruned results
                    up_acts = t.cat([
                        pruned_upstream_info[up_name]['topk'],
                        pruned_upstream_info[up_name]['threshold']
                    ], dim=1)  # [batch, k_up + n_threshold]
                    up_indices = t.cat([
                        pruned_upstream_info[up_name]['all_indices'][:, :k_up],
                        pruned_upstream_info[up_name]['all_indices'][:, k_up:k_up + n_threshold]
                    ], dim=1)  # [batch, k_up + n_threshold]
                    
                    # Get top k from these
                    top_vals, top_idx = up_acts.topk(k_up, dim=1)  # both [batch, k_up]
                    up_idxs = t.gather(up_indices, 1, top_idx)  # [batch, k_up]
                    up_vals = top_vals  # [batch, k_up]
                
                # Get contributions and L1 norms from upstream module
                contributions, up_l1_norms = self.get_pruned_contributions_train(
                    up_name, down_name, up_idxs, down_idxs_all, up_vals, return_l1_norm=True
                )
                metrics[f'contributions/{down_name}/from_{up_name}'] = up_l1_norms.mean().item()
                
                approx_acts = approx_acts + contributions
            
            # Apply layernorm scaling
            approx_acts = approx_acts / layernorm_scales[down_name].unsqueeze(1)
            
            # Add encoder bias and subtract decoder bias contribution
            b_enc = self.aes[down_name].encoder.bias[down_idxs_all]
            approx_acts = approx_acts + b_enc
            
            W_enc = self.aes[down_name].encoder.weight[down_idxs_all]
            approx_acts = approx_acts - W_enc @ self.aes[down_name].b_dec
            
            # Split results and get pruned reconstruction
            topk_threshold_acts = t.cat([
                approx_acts[:, :k_down],
                approx_acts[:, k_down:k_down + n_threshold]
            ], dim=1)  # [batch, k_down + n_threshold]
            
            topk_threshold_indices = t.cat([
                down_idxs_topk,
                down_idxs_threshold
            ], dim=1)  # [batch, k_down + n_threshold]
            
            # Get top k features from the pruned activations
            pruned_top_vals, pruned_top_idx = topk_threshold_acts.topk(k_down, dim=1)  # both [batch, k_down]
            
            # Get the actual feature indices these correspond to
            pruned_top_indices = t.gather(topk_threshold_indices, 1, pruned_top_idx)  # [batch, k_down]
            
            # Create sparse feature tensor for reconstruction
            pruned_features = t.zeros(
                (batch_size, config.dict_size),
                device=device,
                dtype=self.dtype
            )
            pruned_features.scatter_(
                dim=1,
                index=pruned_top_indices,
                src=pruned_top_vals
            )
            
            # Get reconstruction from pruned features
            pruned_reconstruction = self.aes[down_name].decode(pruned_features)
            
            module_outputs = {
                'topk': approx_acts[:, :k_down],
                'threshold': approx_acts[:, k_down:k_down + n_threshold],
                'random': approx_acts[:, k_down + n_threshold:],
                'all_indices': down_idxs_all,  # [batch, k_down + n_threshold + n_random]
                'pruned_reconstruction': pruned_reconstruction
            }
            
            # Store outputs both for return and for use as upstream values
            outputs[down_name] = module_outputs
            pruned_upstream_info[down_name] = module_outputs
        
        return outputs, metrics
    
    def get_ce_loss(
        self,
        initial_acts: t.Tensor,  # [batch, d_in] TODO: need to keep sequence position
        pruned_results: Dict[str, Dict[str, t.Tensor]],  # output from pruned_forward_train
    ) -> t.Tensor:  # [batch, d_in]
        """
        Compute residual final activation from initial acts and all reconstructions.
        
        Args:
            initial_acts: Initial activations
            pruned_results: Dictionary of results from pruned_forward_train
                For MLP modules: contains pruned reconstructions
                For attention modules: contains vanilla reconstructions
                
        Returns:
            resid_final_act: Sum of initial activations and all reconstructions
        """
        
        # Start with initial acts
        resid_final_act = initial_acts
        
        # Add all reconstructions
        for name in self.submodule_names:
            if name in pruned_results:
                resid_final_act = resid_final_act + pruned_results[name]['pruned_reconstruction']
        
        pass # TODO finish this


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
    steps: int
    random_loss_coeff: float = 1.0
    n_threshold: int = 100
    n_random: int = 100
    log_steps: int = 100
    test_batch_size: int = 8
    base_lr: float = 2e-4
    lr_decay_start_proportion: float = 0.8
    use_vanilla_training: bool = False  # New flag to control training mode

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

    def compute_test_metrics(
        self,
        initial_acts: t.Tensor,
        input_acts: Dict[str, t.Tensor],
        target_acts: Dict[str, t.Tensor],
        layernorm_scales: Dict[str, t.Tensor],
    ) -> Dict[str, float]:
        """Compute test-time metrics using both train and test forward passes."""
        metrics = {}
        
        with t.no_grad():
            # Run test forward pass
            test_reconstructions, test_topk_info = self.suite.pruned_forward_test(
                initial_acts=initial_acts,
                inputs=input_acts,
                layernorm_scales=layernorm_scales,
                return_topk=True
            )
            
            # Run train forward pass for comparison
            train_results = self.suite.pruned_forward_train(
                initial_acts=initial_acts,
                inputs=input_acts,
                layernorm_scales=layernorm_scales,
                n_threshold=self.config.n_threshold,
                n_random=self.config.n_random
            )
            
            # Compute metrics for each module
            for name, ae in self.suite.aes.items():
                if name not in input_acts or 'attn' in name:
                    continue
                
                tgt = target_acts[name]
                
                # Compute test FVU using test reconstruction
                test_recon = test_reconstructions[name]
                total_variance = t.var(tgt, dim=0).sum()
                test_residual_variance = t.var(tgt - test_recon, dim=0).sum()
                metrics[f'pruned FVU test/{name}'] = (test_residual_variance / total_variance).item()
                
                # Compare train vs test activations
                test_acts = test_topk_info[name]['values']  # [batch, k]
                train_acts = train_results[name]['topk']  # [batch, k]
                
                # Ensure we're comparing the same features
                test_indices = test_topk_info[name]['indices']
                train_indices = train_results[name]['all_indices'][:, :ae.k]
                
                # Sort both sets of activations by feature index for fair comparison
                test_sort_idx = test_indices.argsort(dim=1)
                train_sort_idx = train_indices.argsort(dim=1)
                
                test_acts = t.gather(test_acts, 1, test_sort_idx)
                train_acts = t.gather(train_acts, 1, train_sort_idx)
                
                # Compute activation FVU
                act_fvu = (train_acts - test_acts).pow(2).sum() / test_acts.pow(2).sum()
                metrics[f'train_vs_test_act_FVU/{name}'] = act_fvu.item()
        
        return metrics

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
        """
        Single training step using either vanilla or pruned forward pass with FVU loss.
        
        Args:
            step: Current training step
            initial_acts: Initial activations
            input_acts: Dictionary mapping submodule names to input tensors
            target_acts: Dictionary mapping submodule names to target tensors
            layernorm_scales: Dictionary of layernorm scales
            buffer: Optional buffer for test metrics
            log_metrics: Whether to log metrics
            
        Returns:
            Total loss value for this step
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
            'FVU': {},
            'random_penalty': {},
        }
        
        if self.config.use_vanilla_training:
            # Get vanilla forward pass results for all modules
            reconstructions = self.suite.vanilla_forward(inputs=input_acts)
        else:
            # Get pruned forward pass results for MLP modules
            pruned_results, l1_norm_metrics = self.suite.pruned_forward_train(
                initial_acts=initial_acts,
                inputs=input_acts,
                layernorm_scales=layernorm_scales,
                n_threshold=self.config.n_threshold,
                n_random=self.config.n_random
            )
            # Also get vanilla forward pass results for attention modules
            vanilla_results = self.suite.vanilla_forward(inputs=input_acts)
        
        # Compute losses for each module
        for name, ae in self.suite.aes.items():
            if name not in input_acts:
                continue
            
            tgt = target_acts[name]
            
            if self.config.use_vanilla_training:
                # Use vanilla results for all modules
                x_hat = reconstructions[name]
                # Compute FVU loss
                total_variance = t.var(tgt, dim=0).sum()
                residual_variance = t.var(tgt - x_hat, dim=0).sum()
                fvu_loss = residual_variance / total_variance
                total_loss = total_loss + fvu_loss
                losses['FVU'][name] = fvu_loss.item()
                
            else:
                # Original pruned training logic
                if 'mlp' in name: # and int(name.split('_')[-1]) < 2 and name != 'mlp_0':
                    module_results = pruned_results[name]
                    x_hat = module_results['pruned_reconstruction']
                    
                    # Compute FVU loss
                    total_variance = t.var(tgt, dim=0).sum()
                    residual_variance = t.var(tgt - x_hat, dim=0).sum()
                    fvu_loss = residual_variance / total_variance
                    total_loss = total_loss + fvu_loss
                    losses['FVU'][name] = fvu_loss.item()
                    
                    # Random activation penalty
                    topk_threshold_acts = t.cat([
                        module_results['topk'],
                        module_results['threshold']
                    ], dim=1)
                    min_top_acts = topk_threshold_acts.min(dim=1, keepdim=True).values
                    
                    # Penalize random features that exceed this minimum
                    random_penalty = t.relu(module_results['random'] - min_top_acts)
                    random_loss = random_penalty.sum()
                    
                    total_loss = total_loss + self.config.random_loss_coeff * random_loss
                    losses['random_penalty'][name] = random_loss.item()
                    
                elif ('attn' in name): # and int(name.split('_')[-1]) < 2) or name == 'mlp_0':
                    x_hat = vanilla_results[name]
                    
                    # Compute FVU loss
                    total_variance = t.var(tgt, dim=0).sum()
                    residual_variance = t.var(tgt - x_hat, dim=0).sum()
                    fvu_loss = residual_variance / total_variance
                    total_loss = total_loss + fvu_loss
                    losses['FVU'][name] = fvu_loss.item()
                
                else:
                    continue
        
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
            
            # Add L1 norm metrics from pruned forward pass
            if not self.config.use_vanilla_training:
                log_dict.update(l1_norm_metrics)
            
            # Add test metrics every log_steps
            if step % self.config.log_steps == 0 and buffer is not None:
                # Get a test batch from buffer
                orig_batch_size = buffer.out_batch_size
                buffer.out_batch_size = self.config.test_batch_size
                test_batch = buffer.get_batch()
                buffer.out_batch_size = orig_batch_size
                
                # Compute and add test metrics
                test_metrics = self.compute_test_metrics(
                    initial_acts=test_batch['initial_acts'],
                    input_acts=test_batch['input_acts'],
                    target_acts=test_batch['target_acts'],
                    layernorm_scales=test_batch['layernorm_scales']
                )
                log_dict.update(test_metrics)
            
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