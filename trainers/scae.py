import torch as t
import torch.nn as nn
import einops
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass
import numpy as np
import json 

from trainers.top_k import AutoEncoderTopK
from transformer_lens import ActivationCache


class SCAESuite(nn.Module):
    """A suite of Sparsely-Connected TopK Autoencoders"""
    
    def __init__(
        self,
        model,
        k: int,
        n_features: int,
        connections: Optional[Union[Dict[str, Dict[str, t.Tensor]], str]] = None,
        dtype: t.dtype = t.float32,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: TransformerLens model
            k: Number of features to select for each autoencoder
            n_features: Dictionary size for each autoencoder
            connections: Optional dictionary specifying sparse connections
            dtype: Data type for the autoencoders
            device: Device to place the autoencoders on
        """
        super().__init__()
        self.model = model
        self.device = device or ('cuda' if t.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.is_pretrained = False
        self.k = k
        self.n_features = n_features
        
        n_layers = model.cfg.n_layers
        self.submodule_names = [f'{type_}_{i}' for i in range(n_layers) for type_ in ['attn', 'mlp']]
        
        # Initialize autoencoders
        self.aes = nn.ModuleDict({
            name: AutoEncoderTopK(
                model.cfg.d_model,
                n_features,
                k
            ).to(device=self.device, dtype=self.dtype)
            for name in self.submodule_names
        })
        
        # Initialize and validate connections
        self.connections = connections
        self.connection_masks = {}
        if isinstance(connections, dict):
            processed_connections = {}
            for down_name, up_dict in connections.items():
                processed_connections[down_name] = {}
                self.connection_masks[down_name] = {}
                for up_name, conn_tensor in up_dict.items():
                    # Verify indices are valid
                    valid_indices = (conn_tensor >= -1) & (conn_tensor < n_features)
                    assert valid_indices.all(), \
                        f"Invalid indices in connection tensor for {down_name}->{up_name}. All values must be -1 or valid feature indices."
                    
                    processed_connections[down_name][up_name] = conn_tensor.to(device=self.device, dtype=t.long)
                    
                    # Precompute connection mask
                    connection_mask = t.zeros(n_features, n_features, device=self.device, dtype=t.bool)
                    valid_connections = conn_tensor != -1  # [num_down, C]
                    
                    # For each downstream feature, scatter its valid connections
                    for i in range(n_features):
                        valid_mask = valid_connections[i]  # [C]
                        valid_targets = conn_tensor[i][valid_mask]  # Get valid target indices
                        connection_mask[i].scatter_(0, valid_targets, t.ones(len(valid_targets), device=self.device, dtype=t.bool))
                    
                    self.connection_masks[down_name][up_name] = connection_mask
            self.connections = processed_connections
        elif connections is not None and connections != "all":
            raise ValueError("connections must be either None, 'all', or a dictionary of connection tensors")

    def vanilla_forward(
        self,
        cache: Dict[str, t.Tensor],
    ) -> Dict[str, t.Tensor]:
        """Forward pass that outputs reconstructions.
        
        Args:
            cache: Dictionary mapping hook points to activation tensors
            
        Returns:
            Dictionary of reconstructions matching input dimensions
        """
        results = {}
        
        for layer in range(len(self.submodule_names) // 2):  # Divide by 2 since we have both attn and mlp for each layer
            # Attention
            attn_name = f'attn_{layer}'
            attn_input = cache[f'blocks.{layer}.ln1.hook_normalized']
            feat = self.aes[attn_name].encode(attn_input)
            recon = self.aes[attn_name].decode(feat)
            results[attn_name] = recon
            
            # MLP
            mlp_name = f'mlp_{layer}'
            mlp_input = cache[f'blocks.{layer}.ln2.hook_normalized']
            feat = self.aes[mlp_name].encode(mlp_input)
            recon = self.aes[mlp_name].decode(feat)
            results[mlp_name] = recon
        
        return results

    def get_initial_contribs_mlp(
        self,
        cache: ActivationCache,
        down_name: str,  # e.g. 'mlp_0'
    ) -> t.Tensor:  # [batch, seq, n_features]
        """Compute initial contributions for MLP autoencoder from residual stream.
        
        Args:
            cache: TransformerLens activation cache
            down_name: Name of downstream autoencoder (must be MLP type)
            
        Returns:
            Tensor of initial contributions to features
        """
        initial_act = cache['blocks.0.hook_resid_pre']  # [batch, seq, d_model]
        W_enc = self.aes[down_name].encoder.weight  # [n_features, d_model]
        
        return initial_act @ W_enc.T  # [batch, seq, n_features]

    def get_pruned_contribs_mlp(
        self,
        cache: ActivationCache,
        up_name: str,
        down_name: str,
        up_facts: t.Tensor,
    ) -> t.Tensor:
        if self.connections is None:
            up_decoder = self.aes[up_name].decoder.weight
            down_encoder = self.aes[down_name].encoder.weight
            virtual_weights = down_encoder @ up_decoder
            contributions = up_facts @ virtual_weights.T
        else:
            up_decoder = self.aes[up_name].decoder.weight
            down_encoder = self.aes[down_name].encoder.weight
            connection_mask = self.connection_masks[down_name][up_name]
            
            # Get non-zero indices from the connection mask
            indices = connection_mask.nonzero(as_tuple=False).t()  # [2, M]
            i_indices, j_indices = indices[0], indices[1]
            
            # Compute only the necessary elements of virtual_weights
            down_selected = down_encoder[i_indices]  # [M, d_model]
            up_selected = up_decoder[:, j_indices]   # [d_model, M]
            values = (down_selected * up_selected.t()).sum(dim=1)  # [M]
            
            # Gather relevant up_facts and compute contributions via scatter-add
            up_facts_selected = up_facts[:, :, j_indices]  # [B, S, M]
            scaled = up_facts_selected * values[None, None, :]  # [B, S, M]
            
            contributions = t.zeros(
                up_facts.shape[0], 
                up_facts.shape[1], 
                down_encoder.shape[0], 
                device=up_facts.device, 
                dtype=up_facts.dtype
            )
            contributions.scatter_add_(-1, i_indices[None, None, :].expand_as(scaled), scaled)
        
        return contributions
    
    def get_initial_contribs_attn(
            self,
            cache: ActivationCache,
            down_name: str,  # e.g. 'attn_1'
        ) -> t.Tensor:  # [batch, qpos, n_down_features]
            """Compute initial contributions for attention autoencoder from residual stream."""
            from einops import einsum
            
            layer = int(down_name.split('_')[1])
            initial_act = cache['blocks.0.hook_resid_pre']
            
            # Get attention weights and combine W_O and W_V
            W_O = self.model.W_O[layer]  # [n_heads, d_head, d_out]
            W_V = self.model.W_V[layer]  # [n_heads, d_model, d_head]
            
            # Break down the einsum operations and clean up intermediates
            W_OV = einsum(W_O, W_V, "n_heads d_head d_out, n_heads d_in d_head -> n_heads d_in d_out")
            del W_O, W_V
            t.cuda.empty_cache()
            
            down_encoder = self.aes[down_name].encoder.weight  # [f_down, d_out]
            initial_contrib_pre_moving = einsum(initial_act, W_OV, down_encoder,
                                            "batch pos d_in, n_heads d_in d_out, f_down d_out -> batch pos n_heads f_down")
            del W_OV
            t.cuda.empty_cache()
            
            # Mix between positions using attention pattern
            probs = cache[f'blocks.{layer}.attn.hook_pattern']  # [batch, n_heads, qpos, kpos]
            initial_contrib = einsum(probs, initial_contrib_pre_moving,
                                "batch n_heads qpos kpos, batch kpos n_heads f_down -> batch qpos f_down")
            del initial_contrib_pre_moving
            t.cuda.empty_cache()
            
            return initial_contrib

    def get_pruned_contribs_attn(
        self,
        cache: ActivationCache,
        up_name: str,
        down_name: str,
        up_facts: t.Tensor,
    ) -> t.Tensor:
        from einops import einsum
        print(f"\nStarting get_pruned_contribs_attn for up_name={up_name}, down_name={down_name}")
        print(f"up_facts shape: {up_facts.shape}")  # [batch, seq, n_up_features]
        
        layer = int(down_name.split('_')[1])
        
        # Compute necessary components (W_OV, temp, etc.)
        W_O = self.model.W_O[layer]  # [n_heads, d_head, d_out]
        W_V = self.model.W_V[layer]  # [n_heads, d_model, d_head]
        print(f"W_O shape: {W_O.shape}")
        print(f"W_V shape: {W_V.shape}")
        
        W_OV = einsum(W_O, W_V, "n_heads d_head d_out, n_heads d_in d_head -> n_heads d_in d_out")
        print(f"W_OV shape: {W_OV.shape}")  # [n_heads, d_in, d_out]
        
        temp = einsum(self.aes[down_name].encoder.weight, W_OV, "f_down d_out, n_heads d_in d_out -> n_heads f_down d_in")
        print(f"temp shape: {temp.shape}")  # [n_heads, f_down, d_in]
        
        up_decoder = self.aes[up_name].decoder.weight  # [d_in, f_up]
        print(f"up_decoder shape: {up_decoder.shape}")
        
        connection_mask = self.connection_masks[down_name][up_name]  # [f_down, f_up]
        print(f"connection_mask shape: {connection_mask.shape}")
        
        indices = connection_mask.nonzero(as_tuple=False)  # [M, 2]
        i_indices, j_indices = indices[:, 0], indices[:, 1]
        print(f"i_indices shape: {i_indices.shape}, j_indices shape: {j_indices.shape}")
        
        # Compute virtual_weights values for each head and mask entry
        selected_temp = temp[:, i_indices, :]  # [n_heads, M, d_in]
        selected_up = up_decoder[:, j_indices]  # [d_in, M]
        print(f"selected_temp shape: {selected_temp.shape}")
        print(f"selected_up shape: {selected_up.shape}")
        
        values = einsum(selected_temp, selected_up, "n_heads m d_in, d_in m -> n_heads m")
        print(f"values shape: {values.shape}")  # [n_heads, M]
        
        # Gather up_facts and scatter-add contributions per head
        up_facts_selected = up_facts[:, :, j_indices]  # [B, S, M]
        print(f"up_facts_selected shape: {up_facts_selected.shape}")
        
        n_heads, m = values.shape
        contributions = t.zeros(up_facts.shape[0], up_facts.shape[1], n_heads, temp.shape[1], 
                        device=up_facts.device, dtype=up_facts.dtype)
        print(f"contributions shape before scatter_add: {contributions.shape}")
        
        for h in range(n_heads):
            scaled = up_facts_selected * values[h][None, None, :]  # [B, S, M]
            print(f"scaled shape for head {h}: {scaled.shape}")
            contributions[:, :, h].scatter_add_(-1, i_indices[None, None, :].expand_as(scaled), scaled)
        
        print(f"contributions shape after scatter_add: {contributions.shape}")
        
        # Apply attention pattern mixing
        probs = cache[f'blocks.{layer}.attn.hook_pattern']  # [batch, n_heads, qpos, kpos]
        print(f"probs shape: {probs.shape}")
        
        final_contribs = einsum(probs, contributions, "b h q k, b k h f -> b q f")
        print(f"final_contribs shape: {final_contribs.shape}")
        
        return final_contribs
    
    def forward_pruned(
            self,
            cache: ActivationCache,
        ) -> Dict[str, t.Tensor]:
            """Forward pass computing sparse reconstructions."""
            reconstructions = {}
            pruned_features = {}  # Store pruned features for each module
            
            def parse_name(name: str) -> Tuple[int, str]:
                type_, layer = name.split('_')
                return int(layer), type_
            
            # Pre-allocate tensors that will be reused
            first_cache_tensor = next(iter(cache.values()))
            batch_size, seq_len = first_cache_tensor.shape[:2]
            device, dtype = first_cache_tensor.device, first_cache_tensor.dtype
            
            feat_buffer = t.zeros(
                (batch_size, seq_len, self.n_features),
                device=device,
                dtype=dtype
            )
            
            for down_name in self.submodule_names:
                print("down_name", down_name)
                down_layer, down_type = parse_name(down_name)
                
                # Initialize approx_acts based on module type
                if down_type == 'attn':
                    approx_acts = self.get_initial_contribs_attn(cache, down_name)
                else:  # mlp
                    approx_acts = self.get_initial_contribs_mlp(cache, down_name)
                
                # Initialize accumulated upstream bias
                upstream_bias = t.zeros(self.model.cfg.d_model, device=device, dtype=dtype)
                
                for up_name in self.submodule_names:
                    up_layer, up_type = parse_name(up_name)
                    
                    if up_layer > down_layer or (up_layer == down_layer and (up_type == 'mlp' or up_name == down_name)):
                        continue
                    print("up_name", up_name)
                        
                    if self.connections is None or (isinstance(self.connections, dict) 
                        and down_name in self.connections 
                        and up_name in self.connections[down_name]):
                        
                        # Use the pruned features from previous modules
                        up_feats = pruned_features[up_name]  # These are already the top-k selected features
                        
                        # Update upstream bias inplace
                        upstream_bias.add_(self.aes[up_name].b_dec)
                        
                        # Get contributions based on module type
                        if down_type == 'attn':
                            contributions = self.get_pruned_contribs_attn(cache, up_name, down_name, up_feats)
                        else:  # mlp
                            contributions = self.get_pruned_contribs_mlp(cache, up_name, down_name, up_feats)
                        
                        # Add contributions inplace and free memory
                        approx_acts.add_(contributions)
                        del contributions
                        t.cuda.empty_cache()
                
                # Apply layernorm scale inplace
                ln_name = 'ln1' if down_type == 'attn' else 'ln2'
                ln_scale = cache[f'blocks.{down_layer}.{ln_name}.hook_scale']
                while ln_scale.dim() < approx_acts.dim():
                    ln_scale = ln_scale.unsqueeze(-1)
                approx_acts.div_(ln_scale)
                
                # Add biases inplace
                bias = self.aes[down_name].encoder.bias
                while bias.dim() < approx_acts.dim():
                    bias = bias.unsqueeze(0)
                approx_acts.add_(bias)
                
                projected_bias = self.aes[down_name].encoder.weight @ upstream_bias
                while projected_bias.dim() < approx_acts.dim():
                    projected_bias = projected_bias.unsqueeze(0)
                approx_acts.add_(projected_bias)
                
                # Get top k features
                top_vals, top_idx = approx_acts.topk(self.k, dim=-1)
                
                # Reuse feat_buffer by zeroing it
                feat_buffer.zero_()
                
                # Scatter top k values into buffer
                feat_buffer.scatter_(-1, top_idx, top_vals)
                
                # Store pruned features for use in later modules
                pruned_features[down_name] = feat_buffer.clone()  # Need to clone since feat_buffer will be reused
                
                # Decode and store reconstruction
                reconstructions[down_name] = self.aes[down_name].decode(feat_buffer)
                
                # Clean up
                del top_vals, top_idx
                t.cuda.empty_cache()
            
            return reconstructions

    def upload_to_hf(self, repo_id: str):
        """
        Upload the model to HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repository ID to upload to
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub package is required to upload models. "
                "Install with: pip install huggingface_hub"
            )

        # Create config
        config = {
            "k": self.k,
            "n_features": self.n_features,
            "connections": self.connections
        }
        
        # Save config
        import json
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save config
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Save state dict
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")
            t.save(self.state_dict(), checkpoint_path)
            
            # Upload files
            api = HfApi()
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                repo_type="model"
            )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        model,
        device: Optional[str] = None,
        dtype: t.dtype = t.float32
    ) -> "SCAESuite":
        """
        Load a pretrained SCAESuite from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID containing the saved model
            model: TransformerLens model
            device: Device to load the model on
            dtype: Data type for model parameters
            
        Returns:
            Initialized SCAESuite with pretrained weights
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
        
        # Initialize suite
        suite = cls(
            model=model,
            k=config["k"],
            n_features=config["n_features"],
            connections=config["connections"],
            dtype=dtype,
            device=device
        )
        
        # Download and load state dict
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pt")
        state_dict = t.load(checkpoint_path, map_location='cpu')
        
        # Load state dict into suite
        missing_keys, unexpected_keys = suite.load_state_dict(state_dict)
        
        if len(missing_keys) > 0:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        suite.is_pretrained = True
        return suite