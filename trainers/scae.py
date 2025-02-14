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
        
        # Get module names from model
        n_layers = model.cfg.n_layers
        self.submodule_names = [f'attn_{i}' for i in range(n_layers)] + [f'mlp_{i}' for i in range(n_layers)]
        
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
        down_name: str,  # e.g. 'mlp_0'
        up_facts: t.Tensor,  # [batch, seq, n_up_features]
    ) -> t.Tensor:  # [batch, seq, n_down_features]
        """Compute contributions from upstream to downstream MLP autoencoder.
        
        Args:
            cache: TransformerLens activation cache
            up_name: Name of upstream autoencoder
            down_name: Name of downstream autoencoder (must be MLP type)
            up_facts: Upstream features/activations
            
        Returns:
            Tensor of contributions from upstream to downstream features
        """
        if self.connections is None:
            up_decoder = self.aes[up_name].decoder.weight  # [d_model, f_up]
            down_encoder = self.aes[down_name].encoder.weight  # [f_down, d_model]
            virtual_weights = down_encoder @ up_decoder  # [f_down, f_up]
            contributions = up_facts @ virtual_weights.T
            
        else:
            up_decoder = self.aes[up_name].decoder.weight
            down_encoder = self.aes[down_name].encoder.weight
            virtual_weights = down_encoder @ up_decoder
            connection_mask = self.connection_masks[down_name][up_name]
            masked_weights = virtual_weights * connection_mask
            contributions = up_facts @ masked_weights.T
        
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
        down_name: str,  # e.g. 'attn_1'
        up_facts: t.Tensor,  # [batch, seq, n_up_features]
    ) -> t.Tensor:  # [batch, qpos, n_down_features]
        """Compute contributions from upstream to downstream attention autoencoder."""
        from einops import einsum
        
        layer = int(down_name.split('_')[1])
        
        # Get attention weights
        W_O = self.model.W_O[layer]  # [n_heads, d_head, d_out]
        W_V = self.model.W_V[layer]  # [n_heads, d_model, d_head]
        
        # Break down the computation into smaller steps with cleanup
        W_OV = einsum(W_O, W_V, "n_heads d_head d_out, n_heads d_in d_head -> n_heads d_in d_out")
        del W_O, W_V
        t.cuda.empty_cache()
        
        up_decoder = self.aes[up_name].decoder.weight  # [d_in, f_up]
        down_encoder = self.aes[down_name].encoder.weight  # [f_down, d_out]
        
        # Break down the three-way einsum into two steps
        temp = einsum(down_encoder, W_OV, "f_down d_out, n_heads d_in d_out -> n_heads f_down d_in")
        del W_OV
        t.cuda.empty_cache()
        
        virtual_weights = einsum(temp, up_decoder, "n_heads f_down d_in, d_in f_up -> n_heads f_down f_up")
        del temp
        t.cuda.empty_cache()
        
        # Apply connection mask if using sparse connections
        if self.connections is not None:
            connection_mask = self.connection_masks[down_name][up_name]
            virtual_weights *= connection_mask.unsqueeze(0)
        
        contribs_pre_moving = einsum(virtual_weights, up_facts, 
                                    "n_heads f_down f_up, batch kpos f_up -> batch kpos n_heads f_down")
        del virtual_weights
        t.cuda.empty_cache()
        
        # Mix between positions using attention pattern
        probs = cache[f'blocks.{layer}.attn.hook_pattern']  # [batch, n_heads, qpos, kpos]
        contribs = einsum(probs, contribs_pre_moving,
                        "batch n_heads qpos kpos, batch kpos n_heads f_down -> batch qpos f_down")
        del contribs_pre_moving
        t.cuda.empty_cache()
        
        return contribs
    
    def forward_pruned(
            self,
            cache: ActivationCache,
        ) -> Dict[str, t.Tensor]:
            """Forward pass computing sparse reconstructions."""
            reconstructions = {}
            
            # Helper to extract layer number and type from module name
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
                    
                    if up_layer > down_layer or (up_layer == down_layer and up_type == 'mlp'):
                        continue
                        
                    if self.connections is None or (isinstance(self.connections, dict) 
                        and down_name in self.connections 
                        and up_name in self.connections[down_name]):
                        
                        # Get cached activation and encode
                        cache_key = f'blocks.{up_layer}.{"ln1" if up_type == "attn" else "ln2"}.hook_normalized'
                        up_feats = self.aes[up_name].encode(cache[cache_key])
                        
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
                        del up_feats
                        t.cuda.empty_cache()
                
                # Apply layernorm scale inplace
                ln_name = 'ln1' if down_type == 'attn' else 'ln2'
                ln_scale = cache[f'blocks.{down_layer}.{ln_name}.hook_scale']
                # Ensure ln_scale has the right shape for broadcasting
                while ln_scale.dim() < approx_acts.dim():
                    ln_scale = ln_scale.unsqueeze(-1)
                approx_acts.div_(ln_scale)
                
                # Add biases inplace
                bias = self.aes[down_name].encoder.bias
                # Ensure bias has the right shape for broadcasting
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