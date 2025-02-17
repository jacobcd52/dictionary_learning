import torch as t
import torch.nn as nn
import einops
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass
import numpy as np
import json
from einops import einsum

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
                    
                    # Vectorized connection mask creation (3B)
                    valid_connections = conn_tensor != -1
                    n_features = conn_tensor.shape[0]
                    
                    # Get all valid (i,j) pairs
                    i_indices, j_indices = valid_connections.nonzero(as_tuple=True)
                    targets = conn_tensor[i_indices, j_indices]
                    
                    # Create mask using vectorized scatter
                    connection_mask = t.zeros(n_features, n_features, 
                                            device=self.device, dtype=t.bool)
                    connection_mask[i_indices, targets] = True
                    
                    self.connection_masks[down_name][up_name] = connection_mask
            self.connections = processed_connections
        elif connections is not None and connections != "all":
            raise ValueError("connections must be either None, 'all', or a dictionary of connection tensors")

        # Precompute W_OV matrices if model is frozen (3E)
        self.W_OVs = None
        if not any(p.requires_grad for p in model.parameters()):
            self.W_OVs = []
            for layer in range(model.cfg.n_layers):
                W_O = model.W_O[layer]
                W_V = model.W_V[layer]
                W_OV = einsum(W_O, W_V,
                    "n_heads d_head d_out, n_heads d_model d_head -> n_heads d_model d_out",
                ).to(device=self.device, dtype=self.dtype)
                self.W_OVs.append(W_OV)

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
            return contributions  # Added return statement here!
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
            
            # Replace inplace scatter_add_ with functional scatter_add
            contributions = t.zeros(
                up_facts.shape[0], 
                up_facts.shape[1], 
                down_encoder.shape[0], 
                device=up_facts.device, 
                dtype=up_facts.dtype
            )
            contributions = t.scatter_add(
                input=contributions,
                dim=-1,
                index=i_indices[None, None, :].expand_as(scaled),
                src=scaled
            )
            
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
        """Compute pruned contributions for attention autoencoder."""
        from einops import einsum

        layer = int(down_name.split('_')[1])
        
        # Add condition for when connections is None
        if self.connections is None:
            # Get precomputed W_OV if available (3E)
            if self.W_OVs is not None:
                W_OV = self.W_OVs[layer]
            else:
                W_O = self.model.W_O[layer]
                W_V = self.model.W_V[layer]
                W_OV = einsum(W_O, W_V, "n_heads d_head d_out, n_heads d_in d_head -> n_heads d_in d_out")
            
            up_decoder = self.aes[up_name].decoder.weight
            down_encoder = self.aes[down_name].encoder.weight
            
            # Compute virtual weights and contributions
            temp = einsum(down_encoder, W_OV, "f_down d_out, n_heads d_in d_out -> n_heads f_down d_in")
            values = einsum(temp, up_decoder, "n_heads f_down d_in, d_in f_up -> n_heads f_down f_up")
            contributions = einsum(up_facts, values, "b s f_up, n_heads f_down f_up -> b s n_heads f_down")
            
            # Apply attention pattern mixing
            probs = cache[f'blocks.{layer}.attn.hook_pattern']
            final_contribs = einsum(probs, contributions, "b h q k, b k h f -> b q f")
            return final_contribs
        
        else:
            # Rest of the code for when connections is not None
            # Get precomputed W_OV if available (3E)
            if self.W_OVs is not None:
                W_OV = self.W_OVs[layer]
            else:
                W_O = self.model.W_O[layer]
                W_V = self.model.W_V[layer]
                W_OV = einsum(W_O, W_V, "n_heads d_head d_out, n_heads d_in d_head -> n_heads d_in d_out")
            
            # Compute temp tensor
            temp = einsum(
                self.aes[down_name].encoder.weight, W_OV,
                "f_down d_out, n_heads d_in d_out -> n_heads f_down d_in"
            )
            
            # Get connection mask and indices
            connection_mask = self.connection_masks[down_name][up_name]
            indices = connection_mask.nonzero(as_tuple=False)
            i_indices, j_indices = indices[:, 0], indices[:, 1]
            
            # Vectorized computation (3A)
            selected_temp = temp[:, i_indices, :]  # [n_heads, M, d_in]
            selected_up = self.aes[up_name].decoder.weight[:, j_indices]  # [d_in, M]
            values = einsum(selected_temp, selected_up, "n_heads m d_in, d_in m -> n_heads m")
            
            # Gather relevant up_facts
            up_facts_selected = up_facts[:, :, j_indices]  # [B, S, M]
            
            # Expand dimensions for vectorized scatter
            scaled = up_facts_selected[:, :, None, :] * values[None, None, :, :]  # [B, S, n_heads, M]
            
            # Initialize contributions tensor
            contributions = t.zeros(
                up_facts.shape[0], up_facts.shape[1], temp.shape[0], temp.shape[1],
                device=up_facts.device, dtype=up_facts.dtype
            )
            
            # Vectorized scatter add
            contributions.scatter_add_(
                dim=-1,
                index=i_indices[None, None, None, :].expand_as(scaled),
                src=scaled
            )
            
            # Apply attention pattern mixing
            probs = cache[f'blocks.{layer}.attn.hook_pattern']
            final_contribs = einsum(probs, contributions, "b h q k, b k h f -> b q f")
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
        
        # Pre-allocate tensors without inplace reuse
        first_cache_tensor = cache['blocks.0.hook_resid_pre']
        batch_size, seq_len = first_cache_tensor.shape[:2]
        device, dtype = first_cache_tensor.device, first_cache_tensor.dtype
        
        # Initialize feat_buffer (no inplace reuse)
        feat_buffer = t.zeros(
            (batch_size, seq_len, self.n_features),
            device=device,
            dtype=dtype
        )
        
        for down_name in self.submodule_names:
            down_layer, down_type = parse_name(down_name)
            
            # Initialize approx_acts (out-of-place)
            if down_type == 'attn':
                approx_acts = self.get_initial_contribs_attn(cache, down_name)
            else:  # mlp
                approx_acts = self.get_initial_contribs_mlp(cache, down_name)
            
            # Initialize upstream_bias (out-of-place)
            upstream_bias = t.zeros(self.model.cfg.d_model, device=device, dtype=dtype)
            
            for up_name in self.submodule_names:
                up_layer, up_type = parse_name(up_name)
                
                if up_layer > down_layer or (up_layer == down_layer and (up_type == 'mlp' or up_name == down_name)):
                    continue
                    
                if self.connections is None or (isinstance(self.connections, dict) 
                    and down_name in self.connections 
                    and up_name in self.connections[down_name]):
                    
                    up_feats = pruned_features[up_name]
                    
                    # Avoid inplace: upstream_bias = upstream_bias + ...
                    upstream_bias = upstream_bias + self.aes[up_name].b_dec
                    
                    if down_type == 'attn':
                        contributions = self.get_pruned_contribs_attn(cache, up_name, down_name, up_feats)
                    else:
                        contributions = self.get_pruned_contribs_mlp(cache, up_name, down_name, up_feats)
                    
                    # Avoid inplace: approx_acts = approx_acts + contributions
                    approx_acts = approx_acts + contributions
                    del contributions
                    t.cuda.empty_cache()
            
            # Avoid inplace division: approx_acts = approx_acts / ln_scale
            ln_name = 'ln1' if down_type == 'attn' else 'ln2'
            ln_scale = cache[f'blocks.{down_layer}.{ln_name}.hook_scale']
            ln_scale = ln_scale.unsqueeze(-1) if ln_scale.dim() < approx_acts.dim() else ln_scale
            approx_acts = approx_acts / ln_scale
            
            # Avoid inplace addition for biases
            bias = self.aes[down_name].encoder.bias
            bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
            approx_acts = approx_acts + bias
            
            projected_bias = self.aes[down_name].encoder.weight @ upstream_bias
            projected_bias = projected_bias.unsqueeze(0) if projected_bias.dim() < approx_acts.dim() else projected_bias
            approx_acts = approx_acts + projected_bias
            
            # Get top k features (no inplace scatter)
            top_vals, top_idx = approx_acts.topk(self.k, dim=-1)
            
            # Avoid inplace zeroing: create a new feat_buffer
            feat_buffer = t.zeros_like(feat_buffer)
            
            # Avoid inplace scatter: create a new tensor
            feat_buffer = feat_buffer.scatter(-1, top_idx, top_vals)
            
            # Store pruned features (clone to avoid accidental inplace later)
            pruned_features[down_name] = feat_buffer.clone()
            
            # Decode and store reconstruction
            reconstructions[down_name] = self.aes[down_name].decode(feat_buffer)
            
            del top_vals, top_idx
            t.cuda.empty_cache()
        
        return reconstructions

    def get_ce_loss(
            self,
            reconstructions: Dict[str, t.Tensor],
            tokens: t.Tensor,  # shape: [batch, seq]
        ) -> t.Tensor:
        """Compute cross entropy loss from reconstructions.
        
        Args:
            reconstructions: Dictionary mapping layer names to reconstruction tensors
            tokens: Integer tensor of target tokens
            
        Returns:
            Cross entropy loss between predicted logits and target tokens
        """
        resid_final = sum(reconstructions.values())
        logits = self.model.unembed(self.model.ln_final(resid_final))  # [batch, seq, n_vocab]
        
        # Shift sequences by 1
        logits = logits[:, :-1, :]  # Remove last position
        tokens = tokens[:, 1:]  # Remove first position
        
        # Flatten batch and sequence dimensions
        logits = logits.reshape(-1, logits.size(-1))  # [batch*seq, n_vocab]
        tokens = tokens.reshape(-1)  # [batch*seq]
        
        loss = nn.functional.cross_entropy(
            logits,
            tokens,
            reduction='mean'
        )
        return loss
            

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