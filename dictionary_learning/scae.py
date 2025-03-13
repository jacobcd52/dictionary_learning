import torch as t
import torch.nn as nn
import einops
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass
import numpy as np
import json
from einops import einsum

from .top_k import AutoEncoderTopK
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
        
        for p in model.parameters():
            p.requires_grad = False
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
        self._process_connections()
        
        # Precompute W_OV matrices
        self.W_OVs = []
        for layer in range(model.cfg.n_layers):
            W_O = model.W_O[layer]
            W_V = model.W_V[layer]
            W_OV = einsum(W_O, W_V,
                "n_heads d_head d_out, n_heads d_model d_head -> n_heads d_model d_out",
            ).to(device=self.device, dtype=self.dtype)
            self.W_OVs.append(W_OV)

    def _process_connections(self):
        """Process connections to create connection masks and validate input."""
        if isinstance(self.connections, dict):
            processed_connections = {}
            self.connection_masks = {}
            for down_name, up_dict in self.connections.items():
                processed_connections[down_name] = {}
                self.connection_masks[down_name] = {}
                for up_name, conn_tensor in up_dict.items():
                    # Verify indices are valid
                    valid_indices = (conn_tensor >= -1) & (conn_tensor < self.n_features)
                    assert valid_indices.all(), \
                        f"Invalid indices in connection tensor for {down_name}->{up_name}. All values must be -1 or valid feature indices."
                    
                    processed_connections[down_name][up_name] = conn_tensor.to(device=self.device, dtype=t.long)
                    
                    # Vectorized connection mask creation
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
        elif self.connections is not None and self.connections != "all":
            raise ValueError("connections must be either None, 'all', or a dictionary of connection tensors")

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
        assert 'mlp' in down_name
        layer = int(down_name.split('_')[1])
        initial_act = cache['blocks.0.hook_resid_pre']  # [batch, seq, d_model]
        W_enc = self.aes[down_name].encoder.weight  # [n_features, d_model]
        
        initial_act_post_ln = initial_act / cache[f'blocks.{layer}.ln2.hook_scale']
        return initial_act_post_ln @ W_enc.T  # [batch, seq, n_features]

    def get_pruned_contribs_mlp(
        self,
        cache: ActivationCache,
        up_name: str,
        down_name: str,
        up_facts: t.Tensor,
    ) -> t.Tensor:
        assert 'mlp' in down_name
        
        layer = int(down_name.split('_')[1])
        up_decoder = self.aes[up_name].decoder.weight
        down_encoder = self.aes[down_name].encoder.weight
        virtual_weights = down_encoder @ up_decoder
        if self.connections is not None:
            virtual_weights = virtual_weights * self.connection_masks[down_name][up_name]

        up_facts_post_ln = up_facts / cache[f'blocks.{layer}.ln2.hook_scale']
        contributions = up_facts_post_ln @ virtual_weights.T # TODO: check transpose
        return contributions  # Added return statement here!
            
    def get_initial_contribs_attn(
            self,
            cache: ActivationCache,
            down_name: str,  # e.g. 'attn_1'
        ) -> t.Tensor:  # [batch, qpos, n_down_features]
            """Compute initial contributions for attention autoencoder from residual stream."""
            assert 'attn' in down_name
            
            layer = int(down_name.split('_')[1])
            initial_act = cache['blocks.0.hook_resid_pre'] # [batch, seq, d_model]
            initial_act_post_ln = initial_act / cache[f'blocks.{layer}.ln1.hook_scale']
            
            # Break down the einsum operations and clean up intermediates
            W_OV = self.W_OVs[layer]  # [n_heads, d_model, d_out]
            t.cuda.empty_cache()
            
            down_encoder = self.aes[down_name].encoder.weight  # [f_down, d_out]
            initial_contrib_pre_moving = einsum(initial_act_post_ln, W_OV, down_encoder,
                                            "batch pos d_in, n_heads d_in d_out, f_down d_out -> batch pos n_heads f_down")
            
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
        assert 'attn' in down_name

        layer = int(down_name.split('_')[1])
        W_OV = self.W_OVs[layer]   
        down_encoder = self.aes[down_name].encoder.weight
        up_decoder  = self.aes[up_name].decoder.weight
        virtual_weights = einsum(down_encoder, W_OV, up_decoder,
            "f_down d_out, n_heads d_in d_out, d_in f_up -> n_heads f_down f_up")
        if self.connections is not None:
            virtual_weights = virtual_weights * self.connection_masks[down_name][up_name].unsqueeze(0)
        up_facts_post_ln = up_facts / cache[f'blocks.{layer}.ln1.hook_scale']
        contributions_post_ov = einsum(up_facts_post_ln, virtual_weights,
            "batch qpos f_up, n_heads f_down f_up -> batch n_heads qpos f_down")
        
        # Mix between positions using attention pattern
        probs = cache[f'blocks.{layer}.attn.hook_pattern']
        contributions = einsum(probs, contributions_post_ov,
            "batch n_heads qpos kpos, batch n_heads kpos f_down -> batch qpos f_down")
        return contributions
    
    def forward_pruned(
        self,
        cache: ActivationCache,
        return_features = False, 
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

            # Messy bias stuff. Beware bugs.
            if down_type == 'attn':
                approx_acts = approx_acts + self.model.b_O[down_layer].squeeze().to(self.dtype) @ self.aes[down_name].encoder.weight.T
            
            # Add downstream b_enc 
            bias = self.aes[down_name].encoder.bias
            bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
            approx_acts = approx_acts + bias

            # Subtract downstream b_dec contribution
            approx_acts = approx_acts - self.aes[down_name].encoder.weight @ self.aes[down_name].b_dec
            
            # Add upstream b_dec contributions
            ln_name = 'ln1' if down_type == 'attn' else 'ln2'
            upstream_bias_post_ln = upstream_bias.unsqueeze(0).unsqueeze(0) / cache[f'blocks.{down_layer}.{ln_name}.hook_scale']
            if down_type == 'attn':
                upstream_bias_post_ln = einsum(self.W_OVs[down_layer], upstream_bias_post_ln, 
                                               "n_heads d_in d_out, b s d_in -> b s d_out")
            projected_bias = einsum(self.aes[down_name].encoder.weight,
                                     upstream_bias_post_ln,
                                     "f d, b s d -> b s f")
            approx_acts = approx_acts + projected_bias
            
            # Get top k features (no inplace scatter)
            top_vals, top_idx = approx_acts.topk(self.k, dim=-1)
            top_vals = t.relu(top_vals)

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
        
        if(return_features):
            return reconstructions, pruned_features
        return reconstructions

    def get_ce_loss(
            self,
            cache,
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
        resid_final += cache['blocks.0.hook_resid_pre'] # don't forget initial contribution! thanks logan
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
            

    def upload_to_hf(self, repo_id: str, private: bool = False):
        """
        Upload the model to HuggingFace Hub. Creates the repository if it doesn't exist.
        
        Args:
            repo_id: HuggingFace repository ID to upload to
            private: Whether the repository should be private if created (default: False)
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
        }
        
        # Save config and state dict with connections
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")
            state_dict = self.state_dict()
            state_dict['connections'] = self.connections
            t.save(state_dict, checkpoint_path)
            
            # Upload files
            api = HfApi()
            
            # Check if repo exists and create it if it doesn't
            try:
                api.repo_info(repo_id=repo_id, repo_type="model")
            except Exception:  # Repository doesn't exist
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=private
                )
                
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
        connections: Optional[Union[Dict[str, Dict[str, t.Tensor]], str]] = None,
        device: Optional[str] = None,
        dtype: t.dtype = t.float32
    ) -> "SCAESuite":
        """
        Load a pretrained SCAESuite from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID containing the saved model
            model: TransformerLens model
            connections: Optional connections to override loaded ones
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
        
        # Initialize suite with possible user-provided connections
        suite = cls(
            model=model,
            k=config["k"],
            n_features=config["n_features"],
            connections=connections,
            dtype=dtype,
            device=device
        )
        
        # Download and load state dict
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pt")
        state_dict = t.load(checkpoint_path, map_location='cpu')
        
        # Extract connections from state_dict if present
        loaded_connections = state_dict.pop('connections', None)
        
        # Handle connections override
        if loaded_connections is not None:
            if connections is not None:
                print("Warning: Provided connections argument overrides the loaded connections from HuggingFace.")
            else:
                suite.connections = loaded_connections
                suite._process_connections()
        
        # Load the state_dict into the suite
        missing_keys, unexpected_keys = suite.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        suite.is_pretrained = True
        return suite