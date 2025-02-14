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
        cache,
        up_name: str,
        down_name: str,
        up_facts: t.Tensor,  # [batch, seq, n_up_features]
    ) -> t.Tensor:  # [batch, seq, n_down_features]
        """Compute contributions from upstream to downstream MLP autoencoder.
        
        Args:
            cache: TransformerLens activation cache
            up_name: Name of upstream autoencoder (e.g. 'mlp_0', 'attn_0')
            down_name: Name of downstream autoencoder (must be MLP type)
            up_facts: Upstream features/activations
            
        Returns:
            Tensor of contributions from upstream to downstream features
        """
        if self.connections is None:
            up_decoder = self.aes[up_name].decoder.weight
            down_encoder = self.aes[down_name].encoder.weight
            virtual_weights = down_encoder @ up_decoder
            contributions = up_facts @ virtual_weights.T
            
        else:
            connections = self.connections[down_name][up_name]
            up_decoder = self.aes[up_name].decoder.weight
            down_encoder = self.aes[down_name].encoder.weight
            virtual_weights = down_encoder @ up_decoder
            connection_mask = self.connection_masks[down_name][up_name]
            masked_weights = virtual_weights * connection_mask
            contributions = up_facts @ masked_weights.T
        
        b_dec_contrib = down_encoder @ self.aes[up_name].b_dec
        contributions = contributions + b_dec_contrib.unsqueeze(0).unsqueeze(0)
        
        return contributions
    
    def forward_pruned(
        self,
        cache: ActivationCache,
    ) -> Dict[str, t.Tensor]:
        """Forward pass computing sparse reconstructions.
        
        Args:
            cache: TransformerLens activation cache
            
        Returns:
            Dictionary mapping module names to their reconstructions
        """
        reconstructions = {}
        
        # Helper to extract layer number and type from module name
        def parse_name(name: str) -> Tuple[int, str]:
            type_, layer = name.split('_')
            return int(layer), type_
        
        # For each downstream module
        for down_name in self.submodule_names:
            down_layer, down_type = parse_name(down_name)
            
            # For attention modules, return empty tensor for now
            if down_type == 'attn':
                reconstructions[down_name] = t.tensor([])
                continue
                
            # For MLP modules, compute all contributions
            approx_acts = self.get_initial_contribs_mlp(cache, down_name)
            
            # Add contributions from all upstream modules
            for up_name in self.submodule_names:
                up_layer, up_type = parse_name(up_name)
                
                # Skip if not upstream
                if up_layer > down_layer or (up_layer == down_layer and up_type == 'mlp'):
                    continue
                    
                if self.connections is None or (isinstance(self.connections, dict) and down_name in self.connections and up_name in self.connections[down_name]):
                    # Get features from upstream module
                    up_feats = self.aes[up_name].encode(cache[f'blocks.{up_layer}.ln1.hook_normalized' if up_type == 'attn' else f'blocks.{up_layer}.ln2.hook_normalized'])
                    
                    # Get contributions
                    contributions = self.get_pruned_contribs_mlp(cache, up_name, down_name, up_feats)
                    approx_acts = approx_acts + contributions
            
            # Apply layernorm scale
            approx_acts = approx_acts / cache[f'blocks.{down_layer}.ln2.hook_scale']
            
            # Add encoder bias
            b_enc = self.aes[down_name].encoder.bias
            approx_acts = approx_acts + b_enc
            
            # Subtract decoder bias contribution through encoder
            W_enc = self.aes[down_name].encoder.weight
            approx_acts = approx_acts - W_enc @ self.aes[down_name].b_dec
            
            # Get top k features
            top_vals, top_idx = approx_acts.topk(self.k, dim=-1)
            
            # Create feature tensor
            batch_size, seq_len = approx_acts.shape[:2]
            feat = t.zeros(
                (batch_size, seq_len, self.n_features),
                device=approx_acts.device,
                dtype=self.dtype
            )
            
            # Scatter top k values
            feat.scatter_(-1, top_idx, top_vals)
            
            # Decode to get reconstruction
            reconstructions[down_name] = self.aes[down_name].decode(feat)
        
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