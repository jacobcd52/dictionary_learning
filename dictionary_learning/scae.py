from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Literal, Tuple, Optional, Union
import json
import einops
import torch as t
import torch.nn as nn
from transformer_lens import ActivationCache, HookedTransformer
import tempfile
import os


from .top_k_sae import AutoEncoderTopK, CrosscoderTopK
from .mask import SimpleBinaryMask

from utils import set_seed

set_seed(42)


Connections = Dict[str, Dict[str, t.Tensor]]


class SubmoduleName(NamedTuple):
    layer: int
    submodule_type: Literal["attn", "cc"]

    @property
    def name(self):
        return f"{self.submodule_type}_{self.layer}"

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return (
            self.layer == other.layer
            and self.submodule_type == other.submodule_type
        )

    @classmethod
    def from_str(cls, name_str: str) -> "SubmoduleName":
        parts = name_str.split("_")
        submodule_type = parts[0]
        if submodule_type not in ("attn", "cc"):
            raise ValueError(f"Invalid submodule type in name string: {submodule_type}")
        layer = int(parts[1])
        return cls(layer=layer, submodule_type=submodule_type)


class SCAEModule(nn.Module, ABC):
    def __init__(
        self,
        model: HookedTransformer,
        ae: Union[AutoEncoderTopK, CrosscoderTopK],
        upstream_aes: Dict[str, Union[AutoEncoderTopK, CrosscoderTopK]],
        connection_masks: nn.ModuleDict,
        name: SubmoduleName,
    ):
        super().__init__()

        self.model = model
        self.ae = ae
        self.upstream_aes = upstream_aes
        self.connection_masks = connection_masks
        self.name = name

    def forward(
        self,
        cache: ActivationCache,
        pruned_features: Dict[str, t.Tensor],
        feature_buffer: t.Tensor,
        temperature: float,
        runtime_use_sparse_connections_override: Optional[bool] = None,
    ) -> t.Tensor:
        actual_run_mode_is_sparse: bool
        if runtime_use_sparse_connections_override is not None:
            actual_run_mode_is_sparse = runtime_use_sparse_connections_override
        else:
            print("Warning: runtime_use_sparse_connections_override is None in SCAEModule.forward. Defaulting to False.")
            actual_run_mode_is_sparse = False

        approx_acts = self.get_initial_contribs(cache, actual_run_mode_is_sparse)

        upstream_bias_sum_for_current_module = None

        if actual_run_mode_is_sparse:
            for up_name, up_ae_instance in self.upstream_aes.items():
                current_b_dec_contrib = None
                if isinstance(up_ae_instance, AutoEncoderTopK):
                    current_b_dec_contrib = up_ae_instance.b_dec
                elif isinstance(up_ae_instance, CrosscoderTopK):
                    up_sm_name = SubmoduleName.from_str(up_name)
                    
                    target_mlp_layers_for_up_ae = list(range(up_sm_name.layer, up_sm_name.layer + up_ae_instance.n_outputs))
                    
                    relevant_indices = [
                        idx for idx, target_layer in enumerate(target_mlp_layers_for_up_ae) 
                        if target_layer < self.name.layer 
                    ]
                    
                    if relevant_indices:
                        current_b_dec_contrib = up_ae_instance.b_dec[relevant_indices, :].sum(dim=0)
                
                if current_b_dec_contrib is not None:
                    if upstream_bias_sum_for_current_module is None:
                        upstream_bias_sum_for_current_module = current_b_dec_contrib.clone()
                    else:
                        upstream_bias_sum_for_current_module = upstream_bias_sum_for_current_module + current_b_dec_contrib

                if self.connection_masks is not None and up_name in self.connection_masks:
                    connection_mask = self.connection_masks[up_name](temperature)
                else:
                    connection_mask = None

                up_pruned_features = pruned_features[up_name]
                pruned_contribs = self.get_pruned_contribs(
                    cache, up_name, up_ae_instance, connection_mask, up_pruned_features
                )
                approx_acts = approx_acts + pruned_contribs

        approx_acts = self.compute_bias(approx_acts, upstream_bias_sum_for_current_module, cache)

        # # Plot histogram of first feature's preactivations
        # import matplotlib.pyplot as plt
        # import os

        # plt.figure(figsize=(10, 6))
        # plt.hist(approx_acts[:, :, 0].flatten().detach().float().cpu().numpy(), bins=50, alpha=0.7)
        # plt.title('Histogram of First Feature Preactivations')
        # plt.xlabel('Preactivation Value')
        # plt.ylabel('Count')

        # # Create directory for plots if it doesn't exist
        # os.makedirs('plots', exist_ok=True)

        # plt.savefig(f'plots/{self.name.name}_preact_hist_feature_0.png')
        # plt.close()
        
        # approx_acts = t.relu(approx_acts)

        k_to_use = self.ae.k if isinstance(self.ae, AutoEncoderTopK) else self.ae.k
        top_vals, top_idx = approx_acts.topk(k_to_use, dim=-1)

        current_ae_dict_size = self.ae.dict_size
        scatter_buffer = t.zeros(
            (*top_idx.shape[:-1], current_ae_dict_size), 
            device=approx_acts.device, 
            dtype=approx_acts.dtype
        )
        scatter_buffer = scatter_buffer.scatter_(-1, top_idx, top_vals)

        reconstructions = self.ae.decode(scatter_buffer)

        return scatter_buffer, reconstructions

    def get_mask_loss(self, temperature: float):
        mask_loss = 0.
        if self.connection_masks:
            for mask in self.connection_masks.values():
                mask_loss += mask.mask_loss(temperature)
        return mask_loss

    @abstractmethod
    def get_initial_contribs(self, cache: ActivationCache, actual_run_mode_is_sparse: bool) -> t.Tensor:
        pass

    @abstractmethod
    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up_name: str,
        up_ae: Union[AutoEncoderTopK, CrosscoderTopK],
        connection_mask: t.Tensor,
        up_pruned_features: t.Tensor,
    ) -> t.Tensor:
        pass

    @abstractmethod
    def compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        pass


class SCAEAttention(SCAEModule):
    def __init__(
        self,
        model: HookedTransformer,
        ae: AutoEncoderTopK,
        upstream_aes: Dict[str, Union[AutoEncoderTopK, CrosscoderTopK]],
        connection_masks: nn.ModuleDict,
        name: SubmoduleName,
    ):
        super().__init__(model, ae, upstream_aes, connection_masks, name)

        W_O = model.W_O[self.name.layer]
        W_V = model.W_V[self.name.layer]

        self.W_OV = einops.einsum(
            W_O,
            W_V,
            "n_heads d_head d_out, n_heads d_model d_head -> n_heads d_model d_out",
        )

    def get_initial_contribs(self, cache: ActivationCache, actual_run_mode_is_sparse: bool) -> t.Tensor:
        if actual_run_mode_is_sparse:
            initial_act_hook_name = "blocks.0.hook_resid_pre"
        else:
            initial_act_hook_name = f"blocks.{self.name.layer}.hook_resid_pre"
        
        initial_act_post_ln = (
            cache[initial_act_hook_name]
            / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
        )

        down_enc = self.ae.encoder.weight

        initial_contrib_pre_moving = t.einsum(
            "b s i, h i o, d o -> b s h d",
            initial_act_post_ln,
            self.W_OV,
            down_enc,
        )

        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]

        initial_contrib = t.einsum(
            "b h q k, b k h d -> b q d", probs, initial_contrib_pre_moving
        )

        return initial_contrib

    def get_virtual_weights(
        self,
        up_name: str,
        up_ae: Union[AutoEncoderTopK, CrosscoderTopK],
        down_enc: t.Tensor,
        connection_mask: Optional[t.Tensor],
    ) -> t.Tensor:
        effective_up_dec_matrix = None

        if isinstance(up_ae, AutoEncoderTopK):
            effective_up_dec_matrix = up_ae.decoder.weight
        elif isinstance(up_ae, CrosscoderTopK):
            up_sm_name = SubmoduleName.from_str(up_name)
            target_mlp_layers_for_up_ae = list(range(up_sm_name.layer, up_sm_name.layer + up_ae.n_outputs))
            relevant_indices = [
                idx for idx, target_layer in enumerate(target_mlp_layers_for_up_ae)
                if target_layer < self.name.layer
            ]

            if relevant_indices:
                summed_dec_transposed = up_ae.decoder_weight[:, relevant_indices, :].sum(dim=1)
                effective_up_dec_matrix = summed_dec_transposed.transpose(-1,-2)
            else:
                effective_up_dec_matrix = t.zeros(
                    (self.model.cfg.d_model, up_ae.dict_size),
                    device=up_ae.decoder_weight.device,
                    dtype=up_ae.decoder_weight.dtype
                )

        virtual_weights = t.einsum(
            "d o, h i o, i u -> h d u",
            down_enc,
            self.W_OV,
            effective_up_dec_matrix,
        )
        if connection_mask is not None:
            virtual_weights = virtual_weights * connection_mask
        return virtual_weights

    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up_name: str,
        up_ae: Union[AutoEncoderTopK, CrosscoderTopK],
        connection_mask: t.Tensor,
        up_pruned_features: t.Tensor,
    ) -> t.Tensor:
        down_enc = self.ae.encoder.weight
        
        virtual_weights = self.get_virtual_weights(up_name, up_ae, down_enc, connection_mask)
 
        up_facts_post_ln = (
            up_pruned_features
            / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
        )
        contributions_post_ov = t.einsum(
            "b q u, h d u -> b h q d",
            up_facts_post_ln,
            virtual_weights,
        )

        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]

        contributions = t.einsum(
            "b h q k, b h k d -> b q d", probs, contributions_post_ov
        )

        return contributions

    def compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        down_enc = self.ae.encoder.weight
        b_O = self.model.b_O[self.name.layer].squeeze()
        b_O_contribution = b_O @ down_enc.T
        approx_acts = approx_acts + b_O_contribution

        down_enc_bias = self.ae.encoder.bias
        approx_acts = approx_acts + down_enc_bias

        if upstream_bias is not None:
            upstream_bias_post_ln = (
                upstream_bias.unsqueeze(0).unsqueeze(0)
                / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
            )

            projected_through_ov_bias = t.einsum(
                "h i o, b s i -> b s h o",
                self.W_OV,
                upstream_bias_post_ln,
            )
            
            bias_contrib_pre_moving = t.einsum(
                "b s i, h i o, d o -> b s h d",
                upstream_bias_post_ln,
                self.W_OV,
                down_enc,
            )
            probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]
            projected_bias = t.einsum(
                "b h q k, b k h d -> b q d", probs, bias_contrib_pre_moving
            )
            return approx_acts + projected_bias

        return approx_acts


class SCAECrossCoder(SCAEModule):
    def __init__(
        self,
        model: HookedTransformer,
        ae: CrosscoderTopK,
        upstream_aes: Dict[str, Union[AutoEncoderTopK, CrosscoderTopK]],
        connection_masks: nn.ModuleDict,
        name: SubmoduleName,
    ):
        super().__init__(model, ae, upstream_aes, connection_masks, name)

    def get_initial_contribs(self, cache: ActivationCache, actual_run_mode_is_sparse: bool) -> t.Tensor:
        down_enc = self.ae.encoder.weight

        if actual_run_mode_is_sparse:
            initial_act_hook_name = "blocks.0.hook_resid_pre"
        else:
            initial_act_hook_name = f"blocks.{self.name.layer}.hook_resid_pre"

        initial_act_input = (
            cache[initial_act_hook_name]
        )

        return initial_act_input @ down_enc.T

    def get_virtual_weights(
        self,
        up_name: str,
        up_ae: Union[AutoEncoderTopK, CrosscoderTopK],
        down_enc: t.Tensor,
        connection_mask: Optional[t.Tensor],
    ) -> t.Tensor:
        effective_up_dec_matrix = None

        if isinstance(up_ae, AutoEncoderTopK):
            effective_up_dec_matrix = up_ae.decoder.weight
        elif isinstance(up_ae, CrosscoderTopK):
            up_sm_name = SubmoduleName.from_str(up_name)
            target_mlp_layers_for_up_ae = list(range(up_sm_name.layer, up_sm_name.layer + up_ae.n_outputs))
            relevant_indices = [
                idx for idx, target_layer in enumerate(target_mlp_layers_for_up_ae)
                if target_layer < self.name.layer
            ]

            if relevant_indices:
                summed_dec_transposed = up_ae.decoder_weight[:, relevant_indices, :].sum(dim=1)
                effective_up_dec_matrix = summed_dec_transposed.transpose(-1,-2)
            else:
                effective_up_dec_matrix = t.zeros(
                    (self.model.cfg.d_model, up_ae.dict_size),
                    device=up_ae.decoder_weight.device,
                    dtype=up_ae.decoder_weight.dtype
                )
        
        virtual_weights = down_enc @ effective_up_dec_matrix
        if connection_mask is not None:
            virtual_weights = virtual_weights * connection_mask
        return virtual_weights

    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up_name: str,
        up_ae: Union[AutoEncoderTopK, CrosscoderTopK],
        connection_mask: t.Tensor,
        up_pruned_features: t.Tensor,
    ) -> t.Tensor:
        down_enc = self.ae.encoder.weight

        virtual_weights = self.get_virtual_weights(up_name, up_ae, down_enc, connection_mask)
        
        up_facts_post_ln = (
            up_pruned_features
        )

        contributions = up_facts_post_ln @ virtual_weights.T

        return contributions

    def compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        if self.ae.encoder.bias is not None:
            approx_acts = approx_acts + self.ae.encoder.bias

        if upstream_bias is not None:
            upstream_bias_post_ln = (
                upstream_bias.unsqueeze(0).unsqueeze(0)
            )
            
            projected_bias = upstream_bias_post_ln @ self.ae.encoder.weight.T
            approx_acts = approx_acts + projected_bias
            
        return approx_acts


class SCAESuite(nn.Module):
    """A suite of Sparsely-Connected TopK Autoencoders"""

    def __init__(
        self,
        model,
        k: int,
        target_C: int,
        n_features: int,
        mask_type: str,
        device: str,
        dtype: t.dtype,
    ):
        """
        Args:
            model: TransformerLens model
            k: Number of features to select for each autoencoder
            n_features: Dictionary size for each autoencoder
            dtype: Data type for the autoencoders
            device: Device to place the autoencoders on
        """
        super().__init__()

        self.model = model
        self.dtype = dtype

        self.k = k
        self.n_features = n_features
        self.target_C = target_C
        self.mask_type = mask_type
        self.device = device

        submodule_names = [
            SubmoduleName(layer=i, submodule_type=submodule_type)
            for i in range(model.cfg.n_layers)
            for submodule_type in ["attn", "cc"]
        ]

        aes = {}
        for sm in submodule_names:
            if sm.submodule_type == "attn":
                aes[sm.name] = AutoEncoderTopK(model.cfg.d_model, n_features, k).to(device).to(dtype)
            elif sm.submodule_type == "cc":
                n_outputs = model.cfg.n_layers - sm.layer
                if n_outputs <=0:
                    n_outputs = 1
                aes[sm.name] = CrosscoderTopK(
                    activation_dim=model.cfg.d_model, 
                    dict_size=n_features, 
                    k=k, 
                    n_outputs=n_outputs
                ).to(device).to(dtype)

        self.module_dict = self._make_module_dict(submodule_names, aes)

    def _make_module_dict(
        self, submodule_names: List[SubmoduleName], aes: Dict[str, Union[AutoEncoderTopK, CrosscoderTopK]]
    ) -> nn.ModuleDict:
        def _make_module(submodule_type: Literal["attn", "cc"], *args):
            if submodule_type == "attn":
                return SCAEAttention(*args)
            elif submodule_type == "cc":
                return SCAECrossCoder(*args)
            else:
                raise ValueError(f"Unknown submodule type: {submodule_type}")

        module_dict = {}
        for down in submodule_names:
            upstream_aes_for_module = {}
            for up in submodule_names:
                if not self.does_precede(up, down):
                    continue
                    
                if up.name in aes:
                    upstream_aes_for_module[up.name] = aes[up.name]
                else:
                    print(f"Warning: Upstream AE {up.name} not found in aes dictionary.")
                    continue
            
            connection_masks_for_module = None
            if self.target_C != -1 and upstream_aes_for_module:
                mask_components = {}
                for up_key_loop_var in upstream_aes_for_module.keys():
                    down_ae_instance = aes[down.name]
                    up_ae_instance = upstream_aes_for_module[up_key_loop_var]

                    n_down_features = down_ae_instance.dict_size
                    n_up_features = up_ae_instance.dict_size

                    if self.mask_type == "simple":
                        mask_instance = SimpleBinaryMask(
                            n_features_down=n_down_features,
                            n_features_up=n_up_features,
                            target_C=self.target_C,
                        )
                    else:
                        raise ValueError(f"Unsupported mask_type: {self.mask_type}. Choose 'simple'.")
                    
                    mask_components[up_key_loop_var] = mask_instance.to(self.device).to(self.dtype)
                connection_masks_for_module = nn.ModuleDict(mask_components)

            module_dict[down.name] = _make_module(
                down.submodule_type,
                self.model,
                aes[down.name],
                upstream_aes_for_module,
                connection_masks_for_module,
                down,
            )

        return nn.ModuleDict(module_dict)

    def does_precede(self, up_name: SubmoduleName, down_name: SubmoduleName):
        return up_name.layer < down_name.layer # note difference to old, non-crosscoder suite
    

    def forward(
        self, cache: ActivationCache, temperature: float,
        runtime_use_sparse_connections_override: Optional[bool] = None,
    ) -> t.Tensor:
        reconstructions = {}
        pruned_features = {}

        cache_tensor = cache["blocks.0.hook_resid_pre"]
        batch_size, seq_len = cache_tensor.shape[:2]
        device, dtype = cache_tensor.device, cache_tensor.dtype

        feat_buffer = t.zeros(
            (batch_size, seq_len, self.n_features),
            device=device,
            dtype=dtype,
        )

        for layer in range(self.model.cfg.n_layers):
            for module_type in ["attn", "cc"]:
                module_name = f"{module_type}_{layer}"
                
                if module_name not in self.module_dict:
                    continue
                module = self.module_dict[module_name]

                current_module_ae_dict_size = module.ae.dict_size
                
                current_feat_buffer = t.zeros(
                    (*feat_buffer.shape[:2], current_module_ae_dict_size),
                    device=device,
                    dtype=dtype,
                )

                module_output_features, reconstruction = module(
                    cache, 
                    pruned_features, 
                    current_feat_buffer, 
                    temperature,
                    runtime_use_sparse_connections_override,
                )
                
                pruned_features[module_name] = module_output_features
                reconstructions[module_name] = reconstruction

        return reconstructions, pruned_features

    def get_ce_loss(
        self,
        cache,
        reconstructions: Dict[str, t.Tensor],
        tokens: t.Tensor,
    ) -> t.Tensor:
        resid_final = sum(reconstructions.values())
        resid_final = resid_final + cache["blocks.0.hook_resid_pre"]

        logits = self.model.unembed(self.model.ln_final(resid_final))

        logits = logits[:, :-1, :]
        tokens = tokens[:, 1:]

        logits = logits.reshape(-1, logits.size(-1))
        tokens = tokens.reshape(-1)

        loss = nn.functional.cross_entropy(logits, tokens, reduction="mean")
        return loss

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        model,
        device: Optional[str] = None,
        dtype: t.dtype = t.float32,
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

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        suite = cls(
            model=model,
            k=config["k"],
            target_C=config.get("target_C", -1),
            n_features=config["n_features"],
            mask_type=config.get("mask_type", "simple"),
            device=device,
            dtype=dtype,
        )

        checkpoint_path = hf_hub_download(
            repo_id=repo_id, filename="checkpoint.pt"
        )
        state_dict = t.load(checkpoint_path, map_location="cpu")

        missing_keys, unexpected_keys = suite.load_state_dict(
            state_dict, strict=False
        )

        if len(missing_keys) > 0:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")

        suite.is_pretrained = True

        return suite

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

        config = {
            "k": self.k,
            "n_features": self.n_features,
            "mask_type": self.mask_type,
            "target_C": self.target_C,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")
            state_dict = self.state_dict()
            t.save(state_dict, checkpoint_path)

            api = HfApi()

            try:
                api.repo_info(repo_id=repo_id, repo_type="model")
            except Exception:
                api.create_repo(
                    repo_id=repo_id, repo_type="model", private=private
                )

            api.upload_folder(
                folder_path=tmp_dir, repo_id=repo_id, repo_type="model"
            )


class MergedSCAESuite(nn.Module):
    def __init__(self, transformer: HookedTransformer, scae_suite: SCAESuite):
        super().__init__()

        self.transformer = transformer
        self.scae_suite = scae_suite

        self.hook_list = []
        for layer in range(self.transformer.cfg.n_layers):
            self.hook_list += [
                f"blocks.{layer}.ln1.hook_scale",
                f"blocks.{layer}.hook_attn_out",
                f"blocks.{layer}.hook_mlp_out",
                f"blocks.{layer}.attn.hook_pattern",
                f"blocks.{layer}.hook_resid_pre",
            ]

    def get_trainable_params(self):
        params = []
        for module in self.scae_suite.module_dict.values():
            for submodule in module.modules():
                if isinstance(submodule, (AutoEncoderTopK, CrosscoderTopK)):
                    params.extend(submodule.parameters())
                elif isinstance(submodule, SimpleBinaryMask):
                    params.extend(submodule.parameters())

        return params

    def clip_grad_norm(self, max_norm: float = 1.0):
        for module in self.scae_suite.module_dict.values():
            for submodule in module.modules():
                is_ae = isinstance(submodule, (AutoEncoderTopK, CrosscoderTopK))
                if is_ae and hasattr(submodule, 'decoder') and submodule.decoder.weight.grad is not None:
                    t.nn.utils.clip_grad_norm_(submodule.parameters(), max_norm)
                elif is_ae and hasattr(submodule, 'decoder_weight') and submodule.decoder_weight.grad is not None:
                    t.nn.utils.clip_grad_norm_(submodule.parameters(), max_norm)

    @t.no_grad()
    def _get_cache(self, input_ids: t.Tensor) -> ActivationCache:
        base_loss, cache = self.transformer.run_with_cache(
            input_ids, return_type="loss", names_filter=self.hook_list
        )

        cache = cache.cache_dict
        cache["loss"] = base_loss

        for hook_name in self.hook_list:
            if (".ln" in hook_name) or (".hook_pattern" in hook_name):
                cache[hook_name] = cache[hook_name].to(self.scae_suite.dtype)

        return cache

    def forward(self, input_ids: t.Tensor, temperature: float,
                runtime_use_sparse_connections_override: Optional[bool] = None,
                ):
        cache = self._get_cache(input_ids)
        reconstructions, pruned_features = self.scae_suite(
            cache, temperature,
            runtime_use_sparse_connections_override=runtime_use_sparse_connections_override,
        )

        return reconstructions, pruned_features, cache
