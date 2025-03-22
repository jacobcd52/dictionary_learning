from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Literal, Tuple

import einops
import torch as t
import torch.nn as nn
from transformer_lens import ActivationCache, HookedTransformer

from .top_k import AutoEncoderTopK


Connections = Dict[str, Dict[str, t.Tensor]]


class SubmoduleName(NamedTuple):
    layer: int
    submodule_type: Literal["attn", "mlp"]

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


class SCAEModule(nn.Module, ABC):
    def __init__(
        self,
        model: HookedTransformer,
        ae: AutoEncoderTopK,
        upstream_aes: Dict[str, AutoEncoderTopK],
        connection_masks: Dict[str, t.Tensor],
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
    ) -> t.Tensor:
        approx_acts = self.get_initial_contribs(cache)

        upstream_bias = None

        for up_name, ae in self.upstream_aes.items():
            if upstream_bias is None:
                upstream_bias = ae.b_dec
            else:
                upstream_bias = upstream_bias + ae.b_dec

            if self.connection_masks is not None:
                connection_mask = self.connection_masks[up_name]
            else:
                connection_mask = None

            up_pruned_features = pruned_features[up_name]
            pruned_contribs = self.get_pruned_contribs(
                cache, ae, connection_mask, up_pruned_features
            )
            approx_acts = approx_acts + pruned_contribs

        approx_acts = self.compute_bias(approx_acts, upstream_bias, cache)

        # NOTE: Sorting is off here, check if that's okay.
        top_vals, top_idx = approx_acts.topk(self.ae.k, dim=-1, sorted=False)
        top_vals = t.relu(top_vals)

        feat_buffer = t.zeros_like(feature_buffer)
        feat_buffer = feat_buffer.scatter(-1, top_idx, top_vals)

        reconstructions = self.ae.decode(feat_buffer)

        return feat_buffer, reconstructions

    @abstractmethod
    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        pass

    @abstractmethod
    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up_ae: AutoEncoderTopK,
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
        upstream_aes: List[AutoEncoderTopK],
        connection_masks: Dict[str, t.Tensor],
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

    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        initial_act_post_ln = (
            cache["blocks.0.hook_resid_pre"]
            / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
        )

        # batch seq d_model_in, n_heads d_model_in d_model_out, n_features_down d_model_out
        # -> batch seq n_heads n_features_down
        down_enc = self.ae.encoder.weight
        initial_contrib_pre_moving = t.einsum(
            "b s i, h i o, d o -> b s h d",
            initial_act_post_ln,
            self.W_OV,
            down_enc,
        )

        # Mix between positions using attention pattern
        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]

        # batch n_heads qseq kseq, batch kseq n_heads n_features_down
        # -> batch qseq n_features_down
        initial_contrib = t.einsum(
            "b h q k, b k h d -> b q d", probs, initial_contrib_pre_moving
        )

        return initial_contrib

    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up_ae: AutoEncoderTopK,
        connection_mask: t.Tensor,
        up_pruned_features: t.Tensor,
    ) -> t.Tensor:
        """Compute pruned contributions for attention autoencoder."""

        down_enc = self.ae.encoder.weight
        up_dec = up_ae.decoder.weight

        # n_features_down d_model_out, n_heads d_model_in d_model_out, d_model_in n_features_up
        # -> n_heads n_features_down n_features_up
        virtual_weights = t.einsum(
            "d o, h i o, i u -> h d u",
            down_enc,
            self.W_OV,
            up_dec,
        )
        if connection_mask is not None:
            virtual_weights = virtual_weights * connection_mask.unsqueeze(0)

        up_facts_post_ln = (
            up_pruned_features
            / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
        )
        # batch qseq n_features_up, n_heads n_features_down n_features_up
        # -> batch n_heads qseq n_features_down
        contributions_post_ov = t.einsum(
            "b q u, h d u -> b h q d",
            up_facts_post_ln,
            virtual_weights,
        )

        # Mix between positions using attention pattern
        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]

        # batch n_heads qseq kseq, batch n_heads kseq n_features_down
        # -> batch qseq n_features_down
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

        # Add downstream b_enc
        down_enc_bias = self.ae.encoder.bias
        approx_acts = approx_acts + down_enc_bias

        # Subtract downstream b_dec contribution
        approx_acts = approx_acts - down_enc @ self.ae.b_dec

        # Add upstream b_dec contributions
        if upstream_bias is not None:
            upstream_bias_post_ln = (
                upstream_bias.unsqueeze(0).unsqueeze(0)
                / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
            )

            # n_heads d_model_in d_model_out, batch seq d_model_in
            # -> batch seq d_out
            upstream_bias_post_ln = t.einsum(
                "h i o, b s i -> b s o",
                self.W_OV,
                upstream_bias_post_ln,
            )

            # n_features_down d_model_in, batch seq d_model_in
            # -> batch seq n_features_down
            projected_bias = t.einsum(
                "d i, b s i -> b s d",
                down_enc,
                upstream_bias_post_ln,
            )

            return approx_acts + projected_bias

        return approx_acts


class SCAEMLP(SCAEModule):
    def __init__(
        self,
        model: HookedTransformer,
        ae: AutoEncoderTopK,
        upstream_aes: List[AutoEncoderTopK],
        connection_masks: Dict[str, t.Tensor],
        name: SubmoduleName,
    ):
        super().__init__(model, ae, upstream_aes, connection_masks, name)

    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        down_enc = self.ae.encoder.weight

        initial_act_post_ln = (
            cache["blocks.0.hook_resid_pre"]
            / cache[f"blocks.{self.name.layer}.ln2.hook_scale"]
        )

        # batch seq d_model, n_features d_model -> batch seq n_features
        return initial_act_post_ln @ down_enc.T

    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up_ae: AutoEncoderTopK,
        connection_mask: t.Tensor,
        up_pruned_features: t.Tensor,
    ) -> t.Tensor:
        up_dec = up_ae.decoder.weight
        down_enc = self.ae.encoder.weight

        virtual_weights = down_enc @ up_dec
        if connection_mask is not None:
            virtual_weights = virtual_weights * connection_mask

        up_facts_post_ln = (
            up_pruned_features
            / cache[f"blocks.{self.name.layer}.ln2.hook_scale"]
        )

        contributions = up_facts_post_ln @ virtual_weights.T

        return contributions

    def compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        down_enc = self.ae.encoder.weight
        down_enc_bias = self.ae.encoder.bias

        # bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
        approx_acts = approx_acts + down_enc_bias
        approx_acts = approx_acts - down_enc @ self.ae.b_dec

        if upstream_bias is not None:
            # Add upstream b_dec contributions
            upstream_bias_post_ln = (
                upstream_bias.unsqueeze(0).unsqueeze(0)
                / cache[f"blocks.{self.name.layer}.ln2.hook_scale"]
            )

            # n_features_down d_model_in, batch seq d_model_in
            # -> batch seq n_features_down
            projected_bias = t.einsum(
                "d i, b s i -> b s d",
                down_enc,
                upstream_bias_post_ln,
            )

            return approx_acts + projected_bias

        return approx_acts


class SCAESuite(nn.Module):
    """A suite of Sparsely-Connected TopK Autoencoders"""

    def __init__(
        self,
        model,
        k: int,
        n_features: int,
        device: str,
        dtype: t.dtype,
        connections: Connections = None,
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
        self.dtype = dtype

        self.k = k
        self.n_features = n_features

        self.connections = connections
        if connections is not None:
            self.connection_masks = self._process_connections(connections, device)
        else:
            self.connection_masks = None

        submodule_names = [
            SubmoduleName(layer=i, submodule_type=submodule_type)
            for i in range(model.cfg.n_layers)
            for submodule_type in ["attn", "mlp"]
        ]

        aes = {
            sm.name: AutoEncoderTopK(model.cfg.d_model, n_features, k)
            .to(device)
            .to(dtype)
            for sm in submodule_names
        }

        self.module_dict = self._make_module_dict(submodule_names, aes)

    def _make_module_dict(
        self, submodule_names: List[SubmoduleName], aes: List[AutoEncoderTopK]
    ) -> nn.ModuleDict:
        def _make_module(submodule_type: Literal["attn", "mlp"], *args):
            submodule = SCAEAttention if submodule_type == "attn" else SCAEMLP
            return submodule(*args)

        module_dict = {}
        for down in submodule_names:
            upstream_aes = {}

            for up in submodule_names:
                if up.layer >= down.layer:
                    continue

                if self.connections is None or (
                    down.name in self.connections
                    and up.name in self.connections[down.name]
                ):
                    upstream_aes[up.name] = aes[up.name]

            connection_masks = (
                self.connection_masks[down.name]
                if self.connections is not None
                else None
            )
            module_dict[down.name] = _make_module(
                down.submodule_type,
                self.model,
                aes[down.name],
                upstream_aes,
                connection_masks,
                down,
            )

        return nn.ModuleDict(module_dict)

    def _process_connections(self, connections: Connections, device: t.device):
        """Process connections to create connection masks and validate input."""

        connection_masks = {}
        for down_name, up_dict in connections.items():
            connection_masks[down_name] = {}
            for up_name, conn_tensor in up_dict.items():
                valid_indices = (conn_tensor >= -1) & (
                    conn_tensor < self.n_features
                )

                if not valid_indices.all():
                    raise ValueError(
                        "Invalid indices in connection tensor for",
                        f"{down_name}->{up_name}. All values must be",
                        "-1 or valid feature indices.",
                    )

                # Vectorized connection mask creation
                valid_connections = conn_tensor != -1
                n_features = conn_tensor.shape[0]

                # Get all valid (i,j) pairs
                i_indices, j_indices = valid_connections.nonzero(as_tuple=True)
                targets = conn_tensor[i_indices, j_indices]

                # Create mask using vectorized scatter
                connection_mask = t.zeros(
                    n_features,
                    n_features,
                    device=device,
                    dtype=t.bool,
                )
                connection_mask[i_indices, targets] = True

                connection_masks[down_name][up_name] = connection_mask

        return connection_masks

    def forward(self, cache: ActivationCache) -> t.Tensor:
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

        for module_name, module in self.module_dict.items():
            feature_buffer, reconstruction = module(
                cache, pruned_features, feat_buffer
            )
            pruned_features[module_name] = feature_buffer
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

        # Shift sequences by 1
        logits = logits[:, :-1, :]
        tokens = tokens[:, 1:]

        logits = logits.reshape(-1, logits.size(-1))
        tokens = tokens.reshape(-1)

        loss = nn.functional.cross_entropy(logits, tokens, reduction="mean")
        return loss


class MergedSCAESuite(nn.Module):
    def __init__(self, transformer: HookedTransformer, scae_suite: SCAESuite):
        super().__init__()

        self.transformer = transformer
        self.scae_suite = scae_suite

        self.hook_list = ["blocks.0.hook_resid_pre"]
        for layer in range(self.transformer.cfg.n_layers):
            self.hook_list += [
                f"blocks.{layer}.ln1.hook_scale",
                f"blocks.{layer}.ln2.hook_scale",
                f"blocks.{layer}.hook_attn_out",
                f"blocks.{layer}.hook_mlp_out",
                f"blocks.{layer}.attn.hook_pattern",
            ]

    def get_trainable_params(self):
        params = []
        for module in self.scae_suite.module_dict.values():
            for submodule in module.modules():
                if isinstance(submodule, AutoEncoderTopK):
                    params.extend(submodule.parameters())

        return params

    def clip_grad_norm(self, max_norm: float = 1.0):
        for module in self.scae_suite.module_dict.values():
            for submodule in module.modules():
                is_ae = isinstance(submodule, AutoEncoderTopK)
                if is_ae and submodule.decoder.weight.grad is not None:
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

    def forward(self, input_ids: t.Tensor):
        cache = self._get_cache(input_ids)
        reconstructions, pruned_features = self.scae_suite(cache)

        return reconstructions, pruned_features, cache
