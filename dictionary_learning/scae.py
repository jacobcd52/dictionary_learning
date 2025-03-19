from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Literal

import einops
from cut_cross_entropy import linear_cross_entropy
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
        upstream_bias = t.zeros(
            self.model.cfg.d_model, device="cuda", dtype=t.bfloat16
        )

        for up_name, ae in self.upstream_aes.items():
            upstream_bias = upstream_bias + ae.b_dec

            connection_mask = self.connection_masks[up_name]
            up_pruned_features = pruned_features[up_name]
            pruned_contribs = self.get_pruned_contribs(
                cache, ae, connection_mask, up_pruned_features
            )
            approx_acts = approx_acts + pruned_contribs

        approx_acts = self.compute_bias(approx_acts, upstream_bias, cache)

        # top_vals, top_idx = approx_acts.topk(self.k, dim=-1)
        top_vals, top_idx = approx_acts.topk(64, dim=-1)
        top_vals = t.relu(top_vals)

        feat_buffer = feature_buffer.scatter(-1, top_idx, top_vals)

        reconstructions = self.ae.decode(feat_buffer)

        return feature_buffer, reconstructions

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

        # batch seq d_model, n_heads d_model_in d_model_out, n_features d_model_out
        # -> batch seq n_heads n_features
        down_enc = self.ae.encoder.weight
        initial_contrib_pre_moving = t.einsum(
            "b s m, h i o, f o -> b s h f",
            initial_act_post_ln,
            self.W_OV,
            down_enc,
        )

        # Mix between positions using attention pattern
        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]

        # batch n_heads qseq kseq, batch kseq n_heads n_features
        # -> batch qseq n_features
        initial_contrib = t.einsum(
            "b h q k, b k h f -> b q f", probs, initial_contrib_pre_moving
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
            up_pruned_features / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
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

        down_enc_bias = self.ae.encoder.bias
        # bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
        approx_acts = approx_acts + down_enc_bias
        approx_acts = approx_acts - down_enc @ self.ae.b_dec

        # Add upstream b_dec contributions
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
        up_decoder = up_ae.decoder.weight
        down_encoder = self.ae.encoder.weight

        virtual_weights = down_encoder @ up_decoder
        if connection_mask is not None:
            virtual_weights = virtual_weights * connection_mask

        up_facts_post_ln = (
            up_pruned_features / cache[f"blocks.{self.name.layer}.ln2.hook_scale"]
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


class SCAESuite(nn.Module):
    """A suite of Sparsely-Connected TopK Autoencoders"""

    def __init__(
        self,
        model,
        k: int,
        n_features: int,
        device: str = "cuda",
        dtype: t.dtype = t.bfloat16,
        connections: Connections | str = None,
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
        self.device = device
        self.dtype = dtype

        self.k = k
        self.n_features = n_features

        n_layers = model.cfg.n_layers

        self.connections = connections
        self.connection_masks = self._process_connections(connections)

        submodule_names = [
            SubmoduleName(layer=i, submodule_type=submodule_type)
            for i in range(n_layers)
            for submodule_type in ["attn", "mlp"]
        ]

        aes = {
            name: AutoEncoderTopK(model.cfg.d_model, n_features, k)
            .to(self.device)
            .to(self.dtype)
            for name in submodule_names
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
                    upstream_aes[up.name] = aes[up]

            module_dict[down.name] = _make_module(
                down.submodule_type,
                self.model,
                aes[down],
                upstream_aes,
                self.connection_masks[down.name],
                down,
            )

        return nn.ModuleDict(module_dict)

    def _process_connections(self, connections: Connections):
        """Process connections to create connection masks and validate input."""

        connection_masks = {}
        for down_name, up_dict in connections.items():
            connection_masks[down_name] = {}
            for up_name, conn_tensor in up_dict.items():
                # Verify indices are valid

                # valid_indices = (
                #     conn_tensor >= -1 and conn_tensor < self.n_features
                # )

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
                    device=self.device,
                    dtype=t.bool,
                )
                connection_mask[i_indices, targets] = True

                connection_masks[down_name][up_name] = connection_mask

        return connection_masks

    def forward(self, cache: ActivationCache) -> t.Tensor:
        reconstructions = {}
        pruned_features = {}

        # Pre-allocate tensors without inplace reuse
        first_cache_tensor = cache["blocks.0.hook_resid_pre"]
        batch_size, seq_len = first_cache_tensor.shape[:2]
        device, dtype = first_cache_tensor.device, first_cache_tensor.dtype

        # Initialize feat_buffer (no inplace reuse)
        feat_buffer = t.zeros(
            (batch_size, seq_len, self.n_features), device=device, dtype=dtype
        )

        for module_name, module in self.module_dict.items():
            feature_buffer, reconstruction = module(
                cache, pruned_features, feat_buffer
            )
            pruned_features[module_name] = feature_buffer
            reconstructions[module_name] = reconstruction

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
        resid_final += cache["blocks.0.hook_resid_pre"]

        logits = self.model.unembed(self.model.ln_final(resid_final))

        # https://github.com/apple/ml-cross-entropy
        # GPTNeo TinyStories has no unembed bias.
        loss = linear_cross_entropy(
            logits, self.model.unembed.weight, tokens, shift=1
        )

        return loss
