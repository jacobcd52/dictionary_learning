from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Literal

import einops
from cut_cross_entropy import linear_cross_entropy
import torch as t
import torch.nn as nn
from transformer_lens import ActivationCache

from .gpt_neox import GPTNeoXForCausalLM
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
        model: GPTNeoXForCausalLM,
        ae: AutoEncoderTopK,
        upstream_aes: List[AutoEncoderTopK],
        connection_mask: Connections,
        name: SubmoduleName,
    ):
        super().__init__()

        self.model = model
        self.ae = ae
        self.upstream_aes = upstream_aes
        self.connection_mask = connection_mask
        self.name = name

        self.pruned_features = None

    def forward(
        self, feature_buffer: t.Tensor, cache: ActivationCache
    ) -> t.Tensor:
        approx_acts = self.get_initial_contribs(cache)
        upstream_bias = t.zeros(
            self.model.cfg.d_model, device=self.device, dtype=self.dtype
        )

        for ae in self.upstream_aes:
            upstream_bias = upstream_bias + ae.b_dec

            pruned_contribs = self.get_pruned_contribs(cache, ae)
            approx_acts = approx_acts + pruned_contribs

        approx_acts = self._compute_bias(approx_acts, upstream_bias, cache)

        top_vals, top_idx = approx_acts.topk(self.k, dim=-1)
        top_vals = t.relu(top_vals)

        feat_buffer = feature_buffer.scatter(-1, top_idx, top_vals)
        self.pruned_features = feat_buffer.clone()

        reconstructions = self.ae.decode(feat_buffer)

        return reconstructions

    @abstractmethod
    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        pass

    @abstractmethod
    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up_ae: AutoEncoderTopK,
    ) -> t.Tensor:
        pass

    @abstractmethod
    def _compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        pass


class SCAEAttention(SCAEModule):
    def __init__(
        self,
        model: GPTNeoXForCausalLM,
        ae: AutoEncoderTopK,
        upstream_aes: List[AutoEncoderTopK],
        connection_mask: Connections,
        name: SubmoduleName,
    ):
        super().__init__(model, ae, upstream_aes, connection_mask, name)

        W_QKV = model.gpt_neox.layers[name.layer].attention.query_key_value
        _, _, W_V = W_QKV.chunk(3, dim=-1)

        W_O = model.gpt_neox.layers[name.layer].attention.dense.weight

        self.W_OV = einops.einsum(
            W_O,
            W_V,
            "(n_heads d_head) d_model_out, (n_heads d_head) d_model_in \
                -> n_heads d_model_in d_model_out",
            d_head=64,
            n_heads=8,
        )

    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        initial_act_post_ln = (
            cache["embed_out"] / cache[f"blocks.{self.name.layer}.ln1_scale"]
        )

        # batch seq d_model, n_heads d_model_in d_model_out, n_features d_model_out
        # -> batch seq n_heads n_features
        down_enc = self.ae.encoder.weight
        initial_contrib_pre_moving = t.einsum(
            initial_act_post_ln,
            self.W_OV,
            down_enc,
            "b s m, h i o, f o -> b s h f",
        )

        # Mix between positions using attention pattern
        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]

        # batch n_heads qseq kseq, batch kseq n_heads n_features
        # -> batch qseq n_features
        initial_contrib = t.einsum(
            probs, initial_contrib_pre_moving, "b h q k, b k h f -> b q f"
        )

        return initial_contrib

    def get_pruned_contribs(
        self, cache: ActivationCache, up_ae: SubmoduleName
    ) -> t.Tensor:
        """Compute pruned contributions for attention autoencoder."""

        down_enc = self.ae.encoder.weight
        up_dec = up_ae.decoder.weight

        # n_features_down d_model_out, n_heads d_model_in d_model_out, d_model_in n_features_up
        # -> n_heads n_features_down n_features_up
        virtual_weights = t.einsum(
            down_enc,
            self.W_OV,
            up_dec,
            "d o, h i o, i u -> h d u",
        )
        if self.connection_mask is not None:
            virtual_weights = virtual_weights * self.connection_mask.unsqueeze(
                0
            )

        up_facts_post_ln = (
            self.pruned_features / cache[f"blocks.{self.name.layer}.ln1_scale"]
        )
        contributions_post_ov = t.einsum(
            up_facts_post_ln,
            virtual_weights,
            "batch qseq n_features_up, n_heads n_features_down n_features_up -> batch n_heads qseq n_features_down",
        )

        # Mix between positions using attention pattern
        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]

        # batch n_heads qseq kseq, batch n_heads kseq n_features_down
        # -> batch qseq n_features_down
        contributions = t.einsum(
            probs, contributions_post_ov, "b h q k, b h k d -> b q d"
        )

        return contributions

    def compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        
        down_enc = self.ae.encoder.weight

        b_O = self.model.gpt_neox.layers[self.name.layer].attention.dense.bias
        b_O_contribution = b_O.squeeze() @ down_enc.T
        approx_acts = approx_acts + b_O_contribution

        down_enc_bias = self.ae.encoder.bias
        # bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
        approx_acts = approx_acts + down_enc_bias
        approx_acts = approx_acts - down_enc @ self.ae.b_dec

        # Add upstream b_dec contributions
        upstream_bias_post_ln = (
            upstream_bias.unsqueeze(0).unsqueeze(0)
            / cache[f"blocks.{self.name.layer}.ln1_scale"]
        )

        # n_heads d_model_in d_model_out, batch seq d_model_in 
        # -> batch seq d_out
        upstream_bias_post_ln = t.einsum(
            self.W_OV,
            upstream_bias_post_ln,
            "h i o, b s i -> b s o",
        )

        # n_features_down d_model_in, batch seq d_model_in 
        # -> batch seq n_features_down
        projected_bias = t.einsum(
            down_enc,
            upstream_bias_post_ln,
            "d i, b s i -> b s d"
        )

        return approx_acts + projected_bias


class SCAEMLP(SCAEModule):
    def __init__(
        self,
        model: GPTNeoXForCausalLM,
        ae: AutoEncoderTopK,
        name: SubmoduleName,
        upstream_aes: List[AutoEncoderTopK],
        connection_mask: Connections = None,
    ):
        super().__init__(model, ae, upstream_aes, connection_mask, name)

    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        down_enc = self.ae.encoder.weight

        initial_act_post_ln = (
            cache["embed_out"] / cache[f"blocks.{self.name.layer}.ln2_scale"]
        )

        # batch seq d_model, n_features d_model -> batch seq n_features
        return initial_act_post_ln @ down_enc.T

    def get_pruned_contribs(
        self, cache: ActivationCache, up_ae: AutoEncoderTopK
    ) -> t.Tensor:
        up_decoder = up_ae.decoder.weight
        down_encoder = self.ae.encoder.weight

        virtual_weights = down_encoder @ up_decoder
        if self.connection_mask is not None:
            virtual_weights = virtual_weights * self.connection_mask

        up_facts_post_ln = (
            self.pruned_features / cache[f"blocks.{self.name.layer}.ln2_scale"]
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
            / cache[f"blocks.{self.name.layer}.ln2_scale"]
        )

        # n_features_down d_model_in, batch seq d_model_in 
        # -> batch seq n_features_down
        projected_bias = t.einsum(
            down_enc,
            upstream_bias_post_ln,
            "d i, b s i -> b s d"
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

        submodule_names = [
            SubmoduleName(layer=i, submodule_type=submodule_type)
            for i in range(n_layers)
            for submodule_type in ["attn", "mlp"]
        ]

        aes = {
            name: AutoEncoderTopK(model.cfg.d_model, n_features, k).to(
                device=self.device, dtype=self.dtype
            )
            for name in submodule_names
        }

        self.module_dict = self._make_module_dict(submodule_names, aes)

        # Initialize and validate connections
        self.connection_masks = self._process_connections(connections)

    def _make_module_dict(
        self, submodule_names: List[SubmoduleName], aes: List[AutoEncoderTopK]
    ) -> nn.ModuleDict:
        def _make_module(submodule_type: Literal["attn", "mlp"], *args):
            submodule = SCAEAttention if submodule_type == "attn" else SCAEMLP
            return submodule(*args)

        module_dict = {}
        for down in submodule_names:
            upstream_aes = []

            for up in submodule_names:
                # Adjacent means an attention block feeding
                # into an MLP block on the same layer
                adjacent = (
                    up.layer == down.layer
                    and up.submodule_type == "attn"
                    and down.submodule_type == "mlp"
                )

                # Skip if up is on a later layer or not adjacent
                if up.layer > down.layer or not adjacent:
                    continue

                if self.connections is None or (
                    isinstance(self.connections, dict)
                    and down.name in self.connections
                    and up.name in self.connections[down.name]
                ):
                    upstream_aes.append(aes[up])

            module_dict[down.name] = _make_module(
                down.submodule_type,
                self.model,
                aes[down],
                upstream_aes,
                self.connections,
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
                valid_indices = (
                    conn_tensor >= -1 and conn_tensor < self.n_features
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

        # Pre-allocate tensors without inplace reuse
        first_cache_tensor = cache["embed_out"]
        batch_size, seq_len = first_cache_tensor.shape[:2]
        device, dtype = first_cache_tensor.device, first_cache_tensor.dtype

        # Initialize feat_buffer (no inplace reuse)
        feat_buffer = t.zeros(
            (batch_size, seq_len, self.n_features), device=device, dtype=dtype
        )

        for module_name, module in self.module_dict.items():
            reconstructions[module_name] = module(feat_buffer, cache)

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
        resid_final += cache["embed_out"]

        unembed = self.model.lm_head
        logits = unembed(
            self.model.ln_final(resid_final)
        )  # [batch, seq, n_vocab]

        # https://github.com/apple/ml-cross-entropy
        # GPTNeo TinyStories has no unembed bias.
        loss = linear_cross_entropy(logits, unembed.weight, tokens, shift=1)

        return loss
