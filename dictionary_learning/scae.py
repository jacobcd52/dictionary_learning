from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Literal

# from einops import einsum
from cut_cross_entropy import linear_cross_entropy
import torch as t
import torch.nn as nn
from transformer_lens import ActivationCache

from .gpt_neo import GPTNeoForCausalLM
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
        model: GPTNeoForCausalLM,
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
        model: GPTNeoForCausalLM,
        ae: AutoEncoderTopK,
        upstream_aes: List[AutoEncoderTopK],
        connection_mask: Connections,
        name: SubmoduleName,
    ):
        super().__init__(model, ae, upstream_aes, connection_mask, name)

        W_O = model.transformer.h[name.layer].attn.attention.o_proj.weight
        W_V = model.transformer.h[name.layer].attn.attention.v_proj.weight

        self.W_OV = einsum(
            W_O,
            W_V,
            "n_heads d_head d_out, n_heads d_model d_head -> n_heads d_model d_out",
        )

    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        """Compute initial contributions for attention autoencoder from residual stream."""

        initial_act = cache["blocks.0.hook_resid_pre"]  # [batch, seq, d_model]
        initial_act_post_ln = (
            initial_act / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
        )

        down_encoder = self.ae.encoder.weight  # [f_down, d_out]
        initial_contrib_pre_moving = einsum(
            initial_act_post_ln,
            self.W_OV,
            down_encoder,
            "batch pos d_in, n_heads d_in d_out, f_down d_out -> batch pos n_heads f_down",
        )

        # Mix between positions using attention pattern
        probs = cache[
            f"blocks.{self.name.layer}.attn.hook_pattern"
        ]  # [batch, n_heads, qpos, kpos]
        initial_contrib = einsum(
            probs,
            initial_contrib_pre_moving,
            "batch n_heads qpos kpos, batch kpos n_heads f_down -> batch qpos f_down",
        )

        return initial_contrib

    def get_pruned_contribs(
        self, cache: ActivationCache, up_ae: SubmoduleName
    ) -> t.Tensor:
        """Compute pruned contributions for attention autoencoder."""

        down_encoder = self.ae.encoder.weight
        up_decoder = up_ae.decoder.weight
        virtual_weights = einsum(
            down_encoder,
            self.W_OV,
            up_decoder,
            "f_down d_out, n_heads d_in d_out, d_in f_up -> n_heads f_down f_up",
        )

        virtual_weights = virtual_weights * self.connection_mask.unsqueeze(0)
        up_facts_post_ln = (
            self.pruned_features
            / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
        )
        contributions_post_ov = einsum(
            up_facts_post_ln,
            virtual_weights,
            "batch qpos f_up, n_heads f_down f_up -> batch n_heads qpos f_down",
        )

        # Mix between positions using attention pattern
        probs = cache[f"blocks.{self.name.layer}.attn.hook_pattern"]
        contributions = einsum(
            probs,
            contributions_post_ov,
            "batch n_heads qpos kpos, batch n_heads kpos f_down -> batch qpos f_down",
        )
        return contributions

    def compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        approx_acts = (
            approx_acts
            + self.model.b_O[self.name.layer].squeeze().to(self.dtype)
            @ self.ae.encoder.weight.T
        )

        # Add downstream b_enc
        bias = self.ae.encoder.bias
        bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
        approx_acts = approx_acts + bias

        # Subtract downstream b_dec contribution
        approx_acts = approx_acts - self.ae.encoder.weight @ self.ae.b_dec

        # Add upstream b_dec contributions
        upstream_bias_post_ln = (
            upstream_bias.unsqueeze(0).unsqueeze(0)
            / cache[f"blocks.{self.name.layer}.ln1.hook_scale"]
        )

        upstream_bias_post_ln = einsum(
            self.W_OV,
            upstream_bias_post_ln,
            "n_heads d_in d_out, b s d_in -> b s d_out",
        )

        projected_bias = einsum(
            self.ae.encoder.weight,
            upstream_bias_post_ln,
            "f d, b s d -> b s f",
        )

        return approx_acts + projected_bias


class SCAEMLP(SCAEModule):
    def __init__(
        self,
        model: GPTNeoForCausalLM,
        ae: AutoEncoderTopK,
        name: SubmoduleName,
        upstream_aes: List[AutoEncoderTopK],
        connection_mask: Connections,
    ):
        super().__init__(model, ae, upstream_aes, connection_mask, name)

    def get_initial_contribs(self, cache: ActivationCache) -> t.Tensor:
        initial_act = cache["blocks.0.hook_resid_pre"]  # [batch, seq, d_model]
        W_enc = self.ae.encoder.weight  # [n_features, d_model]

        initial_act_post_ln = (
            initial_act / cache[f"blocks.{self.name.layer}.ln2.hook_scale"]
        )
        return initial_act_post_ln @ W_enc.T  # [batch, seq, n_features]

    def get_pruned_contribs(
        self, cache: ActivationCache, up_ae: AutoEncoderTopK
    ) -> t.Tensor:
        up_decoder = up_ae.decoder.weight
        down_encoder = self.ae.encoder.weight

        virtual_weights = (down_encoder @ up_decoder) * self.connection_mask

        up_facts_post_ln = (
            self.pruned_features
            / cache[f"blocks.{self.name.layer}.ln2.hook_scale"]
        )

        # TODO: check transpose
        contributions = up_facts_post_ln @ virtual_weights.T

        return contributions

    def compute_bias(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        # Add downstream b_enc
        bias = self.ae.encoder.bias
        bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
        approx_acts = approx_acts + bias

        # Subtract downstream b_dec contribution
        approx_acts = approx_acts - self.ae.encoder.weight @ self.ae.b_dec

        # Add upstream b_dec contributions
        upstream_bias_post_ln = (
            upstream_bias.unsqueeze(0).unsqueeze(0)
            / cache[f"blocks.{self.name.layer}.ln2.hook_scale"]
        )

        projected_bias = einsum(
            self.ae.encoder.weight,
            upstream_bias_post_ln,
            "f d, b s d -> b s f",
        )

        return projected_bias


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
                    conn_tensor >= -1
                    and conn_tensor < self.n_features
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
        first_cache_tensor = cache["blocks.0.hook_resid_pre"]
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
        resid_final += cache["blocks.0.hook_resid_pre"]

        unembed = self.model.lm_head
        logits = unembed(
            self.model.ln_final(resid_final)
        )  # [batch, seq, n_vocab]

        # https://github.com/apple/ml-cross-entropy
        # GPTNeo TinyStories has no unembed bias.
        loss = linear_cross_entropy(logits, unembed.weight, tokens, shift=1)

        return loss
