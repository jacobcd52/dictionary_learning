import torch as t
import torch.nn as nn

from typing import Dict, List
from torchtyping import TensorType
from einops import einsum

from .gpt_neo import GPTNeoForCausalLM
from transformer_lens import ActivationCache

from .top_k import AutoEncoderTopK

from cut_cross_entropy import linear_cross_entropy

from typing import NamedTuple, Literal
from abc import ABC, abstractmethod

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

    def forward(self, cache: ActivationCache, down: SubmoduleName) -> t.Tensor:
        initial_contribs = self.get_initial_contribs(cache, down)

    @abstractmethod
    def get_initial_contribs(
        self, cache: ActivationCache, down: SubmoduleName
    ) -> t.Tensor:
        pass

    @abstractmethod
    def get_pruned_contribs(
        self,
        cache: ActivationCache,
        up: SubmoduleName,
        down: SubmoduleName,
        up_facts: t.Tensor,
    ) -> t.Tensor:
        pass

    def _compute_bias(
        self,
        down: SubmoduleName,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        # Add downstream b_enc
        bias = self.aes[down.name].encoder.bias
        bias = bias.unsqueeze(0) if bias.dim() < approx_acts.dim() else bias
        approx_acts = approx_acts + bias

        # Subtract downstream b_dec contribution
        approx_acts = (
            approx_acts
            - self.aes[down.name].encoder.weight @ self.aes[down.name].b_dec
        )

        # Add upstream b_dec contributions
        ln_name = "ln1" if down.submodule_type == "attn" else "ln2"
        upstream_bias_post_ln = (
            upstream_bias.unsqueeze(0).unsqueeze(0)
            / cache[f"blocks.{down.layer}.{ln_name}.hook_scale"]
        )
        if down.submodule_type == "attn":
            # n_heads d_in d_out, b s d_in -> b s d_out
            upstream_bias_post_ln = t.einsum(
                "hmd,bsm->bsd",
                self.W_OVs[down.layer],  # [n_heads, d_in, d_out]
                upstream_bias_post_ln,  # [batch, seq, d_in]
            )

        # f d, b s d -> b s f
        projected_bias = t.einsum(
            "fd,bsd->bsf",
            self.aes[down.name].encoder.weight,  # [f, d]
            upstream_bias_post_ln,  # [batch, seq, d]
        )

        approx_acts = approx_acts + projected_bias
        return approx_acts

    def compute_upstream(
        self,
        approx_acts: t.Tensor,
        upstream_bias: t.Tensor,
        cache: ActivationCache,
    ):
        for ae in self.upstream_aes:
            pass

        up_feats = pruned_features[up.name]

        # Avoid inplace: upstream_bias = upstream_bias + ...
        upstream_bias = upstream_bias + self.aes[up.name].b_dec

        if down.submodule_type == "attn":
            contributions = self.get_pruned_contribs_attn(
                cache, up.name, down.name, up_feats
            )
        else:
            contributions = self.get_pruned_contribs_mlp(
                cache, up.name, down.name, up_feats
            )

        # Avoid inplace: approx_acts = approx_acts + contributions
        approx_acts = approx_acts + contributions


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

        self.ae = ae

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

        self.modules = self._make_modules(submodule_names, aes)

        # Initialize and validate connections
        self.connection_masks = {}
        self.connections = self._process_connections(connections)

        # Precompute W_OV matrices
        self.W_OVs = []
        for layer in range(model.cfg.n_layers):
            W_O = model.W_O[layer]
            W_V = model.W_V[layer]
            W_OV = einsum(
                W_O,
                W_V,
                "n_heads d_head d_out, n_heads d_model d_head -> n_heads d_model d_out",
            ).to(device=self.device, dtype=self.dtype)
            self.W_OVs.append(W_OV)

    def _make_modules(
        self, submodule_names: List[SubmoduleName], aes: List[AutoEncoderTopK]
    ) -> Dict[SubmoduleName, List[SubmoduleName]]:
        modules = []
        for down in submodule_names:
            upstream_modules = []

            for up in submodule_names:
                if up.layer > down.layer or (
                    up.layer == down.layer
                    and (up.submodule_type == "mlp" or up == down)
                ):
                    continue

                if self.connections is None or (
                    isinstance(self.connections, dict)
                    and down.name in self.connections
                    and up.name in self.connections[down.name]
                ):
                    upstream_modules.append(up)

            if down.submodule_type == "attn":
                modules.append(
                    SCAEAttention(self.model, aes[down], down, self.connections)
                )
            elif down.submodule_type == "mlp":
                modules.append(
                    SCAEMLP(self.model, aes[down], down, self.connections)
                )

        return modules

    def _process_connections(self, connections: Connections):
        """Process connections to create connection masks and validate input."""
        if isinstance(connections, dict):
            processed_connections = {}
            self.connection_masks = {}
            for down_name, up_dict in connections.items():
                processed_connections[down_name] = {}
                self.connection_masks[down_name] = {}
                for up_name, conn_tensor in up_dict.items():
                    # Verify indices are valid
                    valid_indices = (conn_tensor >= -1) & (
                        conn_tensor < self.n_features
                    )
                    assert valid_indices.all(), (
                        f"Invalid indices in connection tensor for {down_name}->{up_name}. All values must be -1 or valid feature indices."
                    )

                    processed_connections[down_name][up_name] = conn_tensor.to(
                        device=self.device, dtype=t.long
                    )

                    # Vectorized connection mask creation
                    valid_connections = conn_tensor != -1
                    n_features = conn_tensor.shape[0]

                    # Get all valid (i,j) pairs
                    i_indices, j_indices = valid_connections.nonzero(
                        as_tuple=True
                    )
                    targets = conn_tensor[i_indices, j_indices]

                    # Create mask using vectorized scatter
                    connection_mask = t.zeros(
                        n_features,
                        n_features,
                        device=self.device,
                        dtype=t.bool,
                    )
                    connection_mask[i_indices, targets] = True

                    self.connection_masks[down_name][up_name] = connection_mask
            connections = processed_connections
        elif connections is not None and connections != "all":
            raise ValueError(
                "connections must be either None, 'all', or a dictionary of connection tensors"
            )

        return connections

    def build_modules(self):
        pass

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
        layer = int(down_name.split("_")[1])
        initial_act = cache["blocks.0.hook_resid_pre"]  # [batch, seq, d_model]
        W_enc = self.aes[down_name].encoder.weight  # [n_features, d_model]

        initial_act_post_ln = (
            initial_act / cache[f"blocks.{layer}.ln2.hook_scale"]
        )
        return initial_act_post_ln @ W_enc.T  # [batch, seq, n_features]

    def get_pruned_contribs_mlp(
        self,
        cache: ActivationCache,
        up_name: str,
        down_name: str,
        up_facts: t.Tensor,
    ) -> t.Tensor:
        assert "mlp" in down_name

        layer = int(down_name.split("_")[1])
        up_decoder = self.aes[up_name].decoder.weight
        down_encoder = self.aes[down_name].encoder.weight
        virtual_weights = down_encoder @ up_decoder
        if self.connections is not None:
            virtual_weights = (
                virtual_weights * self.connection_masks[down_name][up_name]
            )

        up_facts_post_ln = up_facts / cache[f"blocks.{layer}.ln2.hook_scale"]
        contributions = (
            up_facts_post_ln @ virtual_weights.T
        )  # TODO: check transpose
        return contributions  # Added return statement here!

    def get_initial_contribs_attn(
        self, cache: ActivationCache, down: SubmoduleName
    ) -> t.Tensor:  # [batch, qpos, n_down_features]
        """Compute initial contributions for attention autoencoder from residual stream."""

        initial_act = cache["blocks.0.hook_resid_pre"]  # [batch, seq, d_model]
        initial_act_post_ln = (
            initial_act / cache[f"blocks.{down.layer}.ln1.hook_scale"]
        )

        # Break down the einsum operations and clean up intermediates
        W_OV = self.W_OVs[down.layer]  # [n_heads, d_model, d_out]
        t.cuda.empty_cache()

        down_encoder = self.aes[down].encoder.weight  # [f_down, d_out]

        # batch pos d_in, n_heads d_in d_out, f_down d_out -> batch pos n_heads f_down
        initial_contrib_pre_moving = t.einsum(
            "bpd,hdq,fq->bphf",
            initial_act_post_ln,  # [batch, pos, d_in]
            W_OV,  # [n_heads, d_in, d_out]
            down_encoder,  # [f_down, d_out]
        )

        # Mix between positions using attention pattern
        probs = cache[
            f"blocks.{down.layer}.attn.hook_pattern"
        ]  # [batch, n_heads, qpos, kpos]

        # batch n_heads qpos kpos, batch kpos n_heads f_down -> batch qpos f_down
        initial_contrib = t.einsum(
            "bhqk,bkhf->bqf",
            probs,  # [batch, n_heads, qpos, kpos]
            initial_contrib_pre_moving,  # [batch, kpos, n_heads, f_down]
        )
        # del initial_contrib_pre_moving
        # t.cuda.empty_cache()

        return initial_contrib

    def get_pruned_contribs_attn(
        self,
        cache: ActivationCache,
        up_name: str,
        down_name: str,
        up_facts: t.Tensor,
    ) -> t.Tensor:
        """Compute pruned contributions for attention autoencoder."""
        assert "attn" in down_name

        layer = int(down_name.split("_")[1])
        W_OV = self.W_OVs[layer]
        down_encoder = self.aes[down_name].encoder.weight
        up_decoder = self.aes[up_name].decoder.weight

        # f_down d_out, n_heads d_in d_out, d_in f_up -> n_heads f_down f_up
        virtual_weights = t.einsum(
            "fd,hmd,mf->hdf",
            down_encoder,  # [f_down, d_out]
            W_OV,  # [n_heads, d_in, d_out]
            up_decoder,  # [d_in, f_up]
        )

        if self.connections is not None:
            virtual_weights = virtual_weights * self.connection_masks[
                down_name
            ][up_name].unsqueeze(0)
        up_facts_post_ln = up_facts / cache[f"blocks.{layer}.ln1.hook_scale"]

        # batch qpos f_up, n_heads f_down f_up -> batch n_heads qpos f_down
        contributions_post_ov = t.einsum(
            "bqf,hdf->bhqd",
            up_facts_post_ln,  # [batch, qpos, f_up]
            virtual_weights,  # [n_heads, f_down, f_up]
        )

        # batch n_heads qpos kpos, batch n_heads kpos f_down -> batch qpos f_down
        probs = cache[f"blocks.{layer}.attn.hook_pattern"]
        contributions = t.einsum(
            "bhqk,bhkd->bqd",
            probs,  # [batch, n_heads, qpos, kpos]
            contributions_post_ov,  # [batch, n_heads, kpos, f_down]
        )
        return contributions

    def forward_pruned(
        self,
        cache: ActivationCache,
        return_features=False,
    ) -> Dict[str, t.Tensor]:
        """Forward pass computing sparse reconstructions."""
        reconstructions = {}
        pruned_features = {}  # Store pruned features for each module

        # Pre-allocate tensors without inplace reuse
        first_cache_tensor = cache["blocks.0.hook_resid_pre"]
        batch_size, seq_len = first_cache_tensor.shape[:2]
        device, dtype = first_cache_tensor.device, first_cache_tensor.dtype

        # Initialize feat_buffer (no inplace reuse)
        feat_buffer = t.zeros(
            (batch_size, seq_len, self.n_features), device=device, dtype=dtype
        )

        for down in self.submodule_names:
            # Initialize approx_acts (out-of-place)
            if down.submodule_typ == "attn":
                approx_acts = self.get_initial_contribs_attn(cache, down)
            elif down.submodule_type == "mlp":  # mlp
                approx_acts = self.get_initial_contribs_mlp(cache, down)

            # Initialize upstream_bias (out-of-place)
            upstream_bias = t.zeros(
                self.model.cfg.d_model, device=device, dtype=dtype
            )

            for up in self.submodule_names:
                if up.layer > down.layer or (
                    up.layer == down.layer
                    and (up.submodule_type == "mlp" or up == down)
                ):
                    continue

                if self.connections is None or (
                    isinstance(self.connections, dict)
                    and down.name in self.connections
                    and up.name in self.connections[down.name]
                ):
                    up_feats = pruned_features[up.name]

                    # Avoid inplace: upstream_bias = upstream_bias + ...
                    upstream_bias = upstream_bias + self.aes[up.name].b_dec

                    if down.submodule_type == "attn":
                        contributions = self.get_pruned_contribs_attn(
                            cache, up.name, down.name, up_feats
                        )
                    else:
                        contributions = self.get_pruned_contribs_mlp(
                            cache, up.name, down.name, up_feats
                        )

                    # Avoid inplace: approx_acts = approx_acts + contributions
                    approx_acts = approx_acts + contributions
                    del contributions
                    t.cuda.empty_cache()

            # Messy bias stuff. Beware bugs.
            if down.submodule_type == "attn":
                approx_acts = (
                    approx_acts
                    + self.model.b_O[down.layer].squeeze().to(self.dtype)
                    @ self.aes[down.name].encoder.weight.T
                )

            approx_acts = self._compute_bias(
                down, approx_acts, upstream_bias, cache
            )

            # Get top k features (no inplace scatter)
            top_vals, top_idx = approx_acts.topk(self.k, dim=-1)
            top_vals = t.relu(top_vals)

            # Avoid inplace zeroing: create a new feat_buffer
            feat_buffer = t.zeros_like(feat_buffer)

            # Avoid inplace scatter: create a new tensor
            feat_buffer = feat_buffer.scatter(-1, top_idx, top_vals)

            # Store pruned features (clone to avoid accidental inplace later)
            pruned_features[down.name] = feat_buffer.clone()

            # Decode and store reconstruction
            reconstructions[down.name] = self.aes[down.name].decode(feat_buffer)

            del top_vals, top_idx
            t.cuda.empty_cache()

        if return_features:
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
        resid_final += cache["blocks.0.hook_resid_pre"]

        unembed = self.model.lm_head
        logits = unembed(
            self.model.ln_final(resid_final)
        )  # [batch, seq, n_vocab]

        # https://github.com/apple/ml-cross-entropy
        # GPTNeo TinyStories has no unembed bias.
        loss = linear_cross_entropy(logits, unembed.weight, tokens, shift=1)

        return loss
