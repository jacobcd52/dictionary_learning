from typing import Optional, Tuple, Union, NamedTuple, List

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    logging,
)
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig


logger = logging.get_logger(__name__)


class LayerHiddenStates(NamedTuple):
    mlp_out: torch.Tensor
    attn_out: torch.Tensor
    attn_weights: torch.Tensor
    ln_1_scale: float
    ln_2_scale: float

class ModelHiddenStates(NamedTuple):
    embed_out: torch.Tensor
    all_hidden_states: List[LayerHiddenStates]


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size, config.hidden_size
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value)

    # Reshape outputs
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GPTNeoXAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.head_size = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.scaling = self.head_size**-0.5
        self.is_causal = True
        self.layer_idx = layer_idx

        self.query_key_value = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.attention_bias,
        )
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        qkv = (
            self.query_key_value(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Compute attention
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        # Reshape outputs and final projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        return attn_output, attn_weights


class GPTNeoXLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPTNeoXAttention(config, layer_idx)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        position_embeddings: torch.Tensor,
    ):
        attn_output, attn_weights = self.attention(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
        )
        attn_output = self.post_attention_dropout(attn_output)

        # Parallel residual by default
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states

        layer_hidden_states = LayerHiddenStates(
            mlp_out=mlp_output,
            attn_out=attn_output,
            attn_weights=attn_weights,
            ln_1_scale=torch.tensor(0.0),
            ln_2_scale=torch.tensor(0.0),
        )
        return hidden_states, layer_hidden_states


class GPTNeoXRotaryEmbedding(nn.Module):
    def __init__(self, config: GPTNeoXConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer(
                "inv_freq", self.original_inv_freq, persistent=False
            )
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GPTNeoXPreTrainedModel(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoXLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    _keys_to_ignore_on_load_unexpected = [
        r"attention.bias",
        r"attention.masked_bias",
    ]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPTNeoXModel(GPTNeoXPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GPTNeoXDecoderLayer`]

    Args:
        config: GPTNeoXConfig
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [GPTNeoXLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.rotary_emb = GPTNeoXRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        inputs_embeds = self.embed_in(input_ids)

        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        hidden_states = self.emb_dropout(inputs_embeds)
        embed_out = hidden_states.clone()

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = []
        for i, layer in enumerate(self.layers):
            hidden_states, layer_hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

            all_hidden_states.append(layer_hidden_states)

        # NOTE: Skip for now
        # hidden_states = self.final_layer_norm(hidden_states)

        model_hidden_states = ModelHiddenStates(
            embed_out=embed_out,
            all_hidden_states=all_hidden_states,
        )

        return model_hidden_states

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["embed_out.weight"]
    _tp_plan = {"embed_out": "colwise_rep"}
    _pp_plan = {"embed_out": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.gpt_neox(
            input_ids,
        )

        # Exclude logits
        # hidden_states = outputs[0]
        # logits = self.embed_out(hidden_states)

        return outputs


__all__ = [
    "GPTNeoXForCausalLM",
    "GPTNeoXLayer",
    "GPTNeoXModel",
    "GPTNeoXPreTrainedModel",
]
