"""PyTorch GPT Neo model."""

from typing import Optional, NamedTuple, List

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig

logger = logging.get_logger(__name__)


class GPTNeoSelfAttention(nn.Module):
    def __init__(self, config, attention_type, layer_id=None):
        super().__init__()
        self.config = config

        max_positions = config.max_position_embeddings
        bias = torch.tril(
            torch.ones((max_positions, max_positions), dtype=bool)
        ).view(1, 1, max_positions, max_positions)

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = torch.bitwise_xor(
                bias, torch.tril(bias, -config.window_size)
            )

        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer(
            "masked_bias", torch.tensor(-1e9), persistent=False
        )

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))
        self.is_causal = True
        self.layer_id = layer_id

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(
            0, 2, 1, 3
        )  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # Apply sliding window masking for local attention layers
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self, hidden_states):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(query, key, value)

        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return (attn_output, attn_weights)


class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = GPTNeoSelfAttention(
                config, self.attention_type, layer_id
            )
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def forward(self, hidden_states):
        return self.attention(hidden_states)


class GPTNeoMLP(nn.Module):
    def __init__(
        self, intermediate_size, config
    ):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTNeoLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Compute layer norm in float32 to avoid precision issues with calculating variance
        original_dtype = x.dtype
        x = x.to(torch.float32)

        x = x - x.mean(-1, keepdim=True)
        scale = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()

        return ((x / scale).to(original_dtype), scale)


class GPTNeoBlockHiddenStates(NamedTuple):
    attn_output: torch.Tensor
    attn_weights: torch.Tensor
    mlp_output: torch.Tensor
    ln_1_scale: torch.Tensor
    ln_2_scale: torch.Tensor

class GPTNeoHiddenStates(NamedTuple):
    embed_out: torch.Tensor
    block_hidden_states: List[GPTNeoBlockHiddenStates]


class GPTNeoBlock(nn.Module):
    def __init__(self, config, layer_id=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (
            config.intermediate_size
            if config.intermediate_size is not None
            else 4 * hidden_size
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(inner_dim, config)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, attn_weights = self.attn(hidden_states)
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + mlp_output

        return hidden_states, GPTNeoBlockHiddenStates(
            attn_output,
            attn_weights,
            mlp_output,
            0,
            0
        )


class GPTNeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False  # TODO: needs a HybridCache

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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


class GPTNeoModel(GPTNeoPreTrainedModel):
    def __init__(self, config):
        print(
            "Warning: Using flash_attn with the assumption that there are no padding tokens!"
        )

        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(float(config.embed_dropout))
        self.h = nn.ModuleList(
            [GPTNeoBlock(config, layer_id=i) for i in range(config.num_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def fold_ln(self, device: str, dtype: torch.dtype):
        from einops import reduce

        assert self.h[0].attn.attention.q_proj.bias is None, "fold_ln assumes attn has no bias"
        for layer in range(self.config.num_layers):
            # 1) Fold ln_1 into attention weights
            # Note attention weight matrices do not have biases
            W_ln_1 = self.h[layer].ln_1.weight.data.clone()

            W_Q = self.h[layer].attn.attention.q_proj.weight.data.clone()
            W_K = self.h[layer].attn.attention.k_proj.weight.data.clone()
            W_V = self.h[layer].attn.attention.v_proj.weight.data.clone()

            W_Q = (
                W_Q * W_ln_1[None, :, None]
            )
            W_K = (
                W_K * W_ln_1[None, :, None]
            )
            W_V = (
                W_V * W_ln_1[None, :, None]
            )

            # Center weights
            self.h[layer].attn.attention.q_proj.weight.data = W_Q - reduce(
                W_Q,
                "head_index d_model d_head -> head_index 1 d_head",
                "mean",
            )
            self.h[layer].attn.attention.k_proj.weight.data = W_K - reduce(
                W_K,
                "head_index d_model d_head -> head_index 1 d_head",
                "mean",
            )
            self.h[layer].attn.attention.v_proj.weight.data = W_V - reduce(
                W_V,
                "head_index d_model d_head -> head_index 1 d_head",
                "mean",
            )

            # 2) Fold ln_2 into MLP weights

            W_ln_2 = self.h[layer].ln_2.weight.data.clone()
            b_ln_2 = self.h[layer].ln_2.bias.data.clone()
            W_mlp = self.h[layer].mlp.c_fc.weight.data.clone()
            b_mlp = self.h[layer].mlp.c_fc.bias.data.clone()

            b_mlp = b_mlp + (
                W_mlp * b_ln_2[:, None]
            ).sum(-2)

            W_mlp = W_mlp * W_ln_2[:, None]
            
        self.to(device=device, dtype=dtype)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)

        seq_length = inputs_embeds.shape[1]
        position_ids = torch.arange(
            0,
            seq_length,
            device=inputs_embeds.device,
        ).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        embed_out = inputs_embeds + position_embeds

        hidden_states = self.drop(embed_out)
        # output_shape = (-1, seq_length, hidden_states.size(-1))

        all_block_outputs = []
        for block in self.h:
            outputs = block(
                hidden_states,
            )

            hidden_states = outputs[0]
            all_block_outputs.append(outputs[1])

        # hidden_states = self.ln_f(hidden_states)

        # hidden_states = hidden_states.view(output_shape)

        return GPTNeoHiddenStates(
            embed_out=embed_out,
            block_hidden_states=all_block_outputs
        )


class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModel(config)

        # This isn't actually used, we just want to load the weights.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.transformer(input_ids)


__all__ = [
    "GPTNeoModel",
    "GPTNeoPreTrainedModel",
]
