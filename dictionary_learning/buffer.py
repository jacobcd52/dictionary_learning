import torch as t
from transformers import PreTrainedTokenizerBase
from typing import Dict, List
from datasets import Dataset
from multiprocessing import cpu_count

from .gpt_neox import GPTNeoXModel, ModelHiddenStates

from torch.utils.data import DataLoader


def chunk_and_tokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_key: str,
    max_length: int,
    num_proc: int = cpu_count() // 2,
    load_from_cache_file: bool = True,
):
    def _tokenize_fn(row: Dict[str, list]):
        output = tokenizer(
            row[text_key],
            max_length=max_length,
            return_attention_mask=False,
            truncation=True,
        )

        return output

    dataset = dataset.map(
        _tokenize_fn,
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
    )

    dataset = dataset.with_format(
        type="torch",
        columns=["input_ids"],
    )

    dataset = dataset.filter(
        lambda x: len(x["input_ids"]) == max_length,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )

    setattr(dataset, "_prepared", True)

    return dataset


def format_cache(hidden_states: List[ModelHiddenStates]) -> Dict[str, t.Tensor]:
    embed_out = t.cat([hs.embed_out for hs in hidden_states])

    cache = {
        "embed_out": embed_out,
    }

    for layer_idx, layer_hidden_states in enumerate(
        zip(*[hs.all_hidden_states for hs in hidden_states])
    ):
        # ln_1_scale = t.cat([hs.ln_1_scale for hs in layer_hidden_states])
        # ln_2_scale = t.cat([hs.ln_2_scale for hs in layer_hidden_states])
        attn_out = t.cat([hs.attn_out for hs in layer_hidden_states])
        mlp_out = t.cat([hs.mlp_out for hs in layer_hidden_states])
        attn_weights = t.cat([hs.attn_weights for hs in layer_hidden_states])

        layer_cache = {
            f"blocks.{layer_idx}.ln1_scale": 0,
            f"blocks.{layer_idx}.ln2_scale": 0,
            f"blocks.{layer_idx}.hook_attn_out": attn_out,
            f"blocks.{layer_idx}.hook_mlp_out": mlp_out,
            f"blocks.{layer_idx}.attn.hook_pattern": attn_weights,
        }

        cache.update(layer_cache)

    return cache


class Buffer:
    def __init__(
        self,
        model: GPTNeoXModel,
        dataset: Dataset,
        batch_out_size: int,
        refresh_batch_size: int,
    ):
        if batch_out_size % refresh_batch_size != 0:
            raise ValueError(
                "Batch_out_size must be divisible by refresh_batch_size",
                f"got {batch_out_size} and {refresh_batch_size}",
            )

        if not hasattr(dataset, "_prepared"):
            raise ValueError("Please prepare with chunk_and_tokenize first")

        self.batch_out_size = batch_out_size
        self.acc_steps = batch_out_size // refresh_batch_size

        self.dataloader = iter(
            DataLoader(
                dataset,
                batch_size=refresh_batch_size,
            )
        )
        self.model = model

    @t.no_grad()
    def __next__(self):
        hidden_states = []
        for _ in range(self.acc_steps):
            batch = next(self.dataloader)
            input_ids = batch["input_ids"].to("cuda")
            hidden_states.append(self.model(input_ids))

        return format_cache(hidden_states)
