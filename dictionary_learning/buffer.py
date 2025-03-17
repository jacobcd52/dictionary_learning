import torch as t
from transformers import PreTrainedTokenizerBase
from typing import Dict, List
from datasets import Dataset
from multiprocessing import cpu_count

from .gpt_neo import GPTNeoModel, GPTNeoHiddenStates

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


def format_cache(hidden_states: List[GPTNeoHiddenStates]) -> Dict[str, t.Tensor]:
    cache = {
        "blocks.0.hook_resid_pre": hidden_states.embed_out,
    }

    for layer_idx, block_hidden_states in enumerate(
        hidden_states.block_hidden_states
    ):
        layer_cache = {
            f"blocks.{layer_idx}.ln1.hook_scale": block_hidden_states.ln_1_scale,
            f"blocks.{layer_idx}.ln2.hook_scale": block_hidden_states.ln_2_scale,
            f"blocks.{layer_idx}.hook_attn_out": block_hidden_states.attn_output,
            f"blocks.{layer_idx}.hook_mlp_out": block_hidden_states.mlp_output,
            f"blocks.{layer_idx}.attn.hook_pattern": block_hidden_states.attn_weights,
        }

        cache.update(layer_cache)

    return cache


class Buffer:
    def __init__(
        self,
        model: GPTNeoModel,
        dataset: Dataset,
        batch_out_size: int,
        refresh_batch_size: int,
    ):
        if batch_out_size % refresh_batch_size != 0:
            raise ValueError(
                "batch_out_size must be divisible by refresh_batch_size",
                f"got {batch_out_size} and {refresh_batch_size}",
            )

        self.batch_out_size = batch_out_size
        self.acc_steps = batch_out_size // refresh_batch_size

        self.dataloader = DataLoader(
            dataset,
            batch_size=refresh_batch_size,
        )
        self.model = model

    def __next__(self):
        hidden_states = []
        for _ in range(self.acc_steps):
            batch = next(self.dataloader)
            hidden_states.append(self.model(batch["input_ids"]))

        return t.cat(hidden_states, dim=0)
