from multiprocessing import cpu_count
from typing import Dict, List

from datasets import Dataset
import torch as t
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer, ActivationCache


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


class Buffer:
    def __init__(
        self,
        model: HookedTransformer,
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

        self.hook_list = ["blocks.0.hook_resid_pre"]
        for layer in range(self.model.cfg.n_layers):
            self.hook_list += [
                f"blocks.{layer}.ln1.hook_scale",
                f"blocks.{layer}.ln2.hook_scale",
                # f"blocks.{layer}.ln1.hook_normalized",
                # f"blocks.{layer}.ln2.hook_normalized",
                # f"blocks.{layer}.hook_attn_out",
                # f"blocks.{layer}.hook_mlp_out",
                f"blocks.{layer}.attn.hook_pattern",
            ]

    @t.no_grad()
    def __next__(self):
        caches = []
        for _ in range(self.acc_steps):
            batch = next(self.dataloader)
            input_ids = batch["input_ids"].to("cuda")
            _, cache = self.model.run_with_cache(
                input_ids, return_type=None, names_filter=self.hook_list
            )
            caches.append(cache)

        return self._format_cache(caches)

    def _format_cache(self, caches: List[ActivationCache]) -> Dict[str, t.Tensor]:
        if len(caches) == 1:
            full_cache = caches[0]
            for hook_name in self.hook_list:
                if (".ln" in hook_name) or (".hook_pattern" in hook_name):
                    full_cache[hook_name] = full_cache[hook_name].to(t.bfloat16)

        else:
            full_cache = {}
            for hook_name in self.hook_list:
                hidden_states = t.cat([cache[hook_name] for cache in caches])
                if (".ln" in hook_name) or (".hook_pattern" in hook_name):
                    hidden_states = hidden_states.to(t.bfloat16)
                full_cache[hook_name] = hidden_states

        return full_cache
