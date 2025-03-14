# %%
from transformers import AutoTokenizer
from dictionary_learning.gpt_neo import GPTNeoModel
import torch as t
from datasets import load_dataset

t.set_grad_enabled(False)

model = GPTNeoModel.from_pretrained(
    "roneneldan/TinyStories-33M",
    torch_dtype=t.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
)
model.fold_ln("cuda", t.bfloat16)


# %%

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:1%]")

batch = tokenizer(
    dataset["text"][:200], return_tensors="pt", padding=True, truncation=True
).to("cuda")
tokens = batch["input_ids"]
mask = ~(tokens == tokenizer.pad_token_id).any(dim=1)
tokens = tokens[mask][:2]
print(tokens.shape)

# %%

outputs = model(tokens)


# %%


from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("roneneldan/TinyStories", split="train[:20%]")
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer.pad_token = tokenizer.eos_token

from typing import Dict
from multiprocessing import cpu_count


def chunk_and_tokenize(
    data: Dict[str, list],
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

    data = dataset.map(
        _tokenize_fn,
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
    )

    data = data.with_format(
        type="torch",
        columns=["input_ids"],
    )

    data = data.filter(
        lambda x: len(x["input_ids"]) == max_length,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )

    return data


data = chunk_and_tokenize(dataset, "text", 512)

# %%
