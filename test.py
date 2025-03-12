# %%
from transformers import AutoTokenizer
from gptneo import GPTNeoForCausalLM
from transformer_lens import HookedTransformer
import torch as t

model = GPTNeoForCausalLM.from_pretrained(
    "roneneldan/TinyStories-33M",
    torch_dtype=t.bfloat16,
    attn_implementation="eager",
    use_cache=False,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer.pad_token = tokenizer.eos_token

tl_model = HookedTransformer.from_pretrained("roneneldan/TinyStories-33M")
tl_model = tl_model.to("cuda").to(t.bfloat16)

# %%

from datasets import load_dataset

dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:1%]")

batch = tokenizer(
    dataset["text"][:200], return_tensors="pt", padding=True, truncation=True
).to("cuda")
tokens = batch["input_ids"]
mask = ~(tokens == tokenizer.pad_token_id).any(dim=1)
tokens = tokens[mask]


# %%
import time

t.set_grad_enabled(False)

times = []

for i in range(10):
    start = time.time()

    out = model(tokens)

    end = time.time()
    times.append(end - start)

avg = t.mean(t.tensor(times)).item()
std = t.std(t.tensor(times)).item()
print(f"{avg=}, {std=}")
