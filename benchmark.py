# %%

import time

from transformers import AutoTokenizer
from dictionary_learning.gpt_neox import GPTNeoXForCausalLM
from dictionary_learning.buffer import Buffer, chunk_and_tokenize
import torch as t
from dictionary_learning.old_buffer import SimpleBuffer
from datasets import load_dataset

model_id = "EleutherAI/pythia-70m-deduped"

with t.no_grad():
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
    ).to("cuda", t.bfloat16)
    # model = t.compile(model)

# %%

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

dummy_input = tokenizer("Hello, world!", return_tensors="pt").to("cuda")['input_ids']
_ = model(dummy_input)

raw_dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:10%]")
dataset = chunk_and_tokenize(raw_dataset, tokenizer, "text", 128)

text_dataset = iter(raw_dataset["text"])

# %%

simple_buffer = SimpleBuffer(
    text_dataset,
    model_id,
    ctx_len=128,
    batch_size=256,
    prepend_bos=False,
    device="cuda",
    dtype=t.bfloat16,
)
buffer = Buffer(model, dataset, 256, 256)

# %%

import numpy as np
def time_buffer(buffer, n_iters=50):
    times = []
    total_start = time.time()
    for _ in range(n_iters):
        start = time.time()
        next(buffer)
        end = time.time()
        times.append(end - start)
    total_end = time.time()

    total_time = total_end - total_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, total_time

simple_time, simple_std, simple_total = time_buffer(simple_buffer)
buffer_time, buffer_std, buffer_total = time_buffer(buffer)

print(f"Simple buffer time: {simple_time} +- {simple_std} seconds")
print(f"Buffer time: {buffer_time} +- {buffer_std} seconds")

print(f"Simple buffer total time: {simple_total} seconds")
print(f"Buffer total time: {buffer_total} seconds")

# %%

scale = 3_000 / 50
scaled_simple_total = simple_total * scale
scaled_buffer_total = buffer_total * scale

print(f"Scaled simple buffer total time: {scaled_simple_total/60} minutes")
print(f"Scaled buffer total time: {scaled_buffer_total/60} minutes")