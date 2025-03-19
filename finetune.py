# %%
from transformers import AutoTokenizer
from dictionary_learning.buffer import Buffer, chunk_and_tokenize
import torch as t
import pickle
from datasets import load_dataset
from transformer_lens import HookedTransformer

with t.no_grad():
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
    ).to("cuda").to(t.bfloat16)

# %%

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:1%]")
dataset = chunk_and_tokenize(dataset, tokenizer, "text", 256)

buffer = Buffer(model, dataset, 8, 2)

# %%

batch = next(buffer)


# %%

n_connections = 200
path = f"/root/pythia_connections/top_connections_{n_connections}.pkl"

with open(path, "rb") as f:
    connections = pickle.load(f)

k = 64
expansion = 4
n_features = model.cfg.d_model * expansion

# %%

from dictionary_learning.scae import SCAESuite, MergedSCAESuite

scae = SCAESuite(model, k, n_features, connections=connections)
merged_scae = MergedSCAESuite(model, scae)

# %%

merged_scae(dataset[:2]["input_ids"], return_loss=True)


# %%
import time
import numpy as np
def time_old(stuff, n_iters=50):
    times = []
    total_start = time.time()
    for _ in range(n_iters):
        start = time.time()
        stuff.forward_pruned(batch)
        end = time.time()
        times.append(end - start)
    total_end = time.time()

    total_time = total_end - total_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, total_time

def time_new(stuff, n_iters=50):
    times = []
    total_start = time.time()
    for _ in range(n_iters):
        start = time.time()
        stuff(batch)
        end = time.time()
        times.append(end - start)
    total_end = time.time()

    total_time = total_end - total_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, total_time

old_time, old_std, old_total = time_old(scae)
new_time, new_std, new_total = time_new(scae_new)

print(f"Old time: {old_time} +- {old_std} seconds")
print(f"New time: {new_time} +- {new_std} seconds")

