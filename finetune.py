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

from dictionary_learning.scae import SCAESuite

scae = SCAESuite(model, k, n_features, connections=connections)