# %%
from transformers import AutoTokenizer
from dictionary_learning.buffer import chunk_and_tokenize
import torch as t
import pickle
from datasets import load_dataset
from transformer_lens import HookedTransformer
from dictionary_learning.buffer import Buffer

with t.no_grad():
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
    ).to("cuda").to(t.bfloat16)


# %%

for p in model.parameters():
    p.requires_grad = False

# %%

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:1%]")
dataset = chunk_and_tokenize(dataset, tokenizer, "text", 256)

n_connections = 20
path = f"/root/dictionary_learning/pythia_connections/Copy of top_connections_{n_connections}.pkl"

with open(path, "rb") as f:
    connections = pickle.load(f)

k = 64
expansion = 4
n_features = model.cfg.d_model * expansion

buffer = Buffer(model, dataset, 16, 16)


# %%

from dictionary_learning.scae import SCAESuite, MergedSCAESuite

scae = SCAESuite(model, k, n_features, connections=connections)
# merged_scae = MergedSCAESuite(model, scae)

# %%

from dictionary_learning.trainer import train_scae, TrainerConfig

cfg = TrainerConfig()

train_scae(scae, buffer, cfg)

# %%
