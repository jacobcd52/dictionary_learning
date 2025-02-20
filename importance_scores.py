#%%
import sys 
sys.path.append('../')
from buffer import SimpleBuffer
from training import train_scae_suite
from utils import load_model_with_folded_ln2, load_iterable_dataset
from find_top_connections import generate_fake_connections, get_avg_contribs, get_top_connections
from trainers.scae import SCAESuite
from find_top_connections import get_avg_contribs

import torch as t
from huggingface_hub import login
import pickle
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
# Jacob's token but feel free to use
login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"

t.set_grad_enabled(False)

#%%
DTYPE = t.bfloat16
MODEL_NAME = "roneneldan/TinyStories-33M"
num_tokens = int(1e6)
batch_size = 32
expansion = 4
ctx_len = 128


#%%
data = load_iterable_dataset('roneneldan/TinyStories')

buffer = SimpleBuffer(
    data=data,
    model_name=MODEL_NAME,
    ctx_len=ctx_len,
    device="cuda",
    batch_size=batch_size,
    dtype=DTYPE,
) 

#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=DTYPE)
suite = SCAESuite.from_pretrained(
    "jacobcd52/TinyStories-33M_suite_4",
    model,
    device=device,
    dtype=DTYPE,
)

#%%
avg_contribs = get_avg_contribs(suite, buffer, n_batches=1)


#%%
from tqdm import tqdm
for c in tqdm([10, 20, 40, 80, 160, 320, 640]):
    get_top_connections = get_top_connections(avg_contribs, c=10)
    # save as pickle file
    with open(f"top_connections_{c}.pkl", "wb") as f:
        pickle.dump(get_top_connections, f)