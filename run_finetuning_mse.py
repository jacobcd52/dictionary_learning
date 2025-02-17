#%%
from buffer import SimpleBuffer
from training import train_scae_suite
from utils import load_model_with_folded_ln2, load_iterable_dataset
from find_top_connections import generate_fake_connections

import torch as t
from huggingface_hub import login
import pickle

# Jacob's token but feel free to use
login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"

#%%
DTYPE = t.bfloat16
MODEL_NAME = "roneneldan/TinyStories-33M"
num_tokens = int(4096*5000)
batch_size = 2
expansion = 4


#%%
data = load_iterable_dataset('roneneldan/TinyStories')

buffer = SimpleBuffer(
    data=data,
    model_name=MODEL_NAME,
    device="cuda",
    batch_size=batch_size,
    dtype=DTYPE,
) 



#%%
# Use some dumb fake connections for now
with open("/root/dictionary_learning/connections_TinyStories-33M_100.pkl", "rb") as f:
    connections = pickle.load(f)

for layer in range(buffer.model.cfg.n_layers):
    connections[f'attn_{layer}'] = {k: v for (k, v) in connections[f'mlp_{layer}'].items() 
                                    if int(k.split('_')[1]) < layer}

for down_name in connections.keys():
    for up_name in connections[down_name].keys():
        connections[down_name][up_name] = connections[down_name][up_name][:768*expansion]

fake_connections = generate_fake_connections(
    connections,
    num_features=768*expansion
)

#%%
trainer = train_scae_suite(
    buffer,
    model_name=MODEL_NAME,
    k=128,
    n_features=768*expansion,
    loss_type="mse",
    connections=fake_connections, # "all"
    steps=num_tokens // batch_size,
    save_steps = 1000,
    dtype = DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_in=None, #'jacobcd52/TinyStories-33M_suite',
    repo_id_out = "jacobcd52/TinyStories-33M_scae",
    wandb_project_name="tinystories33m_scae_3",
)
# %%
