#%%
from buffer import AllActivationBuffer
from training import train_scae_suite
from trainers.scae import TrainerConfig
from utils import load_model_with_folded_ln2, load_iterable_dataset
from find_top_connections import generate_fake_connections

import torch as t
from huggingface_hub import login
import pickle

# Jacob's token but feel free to use
login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"

#%%
DTYPE = t.float32
MODEL_NAME = "roneneldan/TinyStories-33M"
out_batch_size = 4096
num_tokens = int(4096*5000)
remove_bos = True # don't train on BOS activations: they lead to weird loss spikes, especially with bf16.

#%%
trainer_cfg = TrainerConfig(
    steps=num_tokens // out_batch_size,
    base_lr=2e-4, 
)

#%%
model = load_model_with_folded_ln2(MODEL_NAME, device=device, torch_dtype=DTYPE)
data = load_iterable_dataset('roneneldan/TinyStories')

buffer = AllActivationBuffer(
    data=data,
    model=model,
    model_name=MODEL_NAME,
    n_ctxs=512,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size=out_batch_size,
    refresh_batch_size=128,
    dtype=DTYPE,
    remove_bos=remove_bos
) 

#%%
# Load connections dict
with open("connections_TinyStories-33M_100.pkl", "rb") as f:
    connections = pickle.load(f)

# connections = generate_fake_connections(connections, num_features=768*16)
# If using random connections
# fake_connections = generate_fake_connections(
#     connections,
#     num_features=num_features,
# )

#%%
trainer = train_scae_suite(
    buffer,
    trainer_config=trainer_cfg,
    connections=connections, # "all"
    steps=num_tokens // out_batch_size,
    save_steps = 1000,
    dtype = DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_in='jacobcd52/TinyStories-33M_suite',
    repo_id_out = "jacobcd52/TinyStories-33M_scae",
    wandb_project_name="tinystories33m_scae_3",
)
# %%
