#%%
from buffer import SimpleBuffer
from training import train_scae_suite
from utils import load_iterable_dataset
from find_top_connections import generate_fake_connections

import torch as t
from huggingface_hub import login
import pickle

login("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")
device = "cuda:0"

#%%
DTYPE = t.bfloat16
MODEL_NAME = "roneneldan/TinyStories-33M"
ctx_len = 128

num_tokens = int(100e6)
batch_size = 256
loss_type = "ce"


#%%
for num_connections in [10, 30, 100, 300]:
    if loss_type == "ce":
        repo_id_in = f"jacobcd52/TinyStories-33M_scae_{num_connections}_mse"
        repo_id_out = f"jacobcd52/TinyStories-33M_scae_{num_connections}_ce"
    elif loss_type == "mse":
        repo_id_in = "jacobcd52/TinyStories-33M_suite_4"
        repo_id_out = f"jacobcd52/TinyStories-33M_scae_{num_connections}_mse"
    #%%
    data = load_iterable_dataset('roneneldan/TinyStories')

    buffer = SimpleBuffer(
        data=data,
        model_name=MODEL_NAME,
        device=device,
        batch_size=batch_size,
        dtype=DTYPE,
        ctx_len=ctx_len
    ) 

    #%%
    with open(f"/root/dictionary_learning/tinystories_connections/top_connections_{num_connections}.pkl", "rb") as f:
        connections = pickle.load(f)

    #%%
    trainer = train_scae_suite(
        buffer,
        model_name=MODEL_NAME,
        base_lr=1e-3,
        loss_type=loss_type,
        # connections=connections,
        steps=num_tokens // (batch_size * ctx_len),
        save_steps = 1000,
        dtype = DTYPE,
        device=device,
        log_steps = 1,
        use_wandb = True,
        repo_id_in=repo_id_in,
        repo_id_out = repo_id_out,
        wandb_project_name="tinystories33m_scae_5",
        wandb_run_name=f"c{num_connections} bs{batch_size} {loss_type}",
    )
# %%