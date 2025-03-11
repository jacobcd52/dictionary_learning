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
k = 64
expansion = 4

num_tokens = int(100e6)
batch_size = 256
in_type = ""
out_type = "ce"


#%%
for num_connections in [10, 30, 100]:
    if in_type == "":
        repo_id_in = None
    else:
        repo_id_in = f"jacobcd52/TinyStories-33M_scae_{num_connections}_{in_type}"
    repo_id_out = f"jacobcd52/TinyStories-33M_scae_{num_connections}_{in_type}_{out_type}"

    #%%
    data = load_iterable_dataset('roneneldan/TinyStories')

    buffer = SimpleBuffer(
        data=data,
        model_name=MODEL_NAME,
        device=device,
        batch_size=batch_size,
        dtype=DTYPE,
        ctx_len=ctx_len,
        prepend_bos=False
    ) 

    #%%
    if in_type == "":
        with open(f"/root/dictionary_learning/tinystories_connections/top_connections_{num_connections}.pkl", "rb") as f:
            connections = pickle.load(f)
        k=k
        expansion=expansion
    else:
        connections = None
        k = None
        expansion = None
    

    #%%
    trainer = train_scae_suite(
        buffer,
        k=k,
        expansion=expansion,
        model_name=MODEL_NAME,
        base_lr=1e-3,
        loss_type=out_type,
        connections=connections,
        steps=num_tokens // (batch_size * ctx_len),
        save_steps = 1000,
        dtype = DTYPE,
        device=device,
        log_steps = 1,
        use_wandb = True,
        repo_id_in=repo_id_in,
        repo_id_out = repo_id_out,
        wandb_project_name="tinystories33m_scae_6",
        wandb_run_name=f"c{num_connections} {in_type} {out_type}",
    )
# %%