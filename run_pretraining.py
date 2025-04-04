#%%
from buffer import SimpleBuffer
from training import train_scae_suite
from utils import load_iterable_dataset
from find_top_connections import generate_fake_connections

import torch as t
from huggingface_hub import login

# Jacob's token but feel free to use
login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"

DTYPE = t.float32
MODEL_NAME = "roneneldan/TinyStories-33M"
num_tokens = int(500e6)
batch_size = 256
expansion = 4
ctx_len = 128


#%%
data = load_iterable_dataset('roneneldan/TinyStories')


buffer = SimpleBuffer(
    data=data,
    model_name=MODEL_NAME,
    device="cuda",
    batch_size=batch_size,
    dtype=DTYPE,
    ctx_len=ctx_len
) 


#%%
trainer = train_scae_suite(
    buffer,
    vanilla=True,
    model_name=MODEL_NAME,
    k=128,
    expansion=expansion,
    loss_type="mse",
    connections=None,
    steps=num_tokens // (batch_size * ctx_len),
    save_steps = 1000,
    dtype = DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_in=None,
    repo_id_out = "jacobcd52/TinyStories-33M_suite_4",
    wandb_project_name="TinyStories-33m_pretrain",
)
# %%
