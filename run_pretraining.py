#%%
from buffer import AllActivationBuffer
from training import train_scae_suite
from trainers.scae import TrainerConfig, SubmoduleConfig
from utils import load_model_with_folded_ln2, load_iterable_dataset

import torch as t
from huggingface_hub import login
login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"


#%%
DTYPE = t.float32
MODEL_NAME = "roneneldan/TinyStories-33M"
expansion = 16
k = 128
remove_bos = True

out_batch_size = 4096 * 2
num_tokens = int(2e8)


#%%
model = load_model_with_folded_ln2(MODEL_NAME, device=device, torch_dtype=DTYPE)
data = load_iterable_dataset("roneneldan/TinyStories")
# 'Skylion007/openwebtext'
if MODEL_NAME == "gpt2":
    n_embd = model.config.n_embd
    num_features = n_embd * expansion
    n_layer = model.config.n_layer
elif MODEL_NAME == "roneneldan/TinyStories-33M":
    n_embd = model.config.hidden_size
    num_features = n_embd * expansion
    n_layer = model.config.num_layers

#%%
buffer = AllActivationBuffer(
    data=data,
    model=model,
    model_name=MODEL_NAME,
    n_ctxs=1024,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size = out_batch_size,
    refresh_batch_size = 512,
    remove_bos=remove_bos
) 

#%%
submodule_cfg = SubmoduleConfig(
            dict_size=num_features,
            activation_dim=n_embd,
            k=k)
submodule_configs = {f'{module}_{down_layer}' : submodule_cfg for down_layer in range(n_layer) for module in ['attn', 'mlp']}

print("total steps = ", num_tokens // out_batch_size)
trainer_cfg = TrainerConfig(
    steps=num_tokens // out_batch_size,
    use_vanilla_training=True,
    base_lr=5e-4
)

#%%
trainer = train_scae_suite(
    buffer,
    submodule_configs=submodule_configs,
    trainer_config=trainer_cfg,
    steps=num_tokens // out_batch_size,
    save_steps = 1000,
    dtype=DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_out = "jacobcd52/TinyStories-33M_suite",
    seed = 42,
    wandb_project_name="TinyStories-33m_pretrain"
)
# %%
