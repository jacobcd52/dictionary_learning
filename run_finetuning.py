#%%
from buffer import AllActivationBuffer
from training import train_scae_suite
from trainers.scae import TrainerConfig, SubmoduleConfig
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
MODEL_NAME = "gpt2"

out_batch_size = 32
num_tokens = int(1e7)
remove_bos=True # don't train on BOS activations: they lead to weird loss spikes, especially with bf16.

# Only need these if training from scratch
expansion = 16
k = 128


#%%
model = load_model_with_folded_ln2(MODEL_NAME, device=device, torch_dtype=DTYPE)
data = load_iterable_dataset('Skylion007/openwebtext')

num_features = model.config.n_embd * expansion
n_layer = model.config.n_layer



#%%
initial_submodule = model.transformer.h[0]
submodules = {}
layernorm_submodules = {}
for layer in range(n_layer):
    submodules[f"mlp_{layer}"] = (model.transformer.h[layer].mlp, "in_and_out")
    submodules[f"attn_{layer}"] = (model.transformer.h[layer].attn, "out")
    layernorm_submodules[f"mlp_{layer}"] = model.transformer.h[layer].ln_2
submodule_names = list(submodules.keys())

buffer = AllActivationBuffer(
    data=data,
    model=model,
    submodules=submodules,
    initial_submodule=initial_submodule,
    layernorm_submodules=layernorm_submodules,
    d_submodule=model.config.n_embd, # output dimension of the model component
    n_ctxs=512,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size = out_batch_size,
    refresh_batch_size = 256,
    dtype=DTYPE,
    remove_bos=remove_bos
) 


#%%
trainer_cfg = TrainerConfig(
    steps=num_tokens // out_batch_size,
    n_threshold=0,
    n_random=0,
    random_loss_coeff=0.0
)

# Load connections from connections_100.pkl
with open("connections_100.pkl", "rb") as f:
    connections = pickle.load(f)

# If using random connections
# fake_connections = generate_fake_connections(
#     connections,
#     num_features=num_features,
# )

# If training from scratch
# submodule_cfg = SubmoduleConfig(
#             dict_size=num_features,
#             activation_dim=model.config.n_embd,
#             k=k)
# submodule_configs = {f'{module}_{down_layer}' : submodule_cfg for down_layer in range(n_layer) for module in ['attn', 'mlp']}



#%%
trainer = train_scae_suite(
    buffer,
    trainer_config=trainer_cfg,
    submodule_configs=None, #submodule_configs, # use if training from scratch
    connections=connections,
    steps=num_tokens // out_batch_size,
    save_steps = 1000,
    dtype = DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_in='jacobcd52/gpt2_suite_folded_ln',
    repo_id_out = "jacobcd52/gpt2_scae",
    wandb_project_name="gpt2_scae_finetuning",
)
# %%
