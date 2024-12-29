#%%
from buffer import AllActivationBuffer
from training import train_scae_suite
from trainers.scae import TrainerSCAESuite, TrainerConfig, SubmoduleConfig
from utils import load_model_with_folded_ln2, load_iterable_dataset

from datasets import load_dataset
import torch as t
from nnsight import LanguageModel
from collections import defaultdict
from huggingface_hub import login

login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"



#%%
DTYPE = t.bfloat16
MODEL_NAME = "gpt2"
expansion = 16
k = 128 # TODO auto-detect if loading from pretrained

out_batch_size = 64
num_tokens = int(1e7)



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
) 

#%%
pretrained_configs = {}
# Load connections from connections_100.pkl
import pickle
with open("notebooks/connections_100.pkl", "rb") as f:
    connections = pickle.load(f)


trainer_cfg = TrainerConfig(
    connection_sparsity_coeff=5.0,
    steps=num_tokens // out_batch_size,
)

#%%
trainer = train_scae_suite(
    buffer,
    trainer_config=trainer_cfg,
    submodule_configs=None,
    connections=connections,
    steps=num_tokens // out_batch_size,
    save_steps = 1000,
    dtype = DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_in='jacobcd52/gpt2_suite_folded_ln',
    repo_id_out = "jacobcd52/gpt2_scae",
    # seed: Optional[int] = None,
)
# %%
