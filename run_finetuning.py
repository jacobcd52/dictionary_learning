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
C = 10
expansion = 16
k = 128 # TODO auto-detect if loading from pretrained

out_batch_size = 128
num_tokens = int(1e7)



#%%
model = load_model_with_folded_ln2(MODEL_NAME, device=device, torch_dtype=DTYPE)
data = load_iterable_dataset('Skylion007/openwebtext')

num_features = model.config.n_embd * expansion
n_layer = model.config.n_layer


#%%
initial_submodule = model.transformer.h[0]
submodules = {}
for layer in range(n_layer):
    submodules[f"mlp_{layer}"] = (model.transformer.h[layer].mlp, "in_and_out")
    submodules[f"attn_{layer}"] = (model.transformer.h[layer].attn, "out")

submodule_names = list(submodules.keys())

buffer = AllActivationBuffer(
    data=data,
    model=model,
    submodules=submodules,
    initial_submodule=initial_submodule,
    d_submodule=model.config.n_embd, # output dimension of the model component
    n_ctxs=1024,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size = out_batch_size,
    refresh_batch_size = 256,
) 

#%%
pretrained_configs = {}
connections = defaultdict(dict)

for down_layer in range(n_layer):
    for module in ['attn', 'mlp']:
        down_name = f'{module}_{down_layer}'
        pretrained_configs[f'{module}_{down_layer}'] = {
            'repo_id': 'jacobcd52/gpt2_suite_folded_ln', 
            'filename': f'ae_{module}_{down_layer}.pt',
            'k' : k
            }
        
        # Use random connections for testing
        if module=='mlp':
            for up_layer in range(down_layer+1): # mlp sees attn from same layer
                up_name = f'{module}_{up_layer}'
                connections[down_name][up_name] = t.randint(0, num_features, (num_features, C), dtype=t.long)


trainer_cfg = TrainerConfig(
    connection_sparsity_coeff=1.0,
    steps=num_tokens // out_batch_size,
)

#%%
trainer = train_scae_suite(
    buffer,
    module_specs=pretrained_configs,
    trainer_config=trainer_cfg,
    steps=num_tokens // out_batch_size,
    save_steps = 1000,
    dtype = DTYPE,
    device=device,
    # save_dir: Optional[str] = None,
    log_steps = 20,
    use_wandb = True,
    hf_repo_id = "jacobcd52/gpt2_scae",
    # seed: Optional[int] = None,
)
# %%
