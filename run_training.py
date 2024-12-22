#%%
from buffer import AllActivationBuffer
from trainers.scae import TrainingConfig, SCAETrainer, ModuleConfig, SCAESuite

from datasets import load_dataset
import torch as t
from nnsight import LanguageModel
from dataclasses import asdict
from huggingface_hub import login

login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"
DTYPE = t.bfloat16

model = LanguageModel("gpt2", device_map=device, torch_dtype=DTYPE)
model.eval()

dataset = load_dataset(
    'Skylion007/openwebtext', 
    split='train', 
    streaming=True,
    trust_remote_code=True
    )

class CustomData():
    def __init__(self, dataset):
        self.data = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)['text']

data = CustomData(dataset)




#%%
C = 10
expansion = 16
k = 128 # TODO

out_batch_size = 4096
num_tokens = int(1e8)

num_features = model.config.n_embd * expansion
n_layer = model.config.n_layer
n_steps = num_tokens // out_batch_size




#%%
submodules = {}
for layer in range(n_layer):
    submodules[f"mlp_{layer}"] = {"input_point" : (model.transformer.h[layer].mlp, "in"),
                                  "output_point" : (model.transformer.h[layer].mlp, "out")}
    submodules[f"attn_{layer}"] = {"input_point" : (model.transformer.h[layer].attn, "out"),
                                    "output_point" : (model.transformer.h[layer].attn, "out")}

submodule_names = list(submodules.keys())

buffer = AllActivationBuffer(
    data=data,
    model=model,
    submodules=submodules, # TODO rename this something like activation_points
    d_submodule=model.config.n_embd, # output dimension of the model component
    n_ctxs=1024,  # you can set this higher or lower depending on your available memory
    out_batch_size = out_batch_size,
    refresh_batch_size = 256,
    device=device,
    dtype=DTYPE,
) 



#%%  Separate architecture config for each SAE
module_configs = {}
for layer in range(n_layer):
    module_configs[f"mlp_{layer}"] = ModuleConfig(
        activation_dim=model.config.n_embd,
        dict_size=num_features,
        k=k,
        connections={}
      )

    module_configs[f"attn_{layer}"] = ModuleConfig(
        activation_dim=model.config.n_embd,
        dict_size=num_features,
        k=k,
        connections={}
    )

training_config = TrainingConfig(
    steps=num_tokens // out_batch_size,
    save_steps=1000,
    save_dir="checkpoints",
    use_wandb=True,
    connection_sparsity_coeff=0.0,
    hf_repo_id="jacobcd52/scae_include_ln",
    log_steps=10
)



#%%
suite = SCAESuite(
    module_configs=module_configs,
    device=device,
    dtype=buffer.dtype
)

trainer = SCAETrainer(
    suite=suite,
    config=training_config,
)
# %%
trainer.train(buffer)
# %%
