#%%
from buffer import AllActivationBuffer
from trainers.top_k import TrainerSCAE, AutoEncoderTopK
from training import trainSCAE

from datasets import load_dataset
import torch as t
from nnsight import LanguageModel

from huggingface_hub import login
login("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")


#%%
device = "cuda:0" if t.cuda.is_available() else "cpu"
model = LanguageModel("gpt2", device_map=device)
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


#%%
submodules = {}
for layer in range(n_layer):
    submodules[f"mlp_{layer}"] = (model.transformer.h[layer].mlp, "in_and_out")
    submodules[f"attn_{layer}"] = (model.transformer.h[layer].attn, "out")

submodule_names = list(submodules.keys())

buffer = AllActivationBuffer(
    data=data,
    model=model,
    submodules=submodules,
    d_submodule=model.config.n_embd, # output dimension of the model component
    n_ctxs=1024,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size = out_batch_size,
    refresh_batch_size = 256,
) 


#%%
important_features = {} #{f"mlp_{layer}": t.randint(0, num_features, (num_features, C)) for layer in range(n_layer)}

#%%

trainer_cfg = {
    "trainer": TrainerSCAE,
    "activation_dims": {name: model.config.n_embd for name in submodule_names},
    "dict_sizes": {name: model.config.n_embd * expansion for name in submodule_names},
    "ks": {name: k for name in submodule_names},
    "auxk_alpha" : 1/32,
    "device": buffer.device,
    "submodules": submodules,
    "important_features": important_features,
    "connection_sparsity_coeff": 0.01,
    "use_sparse_connections": False,
    "dtype": t.bfloat16
}

# Run the training
trainer = trainSCAE(
    buffer=buffer,
    trainer_cfg=trainer_cfg,
    steps=num_tokens // out_batch_size,
    save_steps=1000,
    save_dir="sae_checkpoints",
    log_steps=100,
    use_wandb=True,  # Set to False if you don't want to use wandb
    hf_repo_id="jacobcd52/scae"
)

# %%
