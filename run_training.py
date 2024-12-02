from buffer import AllActivationBuffer
from trainers.top_k import TrainerSCAE, AutoEncoderTopK
from training import trainSCAE

from datasets import load_dataset
import torch as t
from nnsight import LanguageModel

#
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


C = 10
expansion = 8
k = 64 # TODO

num_features = model.config.n_embd * expansion


n_layer = model.config.n_layer

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
    n_ctxs=128,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size = 1024,
    refresh_batch_size = 256,
) 



trainer_cfg = {
    "trainer": TrainerSCAE,
    "activation_dims": {name: model.config.n_embd for name in submodule_names},
    "dict_sizes": {name: model.config.n_embd * expansion for name in submodule_names},
    "ks": {name: k for name in submodule_names},
    "device": buffer.device,
    "submodules": submodules,
    "important_features": None,
}

# Run the training
trainer = trainSCAE(
    buffer=buffer,
    trainer_cfg=trainer_cfg,
    steps=30000,
    save_steps=1000,
    save_dir="sae_checkpoints",
    log_steps=100,
    use_wandb=True,  # Set to False if you don't want to use wandb
    use_sparse_connections=True,
)

