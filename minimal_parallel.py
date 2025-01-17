from accelerate import Accelerator
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import pickle
from tqdm import tqdm
import os 
import json
from dataclasses import asdict
from trainers.scae import SubmoduleConfig, SCAESuite, TrainerSCAESuite, TrainerConfig
from buffer import AllActivationBuffer
from utils import load_model_with_folded_ln2, load_iterable_dataset

accelerator = Accelerator()
device = accelerator.device

# Load model
def load_model():
    DTYPE = torch.float32
    MODEL_NAME = "gpt2"
    model = load_model_with_folded_ln2(
        MODEL_NAME,
        device="cpu",  # load on CPU first
        torch_dtype=DTYPE)
    return model
model = load_model()
model = accelerator.prepare(model)

# Load dataset
data = load_iterable_dataset("Skylion007/openwebtext")

# Setup activation buffer 
initial_submodule = model.transformer.h[0]
submodules = {}
layernorm_submodules = {}
for layer in range(model.config.n_layer):
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
    d_submodule=768,
    n_ctxs=64,
    device="cpu",  
    out_batch_size=48,
    refresh_batch_size=256,
    dtype=torch.float32,
    remove_bos=True
)
buffer.refresh()

# Setup trainer
out_batch_size = 32
num_tokens = int(1e7)
connection_sparsity_coeff = 1.0
trainer_cfg = TrainerConfig(
    connection_sparsity_coeff=connection_sparsity_coeff,
    steps=num_tokens // out_batch_size
)
try:
    with open("connections_100.pkl", "rb") as f:
        connections = pickle.load(f)
except FileNotFoundError:
    connections = None

# Check if we're loading pretrained or initializing fresh
repo_id_in = "jacobcd52/gpt2_suite_folded_ln"
repo_id_out = "jacobcd52/gpt2_scae"
submodule_configs = None
dtype = torch.float32
if repo_id_in is None:
    # Fresh initialization
    suite = SCAESuite(
        submodule_configs=submodule_configs,
        dtype=dtype,
        device=device,
    )
    # Store config for saving
    config_dict = {
        "submodule_configs": {
            name: asdict(cfg) for name, cfg in submodule_configs.items()
        },
        "is_pretrained": False
    }
else:      
    # Load pretrained
    suite = SCAESuite.from_pretrained(
        repo_id=repo_id_in,
        device=device,
        dtype=dtype,
    )
    # TODO make this cleaner
    # Store config for saving
    config_dict = {
        "is_pretrained": True
    }
suite.connections = connections
suite = accelerator.prepare(suite)

# Initialize trainer
seed = 42
use_wandb = False
trainer = TrainerSCAESuite(
    suite=suite,
    config=trainer_cfg,
    seed=seed,
    wandb_name="gpt2_suite_folded_ln" if use_wandb else None,
)

# Add remaining config information
config_dict.update({
    "trainer_config": asdict(trainer_cfg),
    "dtype": str(dtype),
    "buffer_config": {
        "ctx_len": buffer.ctx_len,
        "refresh_batch_size": buffer.refresh_batch_size,
        "out_batch_size": buffer.out_batch_size,
    }
})

# Save initial config if requested
save_dir = None
if save_dir is not None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

# Training loop
steps = 100
log_steps = 20
pbar = tqdm(buffer, total=steps) if steps is not None else tqdm(buffer)
for step, (initial_acts, input_acts, target_acts, layernorm_scales) in enumerate(pbar):
    if steps is not None and step >= steps:
        break
    
    # Training step with optional logging
    loss = trainer.update(
        step=step,
        initial_acts=initial_acts,
        input_acts=input_acts,
        target_acts=target_acts,
        layernorm_scales=layernorm_scales,
        log_metrics=(log_steps is not None and step % log_steps == 0)
    )
    print(step)