#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('..')

import torch as t
import einops
from typing import Dict, List
import torch.sparse as sparse
from tqdm import tqdm
import gc
import torch as t
import torch.sparse as sparse
import einops
from tqdm import tqdm
import pickle
from typing import Dict, List
from pathlib import Path

from trainers.scae import SCAESuite
from buffer import AllActivationBuffer
from utils import load_model_with_folded_ln2, load_iterable_dataset
from find_top_connections import get_top_connections, get_importance_scores

DTYPE = t.bfloat16
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
t.manual_seed(42)
t.set_grad_enabled(False)
gc.collect()


#%%
model = load_model_with_folded_ln2("gpt2", device=device, torch_dtype=DTYPE)
data = load_iterable_dataset('Skylion007/openwebtext')
suite = SCAESuite.from_pretrained(
    'jacobcd52/gpt2_suite_folded_ln',
    device=device,
    dtype=DTYPE,
    )

initial_submodule = model.transformer.h[0]
layernorm_submodules = {}
submodules = {}
for layer in range(model.config.n_layer):
    submodules[f"mlp_{layer}"] = (model.transformer.h[layer].mlp, "in_and_out")
    submodules[f"attn_{layer}"] = (model.transformer.h[layer].attn, "out")

    layernorm_submodules[f"mlp_{layer}"] = model.transformer.h[layer].ln_2

buffer = AllActivationBuffer(
    data=data,
    model=model,
    submodules=submodules,
    initial_submodule=initial_submodule,
    layernorm_submodules=layernorm_submodules,
    d_submodule=model.config.n_embd,
    n_ctxs=128,
    out_batch_size = 128,
    refresh_batch_size = 128,
    device=device,
    dtype=DTYPE,
)

#%%
# Compute importance scores
for down_layer in [11]:
    importance_scores = get_importance_scores(
        suite, 
        buffer, 
        f'mlp_{down_layer}',
        num_batches=10_000, 
        save_path=Path(f'importance_scores_{down_layer}.pkl'),
        checkpoint_every=10,
        )
# %%
x = importance_scores['attn_5'].indices()[1]
# %%
import gc
gc.collect()