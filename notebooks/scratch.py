#%%
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('..')

import torch as t
import einops
from typing import Dict, List
import torch.sparse as sparse
from tqdm import tqdm
import gc
import pickle
from pathlib import Path

from trainers.scae import SCAESuite
from buffer import AllActivationBuffer
from utils import load_model_with_folded_ln2, load_iterable_dataset
from find_top_connections import get_importance_scores

DTYPE = t.bfloat16
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
t.manual_seed(42)
t.set_grad_enabled(False)
gc.collect()


#%%
if __name__ == '__main__':
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
        out_batch_size = 2048*16,
        refresh_batch_size = 128,
        device=device,
        dtype=DTYPE,
    )


    importance_scores = get_importance_scores(
        suite, 
        buffer, 
        num_batches=100, 
        top_c=500,
        save_dir=Path(f'importance_scores'),
        )
