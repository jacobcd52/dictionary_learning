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


import gc
import torch as t
import einops
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import pickle
from torch import sparse

import gc
import torch as t
import einops
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import pickle
from torch import sparse

import gc
import torch as t
import einops
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import pickle
from torch import sparse
import sys

def get_valid_pairs(suite: SCAESuite) -> List[Tuple[str, str]]:
    """Get all valid (upstream, downstream) pairs to process."""
    mlp_names = sorted([name for name in suite.submodule_names if name.startswith('mlp_')])
    attn_names = sorted([name for name in suite.submodule_names if name.startswith('attn_')])
    
    pairs = []
    for down_name in mlp_names:
        down_layer = int(down_name.split('_')[1])
        valid_up_names = (
            [name for name in mlp_names if int(name.split('_')[1]) < down_layer] +
            [name for name in attn_names if int(name.split('_')[1]) <= down_layer]
        )
        pairs.extend([(up, down_name) for up in valid_up_names])
    
    return pairs

def process_pair(
    suite: SCAESuite,
    buffer: AllActivationBuffer,
    up_name: str,
    down_name: str,
    num_batches: int,
    top_c: int,
    save_dir: Path = None
) -> sparse.FloatTensor:
    """Process a single (upstream, downstream) pair."""
    
    down_config = suite.configs[down_name]
    scores_shape = (down_config.dict_size, down_config.dict_size)
    
    # Initialize running top-k values and indices for each downstream feature
    top_values = t.zeros((scores_shape[0], top_c), device='cuda') - float('inf')
    top_indices = t.zeros((scores_shape[0], top_c), dtype=t.long, device='cuda')
    
    with t.no_grad():
        for batch_idx in range(num_batches):
            # Get activations
            inputs = next(buffer).src
            _, _, topk_info = suite.vanilla_forward(inputs, return_features=True, return_topk=True)
            
            down_idxs = topk_info[down_name][0]  # [batch, k_down]
            batch_size, k_down = down_idxs.shape
            
            up_idxs = topk_info[up_name][0]  # [batch, k_up]
            up_vals = topk_info[up_name][1]  # [batch, k_up]
            _, k_up = up_idxs.shape
            
            # Compute virtual weights
            up_decoder = suite.aes[up_name].decoder.weight
            down_encoder = suite.aes[down_name].encoder.weight
            
            active_up = up_decoder[:, up_idxs.reshape(-1)].T.view(batch_size, k_up, -1)
            active_down = down_encoder[down_idxs.reshape(-1)].view(batch_size, k_down, -1)
            del up_decoder, down_encoder
            
            virtual_weights = einops.einsum(
                active_down, active_up,
                "batch k_down d_in, batch k_up d_in -> batch k_down k_up"
            )
            del active_up, active_down
            
            # Compute contributions
            up_vals_expanded = up_vals.unsqueeze(1)  # [batch, 1, k_up]
            contributions = virtual_weights * up_vals_expanded  # [batch, k_down, k_up]
            del virtual_weights, up_vals_expanded
            
            # Reshape contributions to [num_down_feats, num_values]
            # First, create a tensor that will hold all values for each downstream feature
            # Size is [dict_size, batch * k_down * k_up]
            num_values_per_feature = batch_size * k_down * k_up
            all_values = t.zeros((scores_shape[0], num_values_per_feature), device='cuda')
            all_indices = t.zeros((scores_shape[0], num_values_per_feature), dtype=t.long, device='cuda')
            
            # Create index tensors for scattering
            # For each downstream feature, we want its values in the corresponding row
            flat_down_idxs = down_idxs.reshape(-1)  # [batch * k_down]
            flat_contributions = contributions.reshape(-1, k_up)  # [batch * k_down, k_up]
            flat_up_idxs = up_idxs.repeat_interleave(k_down, dim=0)  # [batch * k_down, k_up]
            
            # Create scatter indices
            row_idx = flat_down_idxs.repeat_interleave(k_up)  # Repeat each downstream idx k_up times
            col_idx = t.arange(num_values_per_feature, device='cuda')
            
            # Scatter the values and indices
            all_values.scatter_(0, row_idx.unsqueeze(0), flat_contributions.reshape(1, -1))
            all_indices.scatter_(0, row_idx.unsqueeze(0), flat_up_idxs.reshape(1, -1))
            
            # Combine with existing top values
            combined_values = t.cat([top_values, all_values], dim=1)
            combined_indices = t.cat([top_indices, all_indices], dim=1)
            
            # Get top-k for all features at once
            new_top_values, top_idx = combined_values.topk(top_c, dim=1)
            
            # Gather corresponding indices
            new_top_indices = t.gather(combined_indices, 1, top_idx)
            
            # Update top values and indices
            top_values = new_top_values
            top_indices = new_top_indices
            
            del contributions, topk_info
            gc.collect()
            if t.cuda.is_available():
                t.cuda.empty_cache()
            
            # Update progress inline
            sys.stdout.write(f"\rProcessing {up_name} -> {down_name}: {batch_idx+1}/{num_batches} batches")
            sys.stdout.flush()
    
    # Create final sparse tensor from top-k values
    valid_mask = top_values > -float('inf')
    downstream_idx = t.arange(scores_shape[0], device='cuda').unsqueeze(1).expand_as(valid_mask)[valid_mask]
    upstream_idx = top_indices[valid_mask]
    values = top_values[valid_mask]
    
    importance_scores = t.sparse_coo_tensor(
        indices=t.stack([downstream_idx, upstream_idx]),
        values=values,
        size=scores_shape,
        device='cuda'
    ).coalesce()
    
    # Save if directory provided
    if save_dir is not None:
        save_path = save_dir / f"importance_{up_name}_to_{down_name}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(importance_scores.cpu(), f)
    
    return importance_scores

def get_importance_scores(
    suite: SCAESuite,
    buffer: AllActivationBuffer,
    num_batches: int = 1,
    top_c: int = 1000,
    save_dir: Path = None,
) -> Dict[Tuple[str, str], sparse.FloatTensor]:
    """
    Compute importance scores for all valid pairs, processing one pair at a time
    to minimize memory usage.
    """
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True)
    
    pairs = get_valid_pairs(suite)
    print(f"Processing {len(pairs)} (upstream, downstream) pairs...")
    
    importance_scores = {}
    for up_name, down_name in pairs:
        scores = process_pair(
            suite=suite,
            buffer=buffer,
            up_name=up_name,
            down_name=down_name,
            num_batches=num_batches,
            top_c=top_c,
            save_dir=save_dir
        )
        importance_scores[(up_name, down_name)] = scores
        print(f"\nCompleted {up_name} -> {down_name}")
    
    return importance_scores