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

def get_importance_scores(
    suite: SCAESuite,
    buffer: AllActivationBuffer,
    down_name: str,
    num_batches: int = 1,
    save_path: Path = None,
    top_c: int = 1000,
    checkpoint_every: int = 50
) -> Dict[str, sparse.FloatTensor]:
    """
    Compute importance scores using efficient sparse tensor accumulation.
    """
    mlp_names = sorted([name for name in suite.submodule_names if name.startswith('mlp_')])
    attn_names = sorted([name for name in suite.submodule_names if name.startswith('attn_')])
    
    if down_name not in mlp_names:
        raise ValueError(f"Downstream component {down_name} not found in MLPs")
    down_layer = int(down_name.split('_')[1])
    
    valid_up_names = (
        [name for name in mlp_names if int(name.split('_')[1]) < down_layer] +
        [name for name in attn_names if int(name.split('_')[1]) <= down_layer]
    )

    print(f"Valid upstream modules: {valid_up_names}")
    
    down_config = suite.configs[down_name]
    scores_shape = (down_config.dict_size, down_config.dict_size)
    
    if save_path is not None:
        checkpoint_dir = save_path.parent / f"{save_path.stem}_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
    
    running_totals = {
        up_name: t.sparse_coo_tensor(size=scores_shape, device='cuda')
        for up_name in valid_up_names
    }
    
    with t.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            inputs = next(buffer).src
            _, features, topk_info = suite.vanilla_forward(inputs, return_features=True, return_topk=True)
            del features
            
            down_idxs = topk_info[down_name][0]
            batch_size, k_down = down_idxs.shape
            
            for up_name in valid_up_names:
                up_idxs = topk_info[up_name][0]
                up_vals = topk_info[up_name][1]
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
                up_vals_expanded = up_vals.unsqueeze(1)
                contributions = virtual_weights * up_vals_expanded
                del virtual_weights, up_vals_expanded
                
                # Create sparse tensor
                down_expanded = down_idxs.unsqueeze(2).expand(-1, -1, k_up)
                up_expanded = up_idxs.unsqueeze(1).expand(-1, k_down, -1)
                
                indices = t.stack([
                    down_expanded.reshape(-1),
                    up_expanded.reshape(-1)
                ])
                del down_expanded, up_expanded
                
                values = contributions.reshape(-1)
                del contributions
                
                batch_totals = t.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=scores_shape,
                    device='cuda'
                )
                del indices, values
                
                # Update running totals without coalescing every time
                running_totals[up_name] = running_totals[up_name] + batch_totals
                del batch_totals
            
            del topk_info
            
            if save_path is not None and (batch_idx + 1) % checkpoint_every == 0:
                checkpoint_number = batch_idx // checkpoint_every
                
                for up_name in valid_up_names:
                    # Coalesce before saving
                    running_totals[up_name] = running_totals[up_name].coalesce()
                    
                    up_checkpoint_path = checkpoint_dir / f"connections_{checkpoint_number}_{up_name}.pkl"
                    total_data = running_totals[up_name].cpu()
                    with open(up_checkpoint_path, 'wb') as f:
                        pickle.dump(total_data, f)
                    del total_data
                    
                    running_totals[up_name] = t.sparse_coo_tensor(size=scores_shape, device='cuda')
                
                # Cleanup after all checkpoints are saved
                gc.collect()
                if t.cuda.is_available():
                    t.cuda.empty_cache()
                
                print(f"Saved checkpoint {checkpoint_number}")
    
    # Compute final importance scores
    importance_scores = {}
    for up_name in valid_up_names:
        totals = running_totals[up_name].coalesce()
        means = totals.values()
        indices = totals.indices()
        
        means_dense = t.zeros(scores_shape, device='cuda')
        means_dense[indices[0], indices[1]] = means
        del means, indices
        
        top_values, top_indices = means_dense.topk(min(top_c, means_dense.size(1)), dim=1)
        del means_dense
        
        valid_mask = top_values > 0
        downstream_indices = t.arange(scores_shape[0], device='cuda').unsqueeze(1).expand(-1, top_c)
        downstream_indices = downstream_indices[valid_mask]
        upstream_indices = top_indices[valid_mask]
        final_values = top_values[valid_mask]
        del top_values, top_indices, valid_mask
        
        importance_scores[up_name] = t.sparse_coo_tensor(
            indices=t.stack([downstream_indices, upstream_indices]),
            values=final_values,
            size=scores_shape,
            device='cuda'
        ).coalesce()
        del downstream_indices, upstream_indices, final_values
    
    if save_path:
        cpu_scores = {name: tensor.cpu() for name, tensor in importance_scores.items()}
        with open(save_path, 'wb') as f:
            pickle.dump(cpu_scores, f)
        del cpu_scores
    
    return importance_scores


def get_top_connections(
    suite: SCAESuite,
    importance_scores: Dict[str, Dict[str, sparse.FloatTensor]],
    c: int,
    chunk_size: int = 1000
) -> Dict[str, Dict[str, t.Tensor]]:
    """
    Convert importance scores to top connections for each downstream feature.
    
    Args:
        suite: The SCAESuite instance
        importance_scores: Dictionary of importance scores from get_importance_scores
        c: Number of connections to return per downstream feature
        chunk_size: Size of chunks for processing (adjust based on available memory)
        
    Returns:
        Dictionary mapping downstream module names to dictionaries of connections.
        Format: {down_name: {up_name: connection_tensor}}
        where connection_tensor has shape [n_down_features, c] containing indices 
        of connected upstream features
    """
    mlp_names = sorted([name for name in suite.submodule_names if name.startswith('mlp_')])
    attn_names = sorted([name for name in suite.submodule_names if name.startswith('attn_')])
    connections = {}
    
    # Get list of modules to process
    downstream_modules = list(importance_scores.keys())
    
    # Process one downstream module at a time
    for down_name in downstream_modules:
        up_dict = importance_scores[down_name]
        print(f"Processing downstream module: {down_name}")
        down_size = suite.configs[down_name].dict_size
        connections[down_name] = {}
        
        try:
            # Process in chunks
            for chunk_start in range(0, down_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, down_size)
                chunk_size_actual = chunk_end - chunk_start
                
                # Get valid upstream modules based on causal ordering
                down_idx = int(down_name.split('_')[1])
                valid_up_names = (
                    [name for name in mlp_names if int(name.split('_')[1]) < down_idx] +
                    [name for name in attn_names if int(name.split('_')[1]) <= down_idx]
                )
                
                # Stack scores from each valid upstream module
                chunk_scores = []
                up_names = []
                
                for up_name in valid_up_names:
                    if up_name not in up_dict:
                        continue
                    scores = up_dict[up_name]
                    dense_chunk = scores.to_dense()[chunk_start:chunk_end]
                    if dense_chunk.sum() > 0:  # Only include if there are non-zero scores
                        chunk_scores.append(dense_chunk)
                        up_names.append(up_name)
                
                if not chunk_scores:  # Skip if no valid connections in chunk
                    continue
                    
                # Find top connections
                stacked_scores = t.stack(chunk_scores, dim=1)
                flat_scores = stacked_scores.reshape(chunk_size_actual, -1)
                top_values, top_indices = t.topk(flat_scores, min(c, flat_scores.size(1)))
                
                # Convert indices back to module and feature indices
                max_up_size = up_dict[up_names[0]].size(1)
                module_idx = top_indices // max_up_size
                feature_idx = top_indices % max_up_size
                
                # Create connections for each upstream module
                for i, up_name in enumerate(up_names):
                    module_mask = (module_idx == i) & (top_values > 0)
                    if not module_mask.any():
                        continue
                        
                    rows, cols = t.where(module_mask)
                    valid_indices = feature_idx[rows, cols]
                    
                    if up_name not in connections[down_name]:
                        connections[down_name][up_name] = t.full(
                            (down_size, c), 
                            -1, 
                            device=suite.device
                        )
                    
                    connections[down_name][up_name][
                        chunk_start + rows, cols
                    ] = valid_indices
                
                # Clean up
                del stacked_scores, flat_scores, top_values, top_indices
                t.cuda.empty_cache()
                
        finally:
            # Clean up after processing each downstream module
            t.cuda.empty_cache()
    
    return connections