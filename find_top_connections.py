import torch as t
import einops
from typing import Dict, List, Tuple
import torch.sparse as sparse
from tqdm import tqdm
import gc
import pickle
from pathlib import Path
import sys 

from trainers.scae import SCAESuite
from buffer import AllActivationBuffer


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
            
            # 1. First organize all our values and indices
            flat_down_idxs = down_idxs.reshape(-1)  # [batch * k_down]
            flat_up_idxs = up_idxs.repeat_interleave(k_down, dim=0)  # [batch * k_down, k_up]
            flat_contributions = contributions.reshape(-1, k_up)  # [batch * k_down, k_up]
            
            # 2. Create indices for the full matrix of current batch's values
            row_indices = flat_down_idxs.repeat_interleave(k_up)  # Each downstream idx repeated k_up times
            col_indices = t.arange(flat_up_idxs.shape[0] * k_up, device='cuda') % k_up  # [0,1,...,k_up-1] repeated
            values = flat_contributions.reshape(-1)  # All contribution values
            
            # 3. Create sparse tensor of new values 
            new_values = t.sparse_coo_tensor(
                indices=t.stack([row_indices, col_indices]),
                values=values,
                size=(scores_shape[0], k_up),  # We only need k_up columns for this batch
                device='cuda'
            ).coalesce()
            
            # 4. Convert to dense only for active features
            active_rows = t.unique(row_indices)
            
            # 5. Create a dense matrix for active features that combines old and new values
            combined_values = t.zeros((len(active_rows), top_c + k_up), device='cuda')
            combined_indices = t.zeros((len(active_rows), top_c + k_up), dtype=t.long, device='cuda')
            
            # 6. Fill in existing top values and indices for active features
            combined_values[:, :top_c] = top_values[active_rows]
            combined_indices[:, :top_c] = top_indices[active_rows]
            
            # 7. Add new values from sparse tensor
            dense_new = new_values.index_select(0, active_rows).to_dense()
            combined_values[:, top_c:] = dense_new
            
            # 8. Create corresponding indices for new values
            # Expand flat_up_idxs to match the shape we need
            expanded_up_idxs = flat_up_idxs.reshape(-1, k_up)  # [batch * k_down, k_up]
            active_up_idxs = expanded_up_idxs[active_rows]
            combined_indices[:, top_c:] = active_up_idxs
            
            # 9. Get top-k for all features at once
            top_k_values, top_k_indices = combined_values.topk(top_c, dim=1)
            
            # 10. Gather corresponding feature indices
            new_top_indices = t.gather(combined_indices, 1, top_k_indices)
            
            # 11. Update the top values and indices for active features
            top_values[active_rows] = top_k_values
            top_indices[active_rows] = new_top_indices
            
            # Clean up large intermediate tensors
            del combined_values, combined_indices, dense_new
            gc.collect()
            if t.cuda.is_available():
                t.cuda.empty_cache()
            
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


from tqdm import tqdm
import psutil
import sys

def get_top_c_indices(top_connections_dict: Dict[str, t.Tensor], c: int, chunk_size: int = 100, 
                      memory_threshold_gb: float = 32) -> Dict[str, t.Tensor]:
    """
    Args:
        top_connections_dict: Dictionary mapping strings to sparse COO tensors, each of shape [M, N]
        c: Number of top indices to return per row
        chunk_size: Number of rows to process at once to manage memory
        memory_threshold_gb: Maximum allowed CPU memory usage in gigabytes
        
    Returns:
        Dictionary mapping strings to tensors of shape [M, c] containing indices that correspond
        to values that rank in the top c by magnitude across all dictionary entries combined
        
    Raises:
        MemoryError: If CPU memory usage exceeds memory_threshold_gb
    """
    def get_memory_usage_gb():
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def check_memory_usage():
        """Check if memory usage exceeds threshold"""
        current_usage = get_memory_usage_gb()
        if current_usage > memory_threshold_gb:
            raise MemoryError(f"Memory usage ({current_usage:.2f}GB) exceeded threshold ({memory_threshold_gb}GB)")
    
    # Initial memory check
    check_memory_usage()
    
    # Convert all tensors to dense and get shapes
    print("Converting sparse tensors to dense...")
    dense_dict = {key: tensor.to_dense() for key, tensor in top_connections_dict.items()}
    check_memory_usage()
    
    M, N = next(iter(dense_dict.values())).shape
    device = next(iter(dense_dict.values())).device
    num_dicts = len(dense_dict)
    dict_keys = list(dense_dict.keys())
    
    print(f"Processing {M} rows in chunks of {chunk_size}")
    print(f"Current memory usage: {get_memory_usage_gb():.2f}GB")
    
    # Initialize result dictionary with -1s on CPU
    result_dict = {key: t.full((M, c), -1, dtype=t.long) for key in dict_keys}
    check_memory_usage()
    
    # Process chunks
    chunk_pbar = tqdm(range(0, M, chunk_size), desc="Processing chunks")
    for start_idx in chunk_pbar:
        end_idx = min(start_idx + chunk_size, M)
        chunk_pbar.set_postfix({'mem_usage': f'{get_memory_usage_gb():.2f}GB'})
        
        # Stack chunk of all tensors
        chunk_values = t.stack([dense[start_idx:end_idx] for dense in dense_dict.values()], dim=1).cuda()
        chunk_size_actual = end_idx - start_idx
        
        # Get absolute values
        abs_values = chunk_values.abs()
        
        # Create indices tensors
        batch_idx = t.arange(chunk_size_actual, device='cuda')[:, None, None].expand(-1, num_dicts, N)
        dict_idx = t.arange(num_dicts, device='cuda')[None, :, None].expand(chunk_size_actual, -1, N)
        col_idx = t.arange(N, device='cuda')[None, None, :].expand(chunk_size_actual, num_dicts, -1)
        
        # Mask for nonzero values
        nonzero_mask = chunk_values != 0
        
        # Get values and indices where values are nonzero
        values_flat = abs_values[nonzero_mask]
        batch_flat = batch_idx[nonzero_mask]
        dict_flat = dict_idx[nonzero_mask]
        col_flat = col_idx[nonzero_mask]
        
        # Group by batch within chunk
        batch_sizes = nonzero_mask.sum(dim=(1,2))
        batch_groups = t.split(t.arange(values_flat.size(0), device='cuda'), batch_sizes.tolist())
        
        # Sort values within each batch group and get top c
        batch_pbar = tqdm(enumerate(batch_groups), 
                         total=len(batch_groups), 
                         desc="Processing batches",
                         leave=False)
        
        for b, group in batch_pbar:
            if len(group) > 0:
                # Sort this batch's values
                sorted_vals, sort_idx = values_flat[group].sort(descending=True)
                top_c_idx = group[sort_idx[:c]]
                
                # Get corresponding dictionary indices and column indices
                top_dict_indices = dict_flat[top_c_idx]
                top_col_indices = col_flat[top_c_idx]
                
                # For each dictionary
                for d, key in enumerate(dict_keys):
                    # Get indices where this dictionary appears
                    dict_mask = top_dict_indices == d
                    if dict_mask.any():
                        # Get columns for this dictionary and place them in result
                        dict_cols = top_col_indices[dict_mask]
                        num_cols = dict_cols.size(0)
                        result_dict[key][start_idx + b, :num_cols] = dict_cols
            
            check_memory_usage()
        
        # Clear GPU memory
        del chunk_values, abs_values, batch_idx, dict_idx, col_idx
        del values_flat, batch_flat, dict_flat, col_flat
        t.cuda.empty_cache()
    
    return {k : v.cuda() for k, v in result_dict.items()}


import torch
from collections import defaultdict

def generate_fake_connections(
    connections: Dict[str, Dict[str, torch.Tensor]], 
    num_features: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate fake connections with the same structure as the input connections.
    Uses parallelized tensor operations for efficiency.
    
    For each tensor in the original connections:
    - Preserves -1 values (padding)
    - Replaces other values with random integers in range(num_features)
    - Ensures random values are distinct across each N-slice
    
    Args:
        connections: Nested dict of tensors, each with shape [N, c]
        num_features: Upper bound (exclusive) for random integers
        
    Returns:
        Dict with same structure as input but with randomized values
    """
    fake_connections = {}
    
    for outer_key, inner_dict in connections.items():
        fake_connections[outer_key] = {}
        
        for inner_key, tensor in inner_dict.items():
            N, c = tensor.shape
            mask = tensor != -1  # [N, c]
            
            # Check if we can generate enough distinct values
            max_needed = mask.sum(dim=1).max().item()
            if max_needed > num_features:
                raise ValueError(
                    f"Cannot generate {max_needed} distinct values "
                    f"from range(0, {num_features})"
                )
            
            # Create independent random permutations for each row
            # Shape: [N, num_features]
            all_perms = torch.stack([
                torch.randperm(num_features, device=tensor.device) 
                for _ in range(N)
            ])
            
            # Create output tensor filled with padding
            fake_tensor = torch.full_like(tensor, -1)
            
            # For each position that needs a value (each True in mask),
            # we'll take the next unused value from all_perms
            counts = torch.zeros(N, dtype=torch.long, device=tensor.device)
            
            # Create index tensor for gathering values
            idx = torch.arange(c, device=tensor.device).unsqueeze(0).expand(N, -1)  # [N, c]
            
            # Use cumsum on mask to get indices into the permutation array
            # This ensures we use consecutive values from all_perms for each row
            indices = (mask.cumsum(dim=1) - 1)
            
            # Only gather values where mask is True
            fake_tensor[mask] = all_perms[
                torch.arange(N, device=tensor.device).unsqueeze(1).expand(-1, c)[mask],
                indices[mask]
            ]
            
            fake_connections[outer_key][inner_key] = fake_tensor
    
    return fake_connections