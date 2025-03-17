#%%
from typing import Dict
import torch as t
from einops import einsum
from trainers.scae import SCAESuite
from buffer import SimpleBuffer
from tqdm import tqdm
import einops
from huggingface_hub import login
import pickle
from transformer_lens import HookedTransformer
import psutil

from buffer import SimpleBuffer
from training import train_scae_suite
from utils import load_model_with_folded_ln2, load_iterable_dataset
from trainers.scae import SCAESuite


def get_avg_contribs(
        suite: SCAESuite,
        buffer: SimpleBuffer,
        n_batches: int = 10,
    ) -> Dict[str, Dict[str, t.Tensor]]:
    """Calculate average contributions between autoencoder features using sparse representation.
    Memory-efficient implementation using SCAESuite's built-in avg_f_by_f functionality.
    """
    from tqdm.auto import tqdm
    
    # Assert that connections must be None
    assert suite.connections is None, "connections must be None for this implementation"
    
    with t.no_grad():
        # Initialize the results dictionary
        avg_contribs = {
            down_name: {} for down_name in suite.submodule_names
        }
        
        # Process batches with tqdm
        for batch_idx in tqdm(range(n_batches)):
            cache, tokens = next(buffer)
            
            # Get all pruned features at once using forward_pruned
            _, pruned_features = suite.forward_pruned(cache, return_features=True)
            
            # For each downstream autoencoder
            for down_name in suite.submodule_names:
                down_type, down_layer = down_name.split('_')
                down_layer = int(down_layer)
                
                # For each potential upstream autoencoder
                for up_name in suite.submodule_names:
                    up_type, up_layer = up_name.split('_')
                    up_layer = int(up_layer)
                    
                    # Skip if not a valid upstream connection
                    if up_layer > down_layer or (up_layer == down_layer and (up_type == 'mlp' or up_name == down_name)):
                        continue
                    
                    # Get upstream features from the pruned forward pass
                    up_feats = pruned_features[up_name]
                    
                    # Use the built-in method with avg_f_by_f=True to get the feature-by-feature contribution matrix
                    if down_type == 'mlp':
                        contribution_tensor = suite.get_pruned_contribs_mlp(
                            cache, up_name, down_name, up_feats, avg_f_by_f=True
                        )
                    elif down_type == 'attn':
                        contribution_tensor = suite.get_pruned_contribs_attn(
                            cache, up_name, down_name, up_feats, avg_f_by_f=True
                        )
                    else:
                        raise ValueError(f"Invalid down_type: {down_type}")
                    
                    # Initialize or update running average
                    if up_name not in avg_contribs[down_name]:
                        avg_contribs[down_name][up_name] = contribution_tensor
                    else:
                        avg_contribs[down_name][up_name] = (
                            avg_contribs[down_name][up_name] * batch_idx + contribution_tensor
                        ) / (batch_idx + 1)
            
            # Clean up memory
            t.cuda.empty_cache()
        
        return avg_contribs

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


def get_top_connections(avg_contribs, c):
    """
    Get the top c connections for each downstream module.
    Args:
        avg_contribs: Dictionary mapping downstream module names to dictionaries mapping upstream module names
                      to tensors of shape [f_up, f_down]
        c: Number of top connections to keep
    Returns:
        Dictionary mapping downstream module names to dictionaries mapping upstream module names
        to tensors of shape [f_up, c] containing the indices of the top connections
    """
    top_connections = {}
    for down_name, contrib_dict in tqdm(avg_contribs.items()):
        if len(contrib_dict) == 0:
            top_connections[down_name] = {}
        else:
            top_connections[down_name] = get_top_c_indices(contrib_dict, c)
    return top_connections


def generate_fake_connections(
    connections: Dict[str, Dict[str, t.Tensor]], 
    num_features: int
) -> Dict[str, Dict[str, t.Tensor]]:
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
            all_perms = t.stack([
                t.randperm(num_features, device=tensor.device) 
                for _ in range(N)
            ])
            
            # Create output tensor filled with padding
            fake_tensor = t.full_like(tensor, -1)
            
            # For each position that needs a value (each True in mask),
            # we'll take the next unused value from all_perms
            counts = t.zeros(N, dtype=t.long, device=tensor.device)
            
            # Create index tensor for gathering values
            idx = t.arange(c, device=tensor.device).unsqueeze(0).expand(N, -1)  # [N, c]
            
            # Use cumsum on mask to get indices into the permutation array
            # This ensures we use consecutive values from all_perms for each row
            indices = (mask.cumsum(dim=1) - 1)
            
            # Only gather values where mask is True
            fake_tensor[mask] = all_perms[
                t.arange(N, device=tensor.device).unsqueeze(1).expand(-1, c)[mask],
                indices[mask]
            ]
            
            fake_connections[outer_key][inner_key] = fake_tensor
    
    return fake_connections

if __name__ == "__main__":
    #%%
    login("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")
    device = "cuda:0" if t.cuda.is_available() else "cpu"

    t.set_grad_enabled(True)

    #%%
    DTYPE = t.bfloat16
    MODEL_NAME = "pythia-70m"
    batch_size = 1024
    expansion = 4
    ctx_len = 128


    #%%
    data = load_iterable_dataset('monology/pile-uncopyrighted')

    buffer = SimpleBuffer(
        data=data,
        model_name=MODEL_NAME,
        ctx_len=ctx_len,
        device="cuda",
        batch_size=batch_size,
        dtype=DTYPE,
    ) 

    #%%
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=DTYPE)
    suite = SCAESuite.from_pretrained(
        "jacobcd52/pythia-70m_scae_all__mse",
        model,
        device=device,
        dtype=DTYPE,
    )

    #%%
    avg_contribs = get_avg_contribs(suite, buffer, n_batches=100)

    #%%
    from tqdm import tqdm
    import os
    for c in tqdm([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 300]):
        top_connections = get_top_connections(avg_contribs, c=c)
        # save as pickle file - create the file if it does not exist
        file_path = f"pythia_connections/top_connections_{c}.pkl"
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(top_connections, f)