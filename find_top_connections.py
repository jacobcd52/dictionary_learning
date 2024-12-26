#%%
import torch as t
import einops
from typing import Dict, List
import torch.sparse as sparse
from tqdm import tqdm

from trainers.scae import SCAESuite
from buffer import AllActivationBuffer
from utils import load_model_with_folded_ln2, load_iterable_dataset

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
DTYPE = t.bfloat16
t.manual_seed(42)

t.set_grad_enabled(False)

#%%

import torch as t
import torch.sparse as sparse
import einops
from tqdm import tqdm
import pickle
from typing import Dict, List
from pathlib import Path

def get_importance_scores(
    suite: SCAESuite,
    buffer: AllActivationBuffer,
    num_batches: int = 1,
    save_path: Path = None
) -> Dict[str, Dict[str, sparse.FloatTensor]]:
    """
    Compute importance scores between causally valid pairs of autoencoders and optionally save to file.
    Memory-optimized version with periodic saving and explicit cleanup.
    
    Args:
        suite: The SCAESuite instance
        buffer: Iterator yielding batches of data
        num_batches: Number of batches to process
        save_path: Optional path to save importance scores as pickle file
        
    Returns:
        Dictionary mapping downstream module names to dictionaries of importance scores.
        Format: {down_name: {up_name: importance_tensor}}
    """
    # Sort module names
    mlp_names = sorted([name for name in suite.submodule_names if name.startswith('mlp_')])
    attn_names = sorted([name for name in suite.submodule_names if name.startswith('attn_')])
    
    # Initialize importance scores dictionary
    importance_scores: Dict[str, Dict[str, sparse.FloatTensor]] = {}
    
    # Process batches without gradient tracking
    with t.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            # Clear CUDA cache at start of each batch
            if t.cuda.is_available():
                t.cuda.empty_cache()
            
            # Get batch inputs and run forward pass
            inputs = next(buffer).src
            _, features, topk_info = suite.vanilla_forward(inputs, return_features=True, return_topk=True)
            
            # Process each downstream MLP module
            for down_idx, down_name in enumerate(mlp_names):
                if batch_idx == 0:
                    importance_scores[down_name] = {}
                    
                down_config = suite.configs[down_name]
                scores_shape = (down_config.dict_size, down_config.dict_size)
                
                # Get downstream active features
                down_idxs = topk_info[down_name][0]  # [batch, k_down]
                k_down = down_idxs.shape[1]
                
                # Get valid upstream modules based on causal ordering
                valid_up_names = (
                    [name for name in mlp_names if int(name.split('_')[1]) < down_idx] +
                    [name for name in attn_names if int(name.split('_')[1]) <= down_idx]
                )
                
                # Process each valid upstream module
                for up_name in valid_up_names:
                    up_config = suite.configs[up_name]
                    
                    # Get active upstream features
                    up_idxs = topk_info[up_name][0]  # [batch, k_up]
                    up_vals = topk_info[up_name][1]  # [batch, k_up]
                    k_up = up_idxs.shape[1]
                    
                    # Get active weight matrices and compute virtual weights
                    up_decoder = suite.aes[up_name].decoder.weight  # [d_in, d_up]
                    down_encoder = suite.aes[down_name].encoder.weight  # [d_down, d_in]
                    
                    batch_size = next(iter(inputs.values())).size(0)
                    active_up = up_decoder[:, up_idxs.reshape(-1)].T.view(batch_size, k_up, -1)
                    active_down = down_encoder[down_idxs.reshape(-1)].view(batch_size, k_down, -1)
                    
                    # Compute virtual weights and contributions
                    virtual_weights = einops.einsum(
                        active_down, active_up,
                        "batch k_down d_in, batch k_up d_in -> batch k_down k_up"
                    )
                    contributions = virtual_weights * up_vals.unsqueeze(1).expand(-1, k_down, -1)
                    del virtual_weights  # Free memory
                    
                    # Create indices for sparse tensor
                    down_expanded = down_idxs.unsqueeze(-1).expand(-1, -1, k_up)
                    up_expanded = up_idxs.unsqueeze(1).expand(-1, k_down, -1)
                    indices = t.stack([down_expanded.reshape(-1), up_expanded.reshape(-1)])
                    
                    # Create and accumulate sparse tensor of importance scores
                    batch_scores = sparse.FloatTensor(
                        indices, 
                        contributions.abs().reshape(-1),
                        scores_shape
                    ).coalesce()
                    
                    # Clean up intermediate tensors
                    del indices, down_expanded, up_expanded, contributions
                    del active_up, active_down
                    
                    if batch_idx == 0:
                        importance_scores[down_name][up_name] = batch_scores
                    else:
                        # Update existing tensor in-place when possible
                        importance_scores[down_name][up_name].add_(batch_scores)
                        importance_scores[down_name][up_name] = importance_scores[down_name][up_name].coalesce()
                    
            # Periodic saving every 10 batches
            if save_path and batch_idx > 0 and batch_idx % 10 == 0:
                temp_save_path = save_path.with_suffix('.temp.pkl')
                with open(temp_save_path, 'wb') as f:
                    pickle.dump(importance_scores, f)
    
    # Save final importance scores if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(importance_scores, f)
    
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

#%%

# if __name__ == "__main__":
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
    out_batch_size = 256,
    refresh_batch_size = 256,
    device=device,
    dtype=DTYPE,
)

#%%

# Compute importance scores
importance_scores = get_importance_scores(suite, buffer, num_batches=100, save_path=Path('importance_scores.pkl'))


#%%
# Later, load importance scores and compute connections
with open('importance_scores.pkl', 'rb') as f:
    importance_scores = pickle.load(f)

connections = get_top_connections(
    suite=suite,
    importance_scores=importance_scores,
    c=10,
    chunk_size=1000
)
# %%
