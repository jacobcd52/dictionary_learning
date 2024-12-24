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

model = load_model_with_folded_ln2("gpt2", device=device, torch_dtype=DTYPE)
data = load_iterable_dataset('Skylion007/openwebtext')

#%%

import torch as t
import einops
from typing import Dict, List
import torch.sparse as sparse

def get_top_connections(
    suite: SCAESuite, 
    c: int,
    buffer: AllActivationBuffer,
    num_batches: int = 1
) -> Dict[str, Dict[str, t.Tensor]]:
    """
    Compute importance scores between all valid pairs of autoencoders.
    
    Args:
        suite: The SCAESuite instance
        c: Number of connections to return per downstream feature
        buffer: Iterator yielding batches of data
        num_batches: Number of batches to process
        
    Returns:
        Dictionary mapping downstream module names to dictionaries of importance scores.
        Format: {down_name: {up_name: importance_tensor}}
        where importance_tensor has shape [n_down_features, c] containing indices 
        of connected upstream features, padded with -1
    """
    # Get all module names and configs
    mlp_names = [name for name in suite.submodule_names if name.startswith('mlp_')]
    attn_names = [name for name in suite.submodule_names if name.startswith('attn_')]
    
    # Sort names to ensure correct ordering
    mlp_names.sort()
    attn_names.sort()
    
    # Initialize dictionary to store importance scores
    importance_scores = {}
    
    # Process multiple batches
    for batch_idx in tqdm(range(num_batches)):
        # Get inputs from buffer and determine batch size
        inputs = next(buffer).src
        batch_size = next(iter(inputs.values())).size(0)
        
        # Run vanilla forward pass to get features and top-k info
        _, features, topk_info = suite.vanilla_forward(inputs, return_features=True, return_topk=True)
        
        # For each downstream MLP module
        for down_idx, down_name in enumerate(mlp_names):
            if batch_idx == 0:
                importance_scores[down_name] = {}
            down_config = suite.configs[down_name]
            
            # Initialize sparse tensor for storing scores
            scores_shape = (down_config.dict_size, down_config.dict_size)
            
            # Get downstream active features and indices
            down_idxs = topk_info[down_name][0]  # [batch, k_down]
            k_down = down_idxs.shape[1]
            
            # Process all valid upstream modules
            valid_up_names = (
                [name for name in mlp_names if int(name.split('_')[1]) < down_idx] +
                [name for name in attn_names if int(name.split('_')[1]) <= down_idx]
            )
            
            for up_name in valid_up_names:
                up_config = suite.configs[up_name]
                
                # Get active features
                up_idxs = topk_info[up_name][0]  # [batch, k_up]
                up_vals = topk_info[up_name][1]  # [batch, k_up]
                k_up = up_idxs.shape[1]
                
                # Get active weight matrices
                up_decoder = suite.aes[up_name].decoder.weight  # [d_in, d_up]
                down_encoder = suite.aes[down_name].encoder.weight  # [d_down, d_in]
                
                # Get active vectors
                active_up_vectors = up_decoder[:, up_idxs.reshape(-1)].T  # [batch*k_up, d_in]
                active_up_vectors = active_up_vectors.view(batch_size, k_up, -1)  # [batch, k_up, d_in]
                
                active_down_vectors = down_encoder[down_idxs.reshape(-1)]  # [batch*k_down, d_in]
                active_down_vectors = active_down_vectors.view(batch_size, k_down, -1)  # [batch, k_down, d_in]
                
                # Compute virtual weights between active features
                virtual_weights = einops.einsum(
                    active_down_vectors, active_up_vectors,
                    "batch k_down d_in, batch k_up d_in -> batch k_down k_up"
                )
                
                # Get upstream values and compute contributions
                up_vals_expanded = up_vals.unsqueeze(1).expand(-1, k_down, -1)
                contributions = virtual_weights * up_vals_expanded
                
                # Accumulate importance scores
                contributions_abs = contributions.abs()
                
                # Create indices for sparse tensor updates
                down_idxs_expanded = down_idxs.unsqueeze(-1).expand(-1, -1, k_up)  # [batch, k_down, k_up]
                up_idxs_expanded = up_idxs.unsqueeze(1).expand(-1, k_down, -1)  # [batch, k_down, k_up]
                
                # Flatten indices and values
                flat_down_idxs = down_idxs_expanded.reshape(-1)
                flat_up_idxs = up_idxs_expanded.reshape(-1)
                flat_values = contributions_abs.reshape(-1)
                
                # Create sparse tensor of importance scores
                indices = t.stack([flat_down_idxs, flat_up_idxs])
                batch_scores = sparse.FloatTensor(
                    indices, flat_values,
                    scores_shape
                ).coalesce()
                
                if batch_idx == 0:
                    importance_scores[down_name][up_name] = batch_scores
                else:
                    existing_scores = importance_scores[down_name][up_name]
                    importance_scores[down_name][up_name] = (existing_scores + batch_scores).coalesce()
    
    connections = {}
    downstream_modules = list(importance_scores.keys())
    
    # Define chunk size for processing
    chunk_size = 1000  # Adjust based on available memory
    
    # Process one downstream module at a time
    for down_name in downstream_modules:
        up_dict = importance_scores[down_name]
        connections[down_name] = {}
        down_size = suite.configs[down_name].dict_size
        module_names = list(up_dict.keys())
        
        try:
            # Initialize connection tensors for each upstream module
            for up_name in module_names:
                connections[down_name][up_name] = t.full((down_size, c), -1, device=suite.device)
            
            # Process in chunks
            for chunk_start in range(0, down_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, down_size)
                chunk_scores = []
                
                # Get chunk of scores from each upstream module
                for up_name, scores in up_dict.items():
                    # Convert to dense first, then slice the chunk
                    dense_scores = scores.to_dense()
                    dense_chunk = dense_scores[chunk_start:chunk_end]
                    del dense_scores  # Clean up the full dense tensor
                    chunk_scores.append(dense_chunk)
                
                # Stack and process just this chunk
                chunk_all_scores = t.stack(chunk_scores, dim=1)
                flat_chunk_scores = chunk_all_scores.reshape(chunk_end - chunk_start, -1)
                del chunk_all_scores
                
                # Get top c for this chunk
                top_values, top_indices = t.topk(flat_chunk_scores, c)
                del flat_chunk_scores
                
                # Convert flat indices back to module and feature indices
                max_up_size = up_dict[module_names[0]].size(1)
                module_idx = top_indices // max_up_size
                feature_idx = top_indices % max_up_size
                del top_indices
                
                # Update connections for each upstream module
                for i, up_name in enumerate(module_names):
                    module_mask = (module_idx == i) & (top_values > 0)
                    rows, cols = t.where(module_mask)
                    valid_indices = feature_idx[rows, cols]
                    
                    if len(valid_indices) > 0:
                        connections[down_name][up_name][chunk_start + rows, cols] = valid_indices
                
                # Clean up chunk processing
                del module_idx, feature_idx, top_values
                t.cuda.empty_cache()
            
        finally:
            # Clean up importance scores after processing each downstream module
            del importance_scores[down_name]
            t.cuda.empty_cache()
    
    return connections

# %%
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
    refresh_batch_size = 256,
    device=device,
    dtype=DTYPE,
)
# %%
out = get_top_connections(
    suite,
    c=200,
    buffer=buffer,
    num_batches=40,
)
# %%
# save the output as pickle
import pickle
with open('top_connections.pkl', 'wb') as f:
    pickle.dump(out, f)

#%%
