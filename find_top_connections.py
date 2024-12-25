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
    Compute importance scores between causally valid pairs of autoencoders.
    
    Args:
        suite: The SCAESuite instance
        c: Number of connections to return per downstream feature
        buffer: Iterator yielding batches of data
        num_batches: Number of batches to process
        
    Returns:
        Dictionary mapping downstream module names to dictionaries of importance scores.
        Format: {down_name: {up_name: importance_tensor}}
        where importance_tensor has shape [n_down_features, c] containing indices 
        of connected upstream features
    """
    # Sort module names
    mlp_names = sorted([name for name in suite.submodule_names if name.startswith('mlp_')])
    attn_names = sorted([name for name in suite.submodule_names if name.startswith('attn_')])
    
    # Initialize importance scores dictionary
    importance_scores: Dict[str, Dict[str, sparse.FloatTensor]] = {}
    
    # Process batches
    for batch_idx in tqdm(range(num_batches)):
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
                
                # Create indices for sparse tensor
                down_expanded = down_idxs.unsqueeze(-1).expand(-1, -1, k_up)
                up_expanded = up_idxs.unsqueeze(1).expand(-1, k_down, -1)
                
                # Create and accumulate sparse tensor of importance scores
                indices = t.stack([down_expanded.reshape(-1), up_expanded.reshape(-1)])
                batch_scores = sparse.FloatTensor(
                    indices, 
                    contributions.abs().reshape(-1),
                    scores_shape
                ).coalesce()
                
                if batch_idx == 0:
                    importance_scores[down_name][up_name] = batch_scores
                else:
                    importance_scores[down_name][up_name] = (
                        importance_scores[down_name][up_name] + batch_scores
                    ).coalesce()
    
    # Convert importance scores to connections
    connections = {}
    chunk_size = 1000  # Adjust based on available memory
    
    # Get list of modules to process before we start modifying the dictionary
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
    num_batches=100,
)
# %%
# save the output as pickle
import pickle
with open('top_connections.pkl', 'wb') as f:
    pickle.dump(out, f)

#%%

