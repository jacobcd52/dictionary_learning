from typing import Dict
import torch as t
from einops import einsum
from trainers.scae import SCAESuite
from buffer import SimpleBuffer
from tqdm import tqdm
import einops

def get_avg_contribs(
        suite: SCAESuite,
        buffer: SimpleBuffer,
        n_batches: int = 10,
    ) -> Dict[str, Dict[str, t.Tensor]]:
    """Calculate average contributions between autoencoder features using sparse representation.
    Memory-efficient implementation.
    """
    from tqdm.auto import tqdm
    
    with t.no_grad():
        # Initialize the results dictionary
        avg_contribs = {
            down_name: {} for down_name in suite.submodule_names
        }
        
        # Process batches with tqdm
        for batch_idx in tqdm(range(n_batches), desc="Computing contributions"):
            cache, tokens = next(buffer)
            
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
                        
                    # Skip if connection not allowed by connections dict
                    if suite.connections is not None and (
                        down_name not in suite.connections or 
                        up_name not in suite.connections[down_name]
                    ):
                        continue

                    # Get upsteam feature acts in sparse form
                    if up_type == 'mlp':
                        inp = cache[f'blocks.{up_layer}.ln2.hook_normalized']
                    elif up_type == 'attn':
                        inp = cache[f'blocks.{up_layer}.ln1.hook_normalized']
                    else:
                        raise ValueError(f"Invalid up_type: {up_type}")
                        
                    _, topk_vals, topk_inds = suite.aes[up_name].encode(inp, return_topk=True)
                    
                    # Get encoder & decoder
                    up_decoder = suite.aes[up_name].decoder.weight
                    down_encoder = suite.aes[down_name].encoder.weight

                    # Find avg contribution tensor
                    if down_type == 'mlp':
                        # Compute virtual weights
                        virtual_weights = einsum(up_decoder, down_encoder, "d f_up, f_down d -> f_up f_down")
                        
                        # Initialize contribution tensor
                        contribution_tensor = t.zeros(
                            virtual_weights.shape,
                            device=virtual_weights.device,
                            dtype=virtual_weights.dtype
                        )
                        
                        # Reshape topk values and indices for vectorized operation
                        batch_size, seq_len = topk_vals.shape[:2]
                        flat_vals = topk_vals.reshape(-1)  # [batch*seq*k]
                        flat_inds = topk_inds.reshape(-1)  # [batch*seq*k]
                        
                        # Compute all contributions at once
                        contributions = virtual_weights[flat_inds] * flat_vals.unsqueeze(-1)  # [batch*seq*k, f_down]
                        
                        # Use scatter_add to sum all contributions
                        contribution_tensor.scatter_add_(
                            0,
                            flat_inds.unsqueeze(-1).expand(-1, virtual_weights.shape[1]),
                            contributions
                        )
                        
                        contribution_tensor = contribution_tensor / (batch_size * seq_len)
                        
                    elif down_type == 'attn':
                        virtual_weights = einsum(up_decoder, suite.W_OVs[down_layer], down_encoder, 
                                              "d_in f_up, n_heads d_in d_out, f_down d_out -> n_heads f_up f_down")
                        
                        # Initialize contribution tensor
                        contribution_tensor = t.zeros(
                            (virtual_weights.shape[1], virtual_weights.shape[2]),  # [f_up, f_down]
                            device=virtual_weights.device,
                            dtype=virtual_weights.dtype
                        )
                        
                        # Get attention probabilities
                        probs = cache[f'blocks.{down_layer}.attn.hook_pattern']  # [b, n_heads, q, k]
                        batch_size, seq_len = topk_vals.shape[:2]
                        n_heads = probs.shape[1]
                        
                        # Process each head separately
                        for head in range(n_heads):
                            head_weights = virtual_weights[head]  # [f_up, f_down]
                            
                            # Reshape for this head
                            flat_vals = topk_vals.reshape(batch_size * seq_len, -1)  # [batch*seq, k]
                            flat_inds = topk_inds.reshape(batch_size * seq_len, -1)  # [batch*seq, k]
                            
                            head_probs = einops.rearrange(probs[:, head], 'b q k -> (b q) k')  # [batch*seq, k]
                            
                            # Get selected weights and scale them
                            selected = head_weights[flat_inds]  # [batch*seq, k, f_down]
                            scaled = selected * (flat_vals * head_probs).unsqueeze(-1)  # [batch*seq, k, f_down]
                            
                            # Flatten batch and k dimensions for scatter_add_
                            scaled = scaled.reshape(-1, scaled.shape[-1])  # [batch*seq*k, f_down]
                            scatter_inds = flat_inds.reshape(-1)  # [batch*seq*k]
                        
                        contribution_tensor = contribution_tensor / (batch_size * seq_len)
                        
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