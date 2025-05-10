import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from IPython.display import display, HTML

import json
import os
import re

# For SCAE compatibility later
from dictionary_learning.mask_scae import SCAESuite, SCAEModule # SubmoduleName might not be directly used here but good for context
from transformer_lens import HookedTransformer


def display_matplotlib_figure(fig, width=None, height=None):
    """
    Convert a matplotlib figure to an HTML img tag for display in Jupyter notebooks
    
    Parameters:
    - fig: matplotlib figure to display
    - width: optional width (in pixels)
    - height: optional height (in pixels)
    """
    # Save the figure to a PNG in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Encode the PNG as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Set the width and height attributes if provided
    style = ""
    if width is not None:
        style += f"width:{width}px;"
    if height is not None:
        style += f"height:{height}px;"
    
    style_attr = f' style="{style}"' if style else ''
    
    # Generate the HTML
    html = f'<img src="data:image/png;base64,{img_str}"{style_attr}/>'
    
    return html

def create_histogram_html(data, bins=30, title="Histogram", width=600, height=400):
    # Create a histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.3)
    
    # Convert to HTML and display
    plt.close(fig)  # Close the figure to prevent it from displaying twice
    return display_matplotlib_figure(fig, width=width, height=height)

def create_logit_lens_html(top_ind, top_val, bot_ind, bot_val, tokenizer, k=10):
    """
    Create an HTML display of top and bottom tokens with their values.
    
    Parameters:
    - top_ind: indices of top tokens
    - top_val: values of top tokens
    - bot_ind: indices of bottom tokens
    - bot_val: values of bottom tokens
    - tokenizer: tokenizer to decode indices
    - k: number of tokens to display (default 10)
    """
    
    # Decode tokens
    top_text = [tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\newline") for tok in top_ind[:k]]
    bot_text = [tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\newline") for tok in bot_ind[:k]]
    
    # Create HTML template with direct background color attributes
    html_template = """
    <style>
        .token-table {{
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }}
        .token-table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .title {{
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }}
    </style>
    
    <div class="title">Logit Lens</div>
    
    <table class="token-table">
        <tr>
            <td><b>Top Token</b></td>
            <td><b>Value</b></td>
            <td><b>Bottom Token</b></td>
            <td><b>Value</b></td>
        </tr>
    """
    
    # Add rows with colored backgrounds applied to spans instead of cells
    for i in range(k):
        html_template += f"""
        <tr>
            <td>{top_text[i]}</td>
            <td><span style="background-color: #0000FF; color: white; padding: 2px 4px; display: inline-block;"><b>{top_val[i].item():.3f}</b></span></td>
            <td>{bot_text[i]}</td>
            <td><span style="background-color: #FF0000; color: white; padding: 2px 4px; display: inline-block;"><b>{bot_val[i].item():.3f}</b></span></td>
        </tr>
        """
    
    html_template += "</table>"
    
    return html_template

def make_colorbar(min_value, max_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    # Add color bar
    colorbar = ""
    num_colors = 4
    if(min_value < -negative_threshold):
        for i in range(num_colors, 0, -1):
            ratio = i / (num_colors)
            value = round((min_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    # Do zero
    colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'
    # Do positive
    if(max_value > positive_threshold):
        for i in range(1, num_colors+1):
            ratio = i / (num_colors)
            value = round((max_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    return colorbar

def value_to_color(activation, max_value, min_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    if activation > positive_threshold:
        ratio = activation/max_value if max_value != 0 else 0 # Avoid division by zero
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value if min_value != 0 else 0 # Avoid division by zero
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
    else:
        text_color = "0,0,0"
        background_color = f'rgba({white},{white},{white},1)'
    return text_color, background_color

def convert_token_array_to_list(array):
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim()==2:
            array = array.tolist()
        else: 
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        # ensure it's a list of lists
        if array and isinstance(array[0], int): # check if array is not empty
            array = [array]
    return array

def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None, model_type="causal", text_above_each_act=None):
    text_spacing = "0.00em"
    toks = convert_token_array_to_list(toks)
    activations = convert_token_array_to_list(activations)
    
    # Ensure toks is not empty and its elements are lists before proceeding
    if not toks or not isinstance(toks[0], list):
        # Handle empty or incorrectly formatted toks (e.g., return empty string or raise error)
        # For now, let's assume it implies no tokens to display or an issue upstream.
        # Depending on expected behavior, this might need adjustment.
        # This check helps prevent errors if, for example, convert_token_array_to_list returns an empty list.
        return "<!-- No tokens to display or token format error -->"

    toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '\\n') for t in tok_seq] for tok_seq in toks]
    
    highlighted_text = []
    highlighted_text.append("""
<body style="background-color: black; color: white;">
""")
    
    # Handle cases where activations might be empty or not structured as expected
    if not activations or not any(activations): # Check if activations is empty or contains only empty lists
        max_value = 0
        min_value = 0
    else:
        # Filter out empty lists from activations before calculating max/min
        non_empty_activations = [act_seq for act_seq in activations if act_seq]
        if not non_empty_activations: # All lists were empty
             max_value = 0
             min_value = 0
        else:
            max_value = max([max(act_seq) for act_seq in non_empty_activations])
            min_value = min([min(act_seq) for act_seq in non_empty_activations])

    if logit_diffs is not None and model_type != "reward_model":
        # Similar safety for logit_diffs
        if not logit_diffs or not any(logit_diffs):
            logit_max_value = 0
            logit_min_value = 0
        else:
            non_empty_logit_diffs = [ld_seq for ld_seq in logit_diffs if ld_seq]
            if not non_empty_logit_diffs:
                logit_max_value = 0
                logit_min_value = 0
            else:
                logit_max_value = max([max(ld_seq) for ld_seq in non_empty_logit_diffs])
                logit_min_value = min([min(ld_seq) for ld_seq in non_empty_logit_diffs])


    highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
    if(logit_diffs is not None and model_type != "reward_model"):
        highlighted_text.append('<div style="margin-top: 0.1em;"></div>')
        highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value if 'logit_max_value' in locals() else 0)) # Ensure logit_max_value is defined
    
    highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
    for seq_ind, (act_seq, tok_seq) in enumerate(zip(activations, toks)):
        if(text_above_each_act is not None and seq_ind < len(text_above_each_act)):
            highlighted_text.append(f'<span>{text_above_each_act[seq_ind]}</span>')
        
        # Ensure act_seq and tok_seq are iterable and of same length
        if not isinstance(act_seq, list) or not isinstance(tok_seq, list): continue # Skip if not lists
        
        for act_ind, (a, t) in enumerate(zip(act_seq, tok_seq)):
            if(logit_diffs is not None and model_type != "reward_model"):
                highlighted_text.append('<div style="display: inline-block;">')
            
            text_color, background_color = value_to_color(a, max_value, min_value)
            highlighted_text.append(f'<span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{t.replace(" ", "&nbsp")}</span>')
            
            if(logit_diffs is not None and model_type != "reward_model"):
                # Ensure logit_diffs[seq_ind] and its elements are accessible
                if seq_ind < len(logit_diffs) and act_ind < len(logit_diffs[seq_ind]):
                    logit_diffs_act = logit_diffs[seq_ind][act_ind]
                    _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value if 'logit_max_value' in locals() else 0 , logit_min_value if 'logit_min_value' in locals() else 0)
                    highlighted_text.append(f'<div style="display: block; margin-right: {text_spacing}; height: 10px; background-color:{logit_background_color}; text-align: center;"></div>')
                highlighted_text.append('</div>') # Close inline-block div

        if(logit_diffs is not None and model_type=="reward_model"):
            if seq_ind < len(logit_diffs) and hasattr(logit_diffs[seq_ind], 'item'): # Check if it's a tensor/item
                 reward_change = logit_diffs[seq_ind].item()
                 text_color, background_color = value_to_color(reward_change, 10, -10) # Assuming fixed scale for reward
                 highlighted_text.append(f'<br><span>Reward: </span><span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{reward_change:.2f}</span>')
        highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
    
    highlighted_text = ''.join(highlighted_text)
    return highlighted_text

def save_token_display(tokens, activations, tokenizer, path, save=True, logit_diffs=None, show=False, model_type="causal", text_above_each_act=None):
    html = tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs, model_type=model_type, text_above_each_act=text_above_each_act)
    # if(save):
    #     # imgkit.from_string(html, path) # imgkit might not be available
    #     with open(path, "w") as f:
    #           f.write(html) # Save as HTML file instead
    if(show):
        return display(HTML(html))
    return html # Return html string if not showing/saving image


def get_feature_indices(feature_activations, k=10, setting="max"):
    # Ensure feature_activations is a 2D tensor
    if not (isinstance(feature_activations, torch.Tensor) and feature_activations.dim() == 2):
        # Fallback or error for unexpected input shape
        # This depends on how you want to handle it. For now, returning empty tensors.
        # Consider logging a warning or raising an error.
        print(f"Warning: get_feature_indices expected a 2D tensor, got {type(feature_activations)} with dim {feature_activations.dim() if isinstance(feature_activations, torch.Tensor) else 'N/A'}")
        return torch.tensor([]).long(), torch.tensor([]).long()

    batch_size, seq_len = feature_activations.shape
    if batch_size == 0 or seq_len == 0: # Handle empty tensor
        return torch.tensor([]).long(), torch.tensor([]).long()

    feature_activations_flat = einops.rearrange(feature_activations, 'b s -> (b s)')
    
    if feature_activations_flat.numel() == 0: # Handle case where tensor becomes empty after rearrange
        return torch.tensor([]).long(), torch.tensor([]).long()

    actual_k = min(k, feature_activations_flat.numel()) # Adjust k if fewer elements than requested

    if setting=="max":
        found_indices = torch.argsort(feature_activations_flat, descending=True)[:actual_k]
    elif setting=="uniform":
        min_val = torch.min(feature_activations_flat)
        max_val = torch.max(feature_activations_flat)

        if min_val == max_val: # All values are the same, just take the first k
             found_indices = torch.arange(actual_k, device=feature_activations_flat.device)
        else:
            bin_boundaries = torch.linspace(min_val, max_val, actual_k + 1, device=feature_activations_flat.device)
            bins = torch.bucketize(feature_activations_flat, bin_boundaries[:-1]) # Exclude last boundary for bucketize

            sampled_indices = []
            unique_bins = torch.unique(bins)
            
            # Ensure we don't try to sample more than available per bin or more than actual_k total
            samples_per_bin = max(1, actual_k // len(unique_bins) if len(unique_bins) > 0 else 1)

            for bin_idx in unique_bins:
                if len(sampled_indices) >= actual_k: break # Stop if we have enough samples

                bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
                if bin_indices.numel() > 0:
                    # Sample without replacement, up to samples_per_bin or remaining needed samples
                    num_to_sample = min(samples_per_bin, bin_indices.numel(), actual_k - len(sampled_indices))
                    
                    perm = torch.randperm(bin_indices.numel(), device=bin_indices.device)[:num_to_sample]
                    sampled_indices.extend(bin_indices[perm])
            
            if not sampled_indices: # Fallback if sampling failed (e.g. all bins empty, though unlikely with actual_k > 0)
                 found_indices = torch.arange(actual_k, device=feature_activations_flat.device)
            else:
                 found_indices = torch.tensor(sampled_indices, device=feature_activations_flat.device).long()
                 # Optionally, sort them or take the top if more were selected than actual_k
                 if found_indices.numel() > actual_k:
                     found_indices = found_indices[:actual_k] # Trim if oversampled
                 # Uniform sampling doesn't imply an order, but for consistency with 'max', could sort by activation
                 # For now, keep the sampled order or reverse if a "high to low" semantic is desired from linspace
                 # found_indices = found_indices.flip(dims=[0]) # if high-to-low activation order is desired

    else: # random
        nonzero_indices = torch.nonzero(feature_activations_flat, as_tuple=False).squeeze(dim=1)
        if nonzero_indices.numel() == 0: # If all activations are zero
            # Fallback: take first actual_k indices if k > 0, or handle as error/empty
            if actual_k > 0:
                found_indices = torch.arange(actual_k, device=feature_activations_flat.device)
            else:
                return torch.tensor([]).long(), torch.tensor([]).long()
        else:
            shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.numel(), device=nonzero_indices.device)]
            found_indices = shuffled_indices[:min(actual_k, shuffled_indices.numel())] # Ensure we don't go out of bounds

    if found_indices.numel() == 0: # If no indices were found (e.g. actual_k was 0)
        return torch.tensor([]).long(), torch.tensor([]).long()
        
    d_indices = found_indices // seq_len
    s_indices = found_indices % seq_len
    return d_indices, s_indices

def get_feature_datapoints(d_idx, seq_pos_idx, all_activations, all_tokens, tokenizer):
    # all_activations: tensor of shape (batch, seq_len) for a single feature
    # all_tokens: tensor of shape (batch, seq_len) or list of lists of tokens
    
    full_activations_list = []
    partial_activations_list = []
    text_list = []
    full_text_list = []
    token_list_for_html = [] # For tokens_and_activations_to_html
    # full_token_list = [] # For full sequence context if needed elsewhere

    # Ensure all_tokens is a tensor for consistent indexing
    if isinstance(all_tokens, list):
        try:
            if all_tokens and isinstance(all_tokens[0], torch.Tensor):
                 all_tokens_tensor = torch.stack(all_tokens) if all_tokens else torch.empty(0,0, dtype=torch.long)
            elif all_tokens and isinstance(all_tokens[0], list):
                 max_len = max(len(t_list) for t_list in all_tokens) if all_tokens else 0
                 all_tokens_tensor = torch.tensor([t_list + [tokenizer.pad_token_id]*(max_len - len(t_list)) for t_list in all_tokens], dtype=torch.long)
            else: 
                 all_tokens_tensor = torch.empty(0,0, dtype=torch.long)
        except Exception as e:
            print(f"Error converting all_tokens to tensor: {e}")
            all_tokens_tensor = torch.empty(0,0, dtype=torch.long) 
    elif isinstance(all_tokens, torch.Tensor):
        all_tokens_tensor = all_tokens
    else:
        raise ValueError("all_tokens must be a list of lists/tensors or a 2D tensor of token IDs")

    if all_tokens_tensor.numel() == 0: 
        return [], [], [], [], [], []

    for md, s_ind in zip(d_idx.tolist(), seq_pos_idx.tolist()):
        if md >= all_tokens_tensor.shape[0] or md >= all_activations.shape[0]:
            print(f"Warning: Index md={md} out of bounds.")
            continue
            
        current_full_tokens = all_tokens_tensor[md] 
        
        if s_ind >= current_full_tokens.shape[0] or s_ind >= all_activations.shape[1]:
            print(f"Warning: Index s_ind={s_ind} out of bounds for sequence length.")
            continue

        tokens_for_display = current_full_tokens[:s_ind+1].tolist() 
        
        text = tokenizer.decode(tokens_for_display)
        text_list.append(text)
        
        full_text_list.append(tokenizer.decode(current_full_tokens.tolist())) 

        token_list_for_html.append(tokens_for_display) 
        
        current_feature_activations_on_seq = all_activations[md]
        partial_acts = current_feature_activations_on_seq[:s_ind+1].tolist()
        partial_activations_list.append(partial_acts)
        
        full_acts = current_feature_activations_on_seq.tolist()
        full_activations_list.append(full_acts)

    return text_list, full_text_list, token_list_for_html, all_tokens_tensor.tolist(), partial_activations_list, full_activations_list 

def get_feature_connections_scae(
    source_module_name: str, 
    source_feature_global_idx: int, 
    scae_suite: SCAESuite, 
    features_to_save_scae: dict, 
    temperature: float = 0.01, 
    top_k_connections: int = 20,
    connection_threshold: float = 0.01 # Minimum absolute value to consider a connection significant
):
    """
    Get connections from a source feature in an SCAEModule's autoencoder to features 
    in other (downstream) SCAEModules within the SCAESuite, using learned connection masks.

    Args:
        source_module_name: The name of the source SCAEModule (e.g., 'mlp_0').
        source_feature_global_idx: The global index of the feature in the source module's AE.
        scae_suite: The SCAESuite object containing all modules and their connections.
        features_to_save_scae: Dict mapping module names to lists of their global feature indices
                               that are included in the viewer (to filter target features).
        temperature: Temperature for evaluating the connection masks.
        top_k_connections: The maximum number of connections to return.
        connection_threshold: The minimum absolute strength for a connection to be considered.

    Returns:
        A list of dictionaries, each representing a connection, sorted by absolute strength.
        Each dictionary has: {'target_key': str, 'target_feature': int, 'value': float}
    """
    feature_connections = []
    if source_module_name in scae_suite.module_dict:
        source_scae_module = scae_suite.module_dict[source_module_name]
    else:
        source_scae_module = None

    if not source_scae_module:
        print(f"Warning: Source module {source_module_name} not found in SCAESuite.")
        return []

    # Iterate through all modules in the SCAESuite to find potential downstream targets
    for target_module_name, target_scae_module in scae_suite.module_dict.items():
        # A module cannot connect to itself in this context of upstream/downstream AEs
        if source_module_name == target_module_name:
            continue

        # Check if the source_module is an upstream AE for the target_module
        # The connection_masks are stored in the *downstream* module and indexed by *upstream* module names
        if target_scae_module.connection_masks and source_module_name in target_scae_module.connection_masks:
            connection_mask_module = target_scae_module.connection_masks[source_module_name]
            
            # Evaluate the mask. Expected shape: (n_features_target_ae, n_features_source_ae)
            # Note: The mask in LearnableMask is (n_features_down, n_features_up)
            # Here, target_ae is 'downstream' and source_ae is 'upstream'
            evaluated_mask_tensor = connection_mask_module(temperature) 

            # Ensure source_feature_global_idx is within bounds for the source AE's features in this mask
            if source_feature_global_idx >= evaluated_mask_tensor.shape[1]:
                # This might happen if the mask dimensions don't align with global indices as expected
                # Or if source_feature_global_idx is for a different AE concept
                # print(f"Warning: source_feature_global_idx {source_feature_global_idx} is out of bounds for connection mask from {source_module_name} to {target_module_name}. Mask shape: {evaluated_mask_tensor.shape}")
                continue

            # Get the connection strengths from the source_feature_global_idx to all features in the target AE
            # This is a column in the evaluated_mask_tensor
            connections_to_target_features = evaluated_mask_tensor[:, source_feature_global_idx]

            # Iterate through these connections
            for target_feature_local_idx, strength in enumerate(connections_to_target_features):
                connection_strength = strength.item()
                
                # Consider only significant connections
                if abs(connection_strength) > connection_threshold:
                    # target_feature_local_idx is local to the target_scae_module's AE
                    # Check if this target feature is one we are visualizing
                    if target_module_name in features_to_save_scae and \
                       target_feature_local_idx in features_to_save_scae[target_module_name]:
                        feature_connections.append({
                            "target_key": target_module_name,
                            "target_feature": target_feature_local_idx, # This is the global index for target AE
                            "value": connection_strength
                        })
    
    # Sort connections by strength (absolute value) in descending order
    feature_connections.sort(key=lambda x: abs(x["value"]), reverse=True)
    
    return feature_connections[:top_k_connections] 

def generate_enhanced_viewer_scae(
    model: HookedTransformer, # Added
    scae_suite: SCAESuite,    # Added
    saved_pruned_features_list: dict, # Changed from saved_feature_act_list, this is dict mapping module_name to (batch, seq, n_ae_features)
    saved_token_list: list, # This should be the raw token dataset, e.g., list of lists or tensor
    tokenizer,
    features_to_save_scae: dict, # Dict mapping module_name (key) to list of global feature indices for that AE
    connection_temperature: float = 0.01, # Added for get_feature_connections_scae
    output_dir="llm_feature_viewer_scae", 
    model_save_name="",
    CHUNK_SIZE = 100,
    num_feature_datapoints = 10 # Number of example texts to show per feature
):
    """
    Generate an enhanced HTML viewer for SCAESuite features.
    
    Args:
        model: The HookedTransformer model.
        scae_suite: The trained SCAESuite object.
        saved_pruned_features_list: Dict mapping module names (e.g., 'mlp_0') to their pruned feature activations 
                                    (typically shape: [batch_size, seq_len, n_module_ae_features]).
        saved_token_list: List of token sequences (input data to the model, e.g., from a dataset).
        tokenizer: The tokenizer.
        features_to_save_scae: Dictionary mapping module names to lists of global feature indices for that module's AE 
                               that you want to include in the viewer.
        connection_temperature: Temperature for evaluating connection masks in get_feature_connections_scae.
        output_dir: Directory where the viewer files will be saved.
        model_save_name: Optional prefix for the output directory.
        CHUNK_SIZE: How many features' data to save per JS chunk file.
        num_feature_datapoints: Number of example texts to show per feature based on top activations.
        
    Returns:
        The path to the generated index.html file.
    """
    
    keys = list(scae_suite.module_dict.keys()) # These are the module names like 'mlp_0', 'attn_5'

    # Prepare output directory
    if model_save_name:
        output_dir = f"{model_save_name}_{output_dir}"
    
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create manifest data
    manifest = {
        "keys": keys, # These are module names
        "features": {} # This will map module_name to list of stringified global feature indices for that module's AE
    }
    
    for key_module_name in keys:
        # Ensure features_to_save_scae has an entry for this module, even if empty
        manifest["features"][key_module_name] = [str(f) for f in features_to_save_scae.get(key_module_name, [])]
        key_dir = os.path.join(data_dir, key_module_name)
        os.makedirs(key_dir, exist_ok=True)

    # Function to generate HTML for feature connections (reused from original, ensure it's defined above or here)
    def generate_connections_html(connections_list):
        if not connections_list:
            return "<div class='no-connections'>No significant connections found</div>"
            
        html_parts = ["<div class='connections-container'>",
                "<h3>Feature Connections</h3>",
                "<table class='connections-table'>",
                "<tr><th>Connected Feature</th><th>Connection Strength</th></tr>"]
                
        for conn in connections_list:
            target_key_html = conn["target_key"]
            target_feature_html = conn["target_feature"]
            conn_value_html = conn["value"]
            
            value_class = "positive-connection" if conn_value_html > 0 else "negative-connection"
            
            html_parts.append(f"<tr>")
            html_parts.append(f"<td><a href='index.html?key={target_key_html}&feature={target_feature_html}' class='feature-link'>{target_key_html} - Feature {target_feature_html}</a></td>")
            html_parts.append(f"<td class='{value_class}'>{conn_value_html:.4f}</td>")
            html_parts.append(f"</tr>")
            
        html_parts.append("</table></div>")
        return "\n".join(html_parts)

    # Function to safely save feature chunk data (reused from original)
    def save_feature_chunk(key_module_name_chunk, chunk_idx, new_chunk_data, data_dir_chunk):
        chunk_path = os.path.join(data_dir_chunk, key_module_name_chunk, f"chunk_{chunk_idx}.js")
        existing_data = {}
        if os.path.exists(chunk_path):
            try:
                with open(chunk_path, 'r') as f_chunk:
                    content = f_chunk.read()
                start_marker = "window.featureChunk = "
                start_pos = content.find(start_marker)
                if start_pos != -1:
                    start_pos += len(start_marker)
                    open_braces = 0
                    for i_chunk in range(start_pos, len(content)):
                        if content[i_chunk] == '{':
                            open_braces += 1
                        elif content[i_chunk] == '}':
                            open_braces -= 1
                            if open_braces == 0 and i_chunk + 1 < len(content) and content[i_chunk+1:].lstrip().startswith(';'):
                                json_str = content[start_pos-1:i_chunk+1]
                                try:
                                    existing_data = json.loads(json_str)
                                    break
                                except json.JSONDecodeError as e_json:
                                    print(f"Warning: Invalid JSON in chunk file {chunk_path}: {e_json}")
                                    break
            except Exception as e_read:
                print(f"Warning: Could not read existing chunk file {chunk_path}: {e_read}")
        
        merged_data = {**existing_data, **new_chunk_data}
        chunk_js = f'''// Features chunk {chunk_idx} data for {key_module_name_chunk}\nwindow.featureChunk = {json.dumps(merged_data)};\n'''
        with open(chunk_path, 'w') as f_chunk_write:
            f_chunk_write.write(chunk_js)

    # Generate chunked data files for each key (module_name)
    for key_module_name_outer in keys: # This is a module name like 'mlp_0'
        # features_for_this_module is a list of global feature indices for this module's AE
        features_for_this_module = features_to_save_scae.get(key_module_name_outer, [])
        
        if not features_for_this_module:
            continue # Skip if no features are selected for this module

        scae_module_obj = scae_suite.module_dict[key_module_name_outer]
        module_ae = scae_module_obj.ae # The AutoEncoderTopK for this module
        
        feature_chunks = {}
        
        # saved_pruned_features_list[key_module_name_outer] is (batch, seq, n_ae_features_for_this_module)
        # The feature_global_idx here is an index into the *dictionary* of this module_ae
        for feature_global_idx_in_ae in features_for_this_module:
            feature_str = str(feature_global_idx_in_ae)
            chunk_idx = feature_global_idx_in_ae // CHUNK_SIZE
            
            if chunk_idx not in feature_chunks:
                feature_chunks[chunk_idx] = {}
            
            feature_data_dict = {
                "tokenActivations": "",
                "logitLens": "",
                "histogram": "",
                "connections": ""
            }
            
            # Get all activations for this specific feature across all batches and sequences
            # This has shape (batch_size, seq_len)
            current_feature_activations_all_samples = saved_pruned_features_list[key_module_name_outer][..., feature_global_idx_in_ae]
            
            if current_feature_activations_all_samples.numel() == 0:
                print(f"Warning: No activations found for {key_module_name_outer} feature {feature_global_idx_in_ae}. Skipping.")
                feature_data_dict["tokenActivations"] = "<div class='error-panel'>No activation data.</div>"
                feature_data_dict["logitLens"] = "<div class='error-panel'>No activation data for logit lens.</div>"
                feature_data_dict["histogram"] = "<div class='error-panel'>No activation data for histogram.</div>"
                feature_data_dict["connections"] = "<div class='error-panel'>No activation data for connections.</div>"
                feature_chunks[chunk_idx][feature_str] = feature_data_dict
                continue

            d_idx, seq_idx = get_feature_indices(current_feature_activations_all_samples, k=num_feature_datapoints, setting="max")
            
            if d_idx.numel() == 0 : # No top datapoints found
                print(f"Warning: No top datapoints found for {key_module_name_outer} - feature {feature_global_idx_in_ae}. Max act: {current_feature_activations_all_samples.max() if current_feature_activations_all_samples.numel() > 0 else 'N/A'}")
                text_list_html, _, token_list_html, _, partial_activations_html, _ = [], [], [], [], [], [] # Empty data
            else:
                text_list_html, _, token_list_html, _, partial_activations_html, _ = get_feature_datapoints(
                    d_idx, seq_idx, current_feature_activations_all_samples, saved_token_list, tokenizer
                )
            
            token_html = tokens_and_activations_to_html(token_list_html, partial_activations_html, tokenizer)
            feature_data_dict["tokenActivations"] = token_html
            
            try:
                feature_decoder = module_ae.decoder.weight[:, feature_global_idx_in_ae] # (d_model,)
                # Ensure model has W_U and ln_final if not using a different model structure
                if hasattr(model, 'W_U') and hasattr(model, 'ln_final'):
                    unembd = model.W_U # (d_model, d_vocab)
                    final_ln = model.ln_final # LayerNorm
                    
                    # Logits produced by this feature: ln_final(feature_decoder) @ W_U
                    # feature_decoder is already in d_model space from AE
                    logit_lens_values = final_ln(feature_decoder) @ unembd # (d_vocab,)
                    top_val, top_ind = torch.topk(logit_lens_values, k=10, dim=-1)
                    bot_val, bot_ind = torch.topk(logit_lens_values, k=10, dim=-1, largest=False)
                    
                    logit_lens_html_content = create_logit_lens_html(top_ind, top_val, bot_ind, bot_val, tokenizer)
                    feature_data_dict["logitLens"] = logit_lens_html_content
                else:
                    feature_data_dict["logitLens"] = "<div class='error-panel'>Model does not have W_U or ln_final.</div>"

            except Exception as e_logit:
                print(f"Error in logit lens for {key_module_name_outer} - feature {feature_global_idx_in_ae}: {e_logit}")
                feature_data_dict["logitLens"] = f"<div class='error-panel'>Logit lens visualization unavailable: {str(e_logit)}</div>"
            
            try:
                # Activations for histogram: use all non-zero activations for this feature
                nz_feature_act = current_feature_activations_all_samples[current_feature_activations_all_samples != 0]
                if nz_feature_act.numel() > 0:
                    frequency = nz_feature_act.numel() / current_feature_activations_all_samples.numel()
                    hist_html_content = create_histogram_html(nz_feature_act.float().cpu().numpy(), 
                                                       title=f"Activation Frequency {frequency*100:.2f}%")
                    feature_data_dict["histogram"] = hist_html_content
                else:
                    feature_data_dict["histogram"] = "<div class='error-panel'>No non-zero activations for histogram.</div>"
            except Exception as e_hist:
                print(f"Error in hist for {key_module_name_outer} - feature {feature_global_idx_in_ae}: {e_hist}")
                feature_data_dict["histogram"] = f"<div class='error-panel'>Histogram visualization unavailable: {str(e_hist)}</div>"
            
            # Get feature connections using the new SCAE-specific function
            # source_feature_global_idx is indeed feature_global_idx_in_ae for this module's AE
            feature_connections_list = get_feature_connections_scae(
                source_module_name=key_module_name_outer, 
                source_feature_global_idx=feature_global_idx_in_ae, 
                scae_suite=scae_suite, 
                features_to_save_scae=features_to_save_scae, # Pass the main dict for filtering targets
                temperature=connection_temperature
            )
            
            connections_html_content = generate_connections_html(feature_connections_list)
            feature_data_dict["connections"] = connections_html_content
            
            feature_chunks[chunk_idx][feature_str] = feature_data_dict
        
        for chunk_idx_save, chunk_data_save in feature_chunks.items():
            save_feature_chunk(key_module_name_outer, chunk_idx_save, chunk_data_save, data_dir)
    
    manifest_js_content = f'''// Feature manifest data
const manifestData = {json.dumps(manifest)};
// Configuration
const CHUNK_SIZE = {CHUNK_SIZE};
'''
    
    manifest_path = os.path.join(output_dir, "manifest.js")
    with open(manifest_path, 'w') as f_manifest:
        f_manifest.write(manifest_js_content)
    
    viewer_html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCAE Feature Viewer</title> <!-- Title updated -->
    <style>
        /* Reset and base styles */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        /* Header styles */
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 24px;
        }
        
        /* Controls section */
        .controls {
            background-color: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .control-group {
            flex: 1;
            min-width: 200px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        
        select, input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: #fff;
            font-size: 16px;
        }
        
        .feature-input-container {
            display: flex;
            gap: 10px;
        }
        
        .feature-input-container input {
            flex: 1;
        }
        
        .feature-input-container button {
            padding: 10px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        /* Navigation section */
        .navigation {
            display: flex;
            justify-content: space-between;
            padding: 15px 20px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }
        
        .button {
            padding: 8px 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
        }
        
        .button:hover {
            background-color: #0069d9;
        }
        
        .button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        
        /* Multi-panel layout */
        .upper-panels-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .panel-cell {
            width: 50%;
            padding: 0 10px;
            vertical-align: top;
        }
        
        .panel {
            border: 1px solid #ddd;
            border-radius: 4px;
            height: 100%;
        }
        
        .panel-header {
            background-color: #f1f1f1;
            padding: 10px;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
        }
        
        .panel-content {
            padding: 15px;
            overflow: auto;
            min-height: 225px; /* Adjusted min-height */
        }
        
        /* Content section for token activations */
        .token-activations-container {
            background-color: black;
            color: white;
            min-height: 300px; /* Adjusted min-height */
            padding: 20px;
            overflow: auto;
            margin: 0 20px 20px 20px;
            border-radius: 4px;
        }
        
        /* Connections section */
        .connections-section {
            margin: 0 20px 20px 20px;
        }
        
        /* Loading indicator */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 150px; /* Adjusted height */
            font-size: 18px;
            color: #6c757d;
        }
        
        /* Footer section */
        .footer {
            background-color: #f8f9fa;
            text-align: center;
            padding: 15px;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }
        
        /* Content styles for token activations */
        .content {
            font-family: monospace;
            line-height: 1.4;
            white-space: pre-wrap; /* Changed from pre to pre-wrap */
        }

        /* Error panel */
        .error-panel {
            background-color: #ffe6e6;
            border: 1px solid #ffcccc;
            color: #990000;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        
        /* Show/hide elements */
        .hidden {
            display: none !important;
        }
        
        /* Token table (for logit lens) */
        .token-table {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }
        
        .token-table td {
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        
        .title { /* Shared by logit lens and potentially others */
            font-size: 18px; /* Adjusted size */
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        
        /* Fix for histogram images */
        .histogram-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%; /* Ensure it takes full cell width */
            height: 100%; /* Ensure it takes full cell height if panel-content has fixed height */
        }
        
        .histogram-container img {
            max-width: 100%;
            max-height: 280px; /* Max height for histogram image */
            width: auto;
            height: auto;
            object-fit: contain;
        }
        
        /* Connections styles */
        .connections-container {
            margin-top: 15px; /* Adjusted margin */
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }
        
        .connections-container h3 {
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 16px; /* Adjusted size */
            color: #495057;
        }
        
        .connections-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        
        .connections-table th {
            background-color: #e9ecef;
            padding: 8px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
        }
        
        .connections-table td {
            padding: 8px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .feature-link {
            color: #007bff;
            text-decoration: none;
        }
        
        .feature-link:hover {
            text-decoration: underline;
        }
        
        .positive-connection {
            color: #28a745;
            font-weight: bold;
        }
        
        .negative-connection {
            color: #dc3545;
            font-weight: bold;
        }
        
        .no-connections {
            font-style: italic;
            color: #6c757d;
            padding: 10px;
            text-align: center;
        }
    </style>
    <script src="manifest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SCAE Feature Viewer</h1> <!-- Title updated -->
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="keySelect">Select Module:</label> <!-- Label updated -->
                <select id="keySelect">
                    <option value="" disabled selected>Choose a module</option>
                </select>
            </div>
            <div class="control-group">
                <label for="featureInput">Go to Feature # (in selected module):</label> <!-- Label updated -->
                <div class="feature-input-container">
                    <input type="number" id="featureInput" min="0" placeholder="Enter feature number">
                    <button id="goToFeature" class="button">Go</button>
                </div>
            </div>
        </div>
        
        <div class="navigation">
            <button id="prevFeature" class="button" disabled>Previous Feature</button>
            <div>
                <span id="featureInfo">No feature selected</span>
            </div>
            <button id="nextFeature" class="button" disabled>Next Feature</button>
        </div>
        
        <table class="upper-panels-table">
            <tr>
                <td class="panel-cell">
                    <div class="panel">
                        <div class="panel-header">Logit Lens</div>
                        <div id="logitLensPanel" class="panel-content">
                            <div class="loading">Select a feature to view logit lens</div>
                        </div>
                    </div>
                </td>
                <td class="panel-cell">
                    <div class="panel">
                        <div class="panel-header">Activation Histogram</div>
                        <div id="histogramPanel" class="panel-content histogram-container"> <!-- Added histogram-container class -->
                            <div class="loading">Select a feature to view histogram</div>
                        </div>
                    </div>
                </td>
            </tr>
        </table>
        
        <div class="token-activations-container">
            <div id="loadingIndicator" class="loading">
                <p>Select a module and feature to view token activations</p>
            </div>
            <div id="contentDisplay" class="content hidden"></div>
        </div>
        
        <div class="connections-section">
            <div id="connectionsPanel"> <!-- Content will be filled here -->
                 <div class="loading">Select a feature to view connections</div>
            </div>
        </div>
        
        <div class="footer">
            <p>SCAE Feature Viewer | Features Explorer</p>
        </div>
    </div>
    
    <script>
        const keySelect = document.getElementById('keySelect');
        const featureInput = document.getElementById('featureInput');
        const goToFeatureBtn = document.getElementById('goToFeature');
        const prevFeatureBtn = document.getElementById('prevFeature');
        const nextFeatureBtn = document.getElementById('nextFeature');
        const featureInfo = document.getElementById('featureInfo');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const contentDisplay = document.getElementById('contentDisplay');
        const logitLensPanel = document.getElementById('logitLensPanel');
        const histogramPanel = document.getElementById('histogramPanel');
        const connectionsPanel = document.getElementById('connectionsPanel');
     
        let currentFeatureData = null;
        let currentSelectedKey = null;
        let currentSelectedFeature = null;
        
        function getUrlParams() {
            const params = {};
            const searchParams = new URLSearchParams(window.location.search);
            for (const [key, value] of searchParams) params[key] = value;
            return params;
        }
        
        function getChunkIndex(feature) {
            return Math.floor(parseInt(feature) / CHUNK_SIZE);
        }
        
        function loadFeatureData(key, feature) {
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error('Timeout loading feature data')), 10000);
                const chunkIndex = getChunkIndex(feature);
                const existingScript = document.getElementById('featureChunkScript');
                if (existingScript) document.head.removeChild(existingScript);
                window.featureChunk = null;
                const script = document.createElement('script');
                script.id = 'featureChunkScript';
                script.src = `data/${key}/chunk_${chunkIndex}.js`;
                script.onload = () => {
                    clearTimeout(timeout);
                    if (window.featureChunk && window.featureChunk[feature]) {
                        resolve(window.featureChunk[feature]);
                    } else {
                        reject(new Error(`Feature ${feature} not found in chunk ${chunkIndex} for module ${key}. Chunk content: ${JSON.stringify(window.featureChunk)}`));
                    }
                };
                script.onerror = () => {
                    clearTimeout(timeout);
                    reject(new Error(`Failed to load chunk ${chunkIndex} for ${key}`));
                };
                document.head.appendChild(script);
            });
        }
        
        function initViewer() {
            if (!window.manifestData || !manifestData.keys || !manifestData.features) {
                showError('Failed to load manifest data. Check manifest.js and ensure it is correctly formatted and loaded.');
                return;
            }
            
            manifestData.keys.forEach(key => {
                const option = document.createElement('option');
                option.value = key; option.textContent = key;
                keySelect.appendChild(option);
            });
            
            keySelect.addEventListener('change', handleKeyChange);
            goToFeatureBtn.addEventListener('click', handleGoToFeature);
            featureInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleGoToFeature(); });
            prevFeatureBtn.addEventListener('click', showPreviousFeature);
            nextFeatureBtn.addEventListener('click', showNextFeature);
            
            const params = getUrlParams();
            if (params.key && params.feature && manifestData.keys.includes(params.key)) {
                keySelect.value = params.key;
                // Delay handleKeyChange slightly to ensure event listeners are ready and manifest fully parsed
                setTimeout(() => {
                     handleKeyChange(false); // Pass false to not load first feature by default yet
                     loadFeature(params.key, params.feature);
                }, 50);
            } else if (manifestData.keys.length > 0) {
                keySelect.value = manifestData.keys[0];
                setTimeout(() => handleKeyChange(true), 50); // Load first feature of first key
            }
        }
        
        function showError(message, panelId = null) {
            const errorHtml = `<div class="error-panel">${message}</div>`;
            if (panelId) {
                const panel = document.getElementById(panelId);
                if (panel) panel.innerHTML = errorHtml;
            } else { // Global error for token activations section or if panelId is null
                 loadingIndicator.classList.remove('hidden');
                 loadingIndicator.innerHTML = errorHtml;
                 contentDisplay.classList.add('hidden');
            }
            // Clear other panels to avoid confusion
            if (panelId !== 'logitLensPanel') logitLensPanel.innerHTML = '<div class="loading">Error loading data.</div>';
            if (panelId !== 'histogramPanel') histogramPanel.innerHTML = '<div class="loading">Error loading data.</div>';
            if (panelId !== 'connectionsPanel') connectionsPanel.innerHTML = '<div class="loading">Error loading data.</div>';
            if (panelId !== 'loadingIndicator') contentDisplay.classList.add('hidden');
        }
        
        function handleKeyChange(loadFirstFeature = true) {
            currentSelectedKey = keySelect.value;
            featureInput.value = ''; // Clear feature input
            resetDisplaysToLoading();

            if (currentSelectedKey && loadFirstFeature) {
                const featuresForThisKey = manifestData.features[currentSelectedKey];
                if (featuresForThisKey && featuresForThisKey.length > 0) {
                    const firstFeature = featuresForThisKey[0];
                    loadFeature(currentSelectedKey, firstFeature);
                } else {
                    featureInfo.textContent = `${currentSelectedKey} - No features available`;
                    showError(`No features available or defined for module ${currentSelectedKey}. Check features_to_save_scae.`, 'loadingIndicator');
                }
            } else if (currentSelectedKey) {
                 featureInfo.textContent = `${currentSelectedKey} - Select a feature`;
                 loadingIndicator.innerHTML = '<p>Please enter a feature number to view</p>';
            } else {
                featureInfo.textContent = 'No module selected';
            }
            updateNavigationButtons(null, null); // Reset nav buttons
        }

        function resetDisplaysToLoading(){
            featureInfo.textContent = 'No feature selected';
            contentDisplay.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');
            loadingIndicator.innerHTML = '<p>Select a module and feature.</p>';
            logitLensPanel.innerHTML = '<div class="loading">Select a feature.</div>';
            histogramPanel.innerHTML = '<div class="loading">Select a feature.</div>';
            connectionsPanel.innerHTML = '<div class="loading">Select a feature.</div>';
            prevFeatureBtn.disabled = true;
            nextFeatureBtn.disabled = true;
            currentFeatureData = null;
        }
        
        function handleGoToFeature() {
            const featureNumStr = featureInput.value.trim();
            if (!currentSelectedKey) {
                alert('Please select a module first.'); return;
            }
            if (!featureNumStr) {
                alert('Please enter a feature number.'); return;
            }
            const featureNum = parseInt(featureNumStr);
            const featuresForThisKey = manifestData.features[currentSelectedKey].map(f => parseInt(f));            
            if (!featuresForThisKey.includes(featureNum)){
                 alert(`Feature ${featureNum} is not in the list of available features for module ${currentSelectedKey}. Available: ${featuresForThisKey.join(', ')}`)
                 return;
            }
            loadFeature(currentSelectedKey, featureNumStr);
        }
        
        async function loadFeature(key, feature) {
            currentSelectedKey = key;
            currentSelectedFeature = feature;
            featureInfo.textContent = `${key} - Feature ${feature}`;
            featureInput.value = feature;
            
            loadingIndicator.classList.remove('hidden');
            loadingIndicator.innerHTML = '<p>Loading content...</p>';
            contentDisplay.classList.add('hidden');
            logitLensPanel.innerHTML = '<div class="loading">Loading logit lens...</div>';
            histogramPanel.innerHTML = '<div class="loading">Loading histogram...</div>';
            connectionsPanel.innerHTML = '<div class="loading">Loading connections...</div>';
            
            const newUrl = new URL(window.location.href);
            newUrl.searchParams.set('key', key);
            newUrl.searchParams.set('feature', feature);
            window.history.pushState({ key, feature }, '', newUrl.href);
            
            try {
                currentFeatureData = await loadFeatureData(key, feature);
                displayFeatureData();
                updateNavigationButtons(key, feature);
            } catch (error) {
                console.error('Error loading feature data:', error);
                showError(`Failed to load data for ${key} - Feature ${feature}. Error: ${error.message}`, 'loadingIndicator');
            }
        }
        
        function displayFeatureData() {
            if (!currentFeatureData) return;
            contentDisplay.innerHTML = currentFeatureData.tokenActivations || '<div class="error-panel">Token activation data missing.</div>';
            contentDisplay.classList.remove('hidden');
            loadingIndicator.classList.add('hidden');
            logitLensPanel.innerHTML = currentFeatureData.logitLens || '<div class="error-panel">Logit lens data missing.</div>';
            histogramPanel.innerHTML = currentFeatureData.histogram || '<div class="error-panel">Histogram data missing.</div>';
            connectionsPanel.innerHTML = currentFeatureData.connections || '<div class="error-panel">Connections data missing.</div>';
        }
        
        function showPreviousFeature() {
            if (!currentSelectedKey || currentSelectedFeature === null) return;
            const featuresForThisKey = manifestData.features[currentSelectedKey].map(f => parseInt(f));
            const currentIdx = featuresForThisKey.indexOf(parseInt(currentSelectedFeature));
            if (currentIdx > 0) {
                loadFeature(currentSelectedKey, featuresForThisKey[currentIdx - 1].toString());
            }
        }
        
        function showNextFeature() {
            if (!currentSelectedKey || currentSelectedFeature === null) return;
            const featuresForThisKey = manifestData.features[currentSelectedKey].map(f => parseInt(f));
            const currentIdx = featuresForThisKey.indexOf(parseInt(currentSelectedFeature));
            if (currentIdx < featuresForThisKey.length - 1) {
                loadFeature(currentSelectedKey, featuresForThisKey[currentIdx + 1].toString());
            }
        }
        
        function updateNavigationButtons(key, feature) {
            if (!key || feature === null || !manifestData.features[key]) {
                prevFeatureBtn.disabled = true;
                nextFeatureBtn.disabled = true;
                return;
            }
            const featuresForThisKey = manifestData.features[key].map(f => parseInt(f));
            const featureNum = parseInt(feature);
            const currentIdx = featuresForThisKey.indexOf(featureNum);
            prevFeatureBtn.disabled = currentIdx <= 0;
            nextFeatureBtn.disabled = currentIdx >= featuresForThisKey.length - 1;
        }
        
        window.addEventListener('popstate', function(event) {
            const params = getUrlParams();
            if (params.key && params.feature) {
                if (keySelect.value !== params.key || currentSelectedFeature !== params.feature) {
                    keySelect.value = params.key;
                    // handleKeyChange will be implicitly called by changing keySelect.value if there's a listener,
                    // but we need to ensure it doesn't auto-load the first feature if we have a specific one from URL.
                    handleKeyChange(false); // Don't auto-load first feature of this new key
                    loadFeature(params.key, params.feature);
                }
            } else {
                 resetDisplaysToLoading(); // Or load default if no params
                 if (manifestData.keys.length > 0) {
                    keySelect.value = manifestData.keys[0];
                    handleKeyChange(true); 
                 }
            }
        });
        
        window.addEventListener('DOMContentLoaded', initViewer);
    </script>
</body>
</html>'''
    
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w') as f_index:
        f_index.write(viewer_html_content)
    
    print(f"Enhanced SCAE feature viewer created at: {output_dir}")
    print(f"Open {index_path} in your browser to use the viewer")
    
    try:
        from IPython.display import HTML as IPHTML, display as ip_display # Alias to avoid conflict
        ip_display(IPHTML(f'<a href="{index_path}" target="_blank">Open SCAE Feature Viewer</a>'))
    except ImportError:
        pass # IPython not available
    
    return index_path 