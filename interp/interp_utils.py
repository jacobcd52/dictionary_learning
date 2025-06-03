from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
# from baukit import Trace
from tqdm import tqdm
import einops

# from baukit import TraceDict  
import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_streaming_dataset(tokenizer, dataset_name, max_length, batch_size, num_datapoints=None, num_cpu_cores=6):
    """Create a generator that streams batches from the dataset"""
    split = "train"
    split_text = f"{split}[:{num_datapoints}]" if num_datapoints else split
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split_text)
    current_batch = []
    
    def process_text(text):
        """Helper function to tokenize text"""
        return tokenizer(tokenizer.bos_token + text)['input_ids']
    
    for item in dataset:
        # Tokenize the text
        input_ids = process_text(item['text'])
        
        # Only keep sequences that are long enough
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]  # Truncate if necessary
            current_batch.append(torch.tensor(input_ids))
            
            # When we have enough samples, yield a batch
            if len(current_batch) == batch_size:
                # Pad the sequences in the batch to the same length
                # padded_batch = pad_sequence(current_batch, batch_first=True)
                yield torch.stack(current_batch)
                current_batch = []
    
    # Yield any remaining samples in the last batch
    if current_batch:
        yield torch.stack(current_batch)






import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from IPython.display import display, HTML

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
    
    # return HTML(html)
    return html

# Example usage with a histogram
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





from IPython.display import display, HTML
import torch

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
    top_text = [tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n") for tok in top_ind[:k]]
    bot_text = [tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n") for tok in bot_ind[:k]]
    
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
    
    # return HTML(html_template)
    return html_template



import numpy as np
from IPython.display import display, HTML
from einops import rearrange

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
        ratio = activation/max_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value
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
        if isinstance(array[0], int):
            array = [array]
    return array

def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None, model_type="causal", text_above_each_act=None):
    # text_spacing = "0.07em"
    text_spacing = "0.00em"
    toks = convert_token_array_to_list(toks)
    activations = convert_token_array_to_list(activations)
    # toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '↵') for t in tok] for tok in toks]
    toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp;').replace('\n', '\\n') for t in tok] for tok in toks]
    highlighted_text = []
    # Make background black
    # highlighted_text.append('<body style="background-color:black; color: white;">')
    highlighted_text.append("""
<body style="background-color: black; color: white;">
""")
    max_value = max([max(activ) for activ in activations])
    min_value = min([min(activ) for activ in activations])
    if(logit_diffs is not None and model_type != "reward_model"):
        logit_max_value = max([max(activ) for activ in logit_diffs])
        logit_min_value = min([min(activ) for activ in logit_diffs])

    # Add color bar
    highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
    if(logit_diffs is not None and model_type != "reward_model"):
        highlighted_text.append('<div style="margin-top: 0.1em;"></div>')
        highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value))
    
    highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
    for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
        if(text_above_each_act is not None):
            highlighted_text.append(f'<span>{text_above_each_act[seq_ind]}</span>')
        for act_ind, (a, t) in enumerate(zip(act, tok)):
            if(logit_diffs is not None and model_type != "reward_model"):
                highlighted_text.append('<div style="display: inline-block;">')
            text_color, background_color = value_to_color(a, max_value, min_value)
            highlighted_text.append(f'<span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{t.replace(" ", "&nbsp")}</span>')
            if(logit_diffs is not None and model_type != "reward_model"):
                logit_diffs_act = logit_diffs[seq_ind][act_ind]
                _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value, logit_min_value)
                highlighted_text.append(f'<div style="display: block; margin-right: {text_spacing}; height: 10px; background-color:{logit_background_color}; text-align: center;"></div></div>')
        if(logit_diffs is not None and model_type=="reward_model"):
            reward_change = logit_diffs[seq_ind].item()
            text_color, background_color = value_to_color(reward_change, 10, -10)
            highlighted_text.append(f'<br><span>Reward: </span><span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{reward_change:.2f}</span>')
        highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
        # highlighted_text.append('<br><br>')
    # highlighted_text.append('</body>')
    highlighted_text = ''.join(highlighted_text)
    return highlighted_text
def save_token_display(tokens, activations, tokenizer, path, save=True, logit_diffs=None, show=False, model_type="causal"):
    html = tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs, model_type=model_type)
    # if(save):
    #     imgkit.from_string(html, path)
    # if(show):
    return display(HTML(html))

def get_feature_indices(feature_activations, k=10, setting="max"):
    # Sort the features by activation, get the indices
    batch_size, seq_len = feature_activations.shape
    feature_activations = rearrange(feature_activations, 'b s -> (b s)')
    if setting=="max":
        found_indices = torch.argsort(feature_activations, descending=True)[:k]
    elif setting=="uniform":
        # min_value = torch.min(feature_activations)
        min_value = torch.min(feature_activations)
        max_value = torch.max(feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        bins = torch.bucketize(feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in torch.unique(bins):
            if(bin_idx==0): # Skip the first one. This is below the median
                continue
            # Get the indices corresponding to the current bin
            bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
            
            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
    else: # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    d_indices = found_indices // seq_len
    s_indices = found_indices % seq_len
    return d_indices, s_indices

def get_feature_datapoints(d_idx, seq_pos_idx, all_activations, all_tokens, tokenizer):
    full_activations = []
    partial_activations = []
    text_list = []
    full_text = []
    token_list = []
    full_token_list = []
    for md, s_ind in zip(d_idx, seq_pos_idx):
        md = int(md)
        s_ind = int(s_ind)
        # full_tok = torch.tensor(dataset[md]["input_ids"])
        
        full_tok = all_tokens[md]
        # [tokenizer.decode(t) for t in tokens[0]]

        full_text.append(tokenizer.decode(full_tok))
        tok = full_tok[:s_ind+1]
        # tok = dataset[md]["input_ids"][:s_ind+1]
        full_activations.append(all_activations[md].tolist())
        partial_activations.append(all_activations[md][:s_ind+1].tolist())
        text = tokenizer.decode(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list, partial_activations, full_activations





import json
import os
import re
import torch

def generate_enhanced_viewer(keys, features_to_save,  saved_feature_act_list, saved_token_list, tokenizer, model, aes, connections, connection_vals, output_dir="llm_feature_viewer", model_save_name="",CHUNK_SIZE = 100):
    """
    Generate an enhanced HTML viewer with separate data files for each feature.
    
    This creates:
    1. A main index.html viewer file
    2. Separate JavaScript data files for each feature to avoid CORS issues
    3. Connection data betweekzn features with links that support ctrl+click to open in new tabs
    
    Args:
        keys: List of model keys or identifiers
        features_to_save: Dictionary mapping keys to lists of feature indices
        saved_feature_act_list: Dictionary mapping keys to feature activation tensors
        saved_token_list: List of tokens for each example
        tokenizer: The tokenizer used to convert tokens to text
        model: The model object (for logit lens)
        suite: The suite object containing feature decoders
        connections: Dictionary of module-to-module connections (indices)
        connection_vals: Dictionary of module-to-module connection values
        output_dir: Directory where the viewer files will be saved
        model_save_name: Optional prefix for the output directory
        
    Returns:
        The path to the generated index.html file
    """
    
    # Number of examples to show per feature
    num_feature_datapoints = 10
    
    # Prepare output directory
    if model_save_name:
        output_dir = f"{model_save_name}_{output_dir}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create manifest data
    manifest = {
        "keys": keys,
        "features": {}
    }
    
    # Prepare string versions of features for manifest
    for key in keys:
        manifest["features"][key] = [str(f) for f in features_to_save[key]]
        # Create directory for each key
        key_dir = os.path.join(data_dir, key)
        os.makedirs(key_dir, exist_ok=True)
    # Function to get connections for a specific feature
    def get_feature_connections(source_key, source_feature_idx):
        feature_connections = []
        
        # Check if this key is in connections
        if source_key not in connections:
            return feature_connections
            
        # Check connections to all other modules
        for target_key in connections[source_key]:
            # Get connection indices tensor for this module pair
            connection_tensor = connections[source_key][target_key]
            
            # Get connection values tensor for this module pair
            value_tensor = connection_vals[source_key][target_key][source_feature_idx]
            
            # Ensure the feature index is valid
            if source_feature_idx >= connection_tensor.shape[0]:
                continue
                
            # Get row for this feature's connections
            connection_row = connection_tensor[source_feature_idx]
            
            # Find non-zero connections (where connected)
            # e.g. [0,1,4,6] for idx over the top-c connections (so 0-c indexed)
            non_zero_indices = (connection_row > 0).nonzero()[:, 0]
            
            # Handle various tensor dimensions
            if non_zero_indices.dim() == 0 and non_zero_indices.nelement() > 0:
                # Single non-zero value case
                non_zero_indices = [non_zero_indices.item()]
            elif non_zero_indices.nelement() > 0:
                non_zero_indices = non_zero_indices.tolist()
            else:
                non_zero_indices = []
                
            # For each connected feature, get the connection value
            for nz_idx in non_zero_indices:
                # Get the connection value from value_tensor
                target_feature_idx = int(connection_row[nz_idx].item())
                connection_value = float(value_tensor[nz_idx].item())
                
                # Only add if target feature is in features_to_save
                # if target_feature_idx in features_to_save.get(target_key, []):
                feature_connections.append({
                    "target_key": target_key,
                    "target_feature": target_feature_idx,
                    "value": connection_value
                })
        
        # Sort connections by strength (absolute value) in descending order
        feature_connections.sort(key=lambda x: abs(x["value"]), reverse=True)
        return feature_connections
    
    # Generate HTML for feature connections with URL links
    def generate_connections_html(connections_list):
        if not connections_list:
            return "<div class='no-connections'>No significant connections found</div>"
            
        html = ["<div class='connections-container'>",
                "<h3>Feature Connections</h3>",
                "<table class='connections-table'>",
                "<tr><th>Connected Feature</th><th>Connection Strength</th></tr>"]
                
        for conn in connections_list:
            target_key = conn["target_key"]
            target_feature = conn["target_feature"]
            conn_value = conn["value"]
            
            # Determine CSS class based on connection value
            if conn_value > 0:
                value_class = "positive-connection"
            else:
                value_class = "negative-connection"
                
            # Create link to target feature with URL parameters
            # This supports ctrl+click to open in new tab
            html.append(f"<tr>")
            html.append(f"<td><a href='index.html?key={target_key}&feature={target_feature}' class='feature-link'>{target_key} - Feature {target_feature}</a></td>")
            html.append(f"<td class='{value_class}'>{conn_value:.4f}</td>")
            html.append(f"</tr>")
            
        html.append("</table></div>")
        return "\n".join(html)
    
    
    
#     # Function to safely save feature chunk data (preserving existing data)
#     def save_feature_chunk(key, chunk_idx, new_chunk_data, data_dir):
#         chunk_path = os.path.join(data_dir, key, f"chunk_{chunk_idx}.js")
#         existing_data = {}
        
#         # Check if chunk file already exists
#         if os.path.exists(chunk_path):
#             try:
#                 # Read existing chunk file
#                 with open(chunk_path, 'r') as f:
#                     content = f.read()
#                     # Extract JSON data between markers
#                     data_match = re.search(r'window\.featureChunk\s*=\s*(\{.*?\});', content, re.DOTALL)
#                     if data_match:
#                         existing_data = json.loads(data_match.group(1))
#             except Exception as e:
#                 print(f"Warning: Could not read existing chunk file: {e}")
        
#         # Merge existing data with new data
#         merged_data = {**existing_data, **new_chunk_data}
        
#         # Write updated chunk file
#         chunk_js = f'''// Features chunk {chunk_idx} data for {key}
# window.featureChunk = {json.dumps(merged_data)};
# '''
#         with open(chunk_path, 'w') as f:
#             f.write(chunk_js)
    
    def save_feature_chunk(key, chunk_idx, new_chunk_data, data_dir):
        chunk_path = os.path.join(data_dir, key, f"chunk_{chunk_idx}.js")
        existing_data = {}
        
        # Check if chunk file already exists
        if os.path.exists(chunk_path):
            try:
                # Read existing chunk file
                with open(chunk_path, 'r') as f:
                    content = f.read()
                    
                # Use bracket counting approach instead of regex
                start_marker = "window.featureChunk = "
                start_pos = content.find(start_marker)
                if start_pos != -1:
                    start_pos += len(start_marker)
                    # Find the end of the JSON object by tracking braces
                    open_braces = 0
                    for i in range(start_pos, len(content)):
                        if content[i] == '{':
                            open_braces += 1
                        elif content[i] == '}':
                            open_braces -= 1
                            if open_braces == 0 and i + 1 < len(content) and content[i+1:].lstrip().startswith(';'):
                                # Found the closing brace, extract the JSON
                                json_str = content[start_pos-1:i+1]  # Include opening brace
                                try:
                                    existing_data = json.loads(json_str)
                                    break
                                except json.JSONDecodeError as e:
                                    print(f"Warning: Invalid JSON in chunk file: {e}")
                                    break
            except Exception as e:
                print(f"Warning: Could not read existing chunk file: {e}")
        
        # Merge existing data with new data
        merged_data = {**existing_data, **new_chunk_data}
        
        # Write updated chunk file
        chunk_js = f'''// Features chunk {chunk_idx} data for {key}
    window.featureChunk = {json.dumps(merged_data)};
    '''
        with open(chunk_path, 'w') as f:
            f.write(chunk_js)

    # Generate chunked data files for each key
    for key in keys:
        features_for_this_key = features_to_save[key]
        
        # Group features into chunks
        feature_chunks = {}
        
        for feature_local_idx, feature_global_idx in enumerate(features_for_this_key):
            # Cast feature to int if it's a string
            feature_global_idx = int(feature_global_idx) if isinstance(feature_global_idx, str) else feature_global_idx
            feature_str = str(feature_global_idx)
            
            # Calculate chunk index
            chunk_idx = feature_global_idx // CHUNK_SIZE
            
            # Initialize chunk if not exists
            if chunk_idx not in feature_chunks:
                feature_chunks[chunk_idx] = {}
            
            # Initialize content container for this feature
            feature_data = {
                "tokenActivations": "",
                "logitLens": "",
                "histogram": "",
                "connections": ""
            }
            
            # Get feature activations for this feature
            feature_activations = saved_feature_act_list[key][..., feature_local_idx]
            
            # Get indices of examples with highest activations
            d_idx, seq_idx = get_feature_indices(feature_activations, k=num_feature_datapoints, setting="max")
            
            # Get data for these examples
            text_list, full_text, token_list, full_token_list, partial_activations, full_activations = get_feature_datapoints(
                d_idx, seq_idx, feature_activations, saved_token_list, tokenizer
            )
            
            # Generate token activations HTML
            token_html = tokens_and_activations_to_html(token_list, partial_activations, tokenizer)
            feature_data["tokenActivations"] = token_html
            
            # Generate logit lens HTML
            try:
                feature_decoder = aes[key].decoder.weight[:, feature_global_idx]
                unembd = model.W_U
                final_ln = model.ln_final
                logit_lens = final_ln(feature_decoder) @ unembd
                top_val, top_ind = torch.topk(logit_lens, k=10, dim=-1)
                bot_val, bot_ind = torch.topk(logit_lens, k=10, dim=-1, largest=False)
                
                # Create logit lens HTML
                logit_lens_html = create_logit_lens_html(top_ind, top_val, bot_ind, bot_val, tokenizer)
                feature_data["logitLens"] = logit_lens_html
            except Exception as e:
                print(f"Error in logit lens: {e}")
                feature_data["logitLens"] = f"<div class='error-panel'>Logit lens visualization unavailable: {str(e)}</div>"
            
            # Generate histogram HTML
            try:
                nz_feature_act = feature_activations[feature_activations != 0]
                frequency = nz_feature_act.numel() / feature_activations.numel()
                
                # Create histogram HTML
                hist_html = create_histogram_html(nz_feature_act.numpy(), 
                                               title=f"Activation Frequency {frequency*100:.2f}%")
                feature_data["histogram"] = hist_html
            except Exception as e:
                print(f"Error in hist: {e}")

                feature_data["histogram"] = f"<div class='error-panel'>Histogram visualization unavailable: {str(e)}</div>"
            
            # Get feature connections
            feature_connections = get_feature_connections(key, feature_global_idx)
            
            # Generate connections HTML
            connections_html = generate_connections_html(feature_connections)
            feature_data["connections"] = connections_html
            
            # Add feature data to chunk
            feature_chunks[chunk_idx][feature_str] = {
                "tokenActivations": feature_data["tokenActivations"],
                "logitLens": feature_data["logitLens"],
                "histogram": feature_data["histogram"],
                "connections": connections_html
            }
        
        # Save each chunk as a separate JS file, preserving existing data
        for chunk_idx, chunk_data in feature_chunks.items():
            save_feature_chunk(key, chunk_idx, chunk_data, data_dir)
    
    # Save manifest as a JavaScript file to avoid CORS issues
    manifest_js = f'''// Feature manifest data
const manifestData = {json.dumps(manifest)};
// Configuration
const CHUNK_SIZE = {CHUNK_SIZE};
'''
    
    manifest_path = os.path.join(output_dir, "manifest.js")
    with open(manifest_path, 'w') as f:
        f.write(manifest_js)
    
    # Create main viewer HTML file
    viewer_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Viewer</title>
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
            min-height: 225px;
        }
        
        /* Content section */
        .token-activations-container {
            background-color: black;
            color: white;
            min-height: 400px;
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
            height: 300px;
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
            white-space: pre-wrap;
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
        
        .title {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        
        /* Fix for histogram images */
        .histogram-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .histogram-container img {
            max-width: 100%;
            max-height: 280px;
            width: auto;
            height: auto;
            object-fit: contain;
        }
        
        /* Connections styles */
        .connections-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }
        
        .connections-container h3 {
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 18px;
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
    <!-- Load manifest data -->
    <script src="manifest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Feature Viewer</h1>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="keySelect">Select Model/Key:</label>
                <select id="keySelect">
                    <option value="" disabled selected>Choose a model/key</option>
                    <!-- Options will be populated by JavaScript -->
                </select>
            </div>
            <div class="control-group">
                <label for="featureInput">Go to Feature #:</label>
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
        
        <!-- Upper panels for logit lens and histogram -->
        <table class="upper-panels-table">
            <tr>
                <!-- Logit lens panel -->
                <td class="panel-cell">
                    <div class="panel">
                        <div class="panel-header">Logit Lens</div>
                        <div id="logitLensPanel" class="panel-content">
                            <div class="loading">Select a feature to view logit lens</div>
                        </div>
                    </div>
                </td>
                
                <!-- Histogram panel -->
                <td class="panel-cell">
                    <div class="panel">
                        <div class="panel-header">Activation Histogram</div>
                        <div id="histogramPanel" class="panel-content">
                            <div class="loading">Select a feature to view histogram</div>
                        </div>
                    </div>
                </td>
            </tr>
        </table>
        
        <!-- Token activations section -->
        <div class="token-activations-container">
            <div id="loadingIndicator" class="loading">
                <p>Select a model/key and feature to view token activations</p>
            </div>
            <div id="contentDisplay" class="content hidden"></div>
        </div>
        
        <!-- Connections section -->
        <div class="connections-section">
            <div id="connectionsPanel"></div>
        </div>
        
        <div class="footer">
            <p>Feature Viewer | Features Explorer</p>
        </div>
    </div>
    
    <script>
        // ===== DOM ELEMENTS =====
        // Get references to DOM elements
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
     
        // Current feature data
        let currentFeatureData = null;
        
        // ===== URL PARAMETER HANDLING =====
        // Parse URL parameters
        function getUrlParams() {
            const params = {};
            const searchParams = new URLSearchParams(window.location.search);
            
            for (const [key, value] of searchParams) {
                params[key] = value;
            }
            
            return params;
        }
        
        // ===== DATA LOADING =====
        // Calculate which chunk contains a feature
        function getChunkIndex(feature) {
            // const CHUNK_SIZE = 100;
            return Math.floor(parseInt(feature) / CHUNK_SIZE);
        }
        
        // Load feature data from chunk (avoids CORS issues)
        function loadFeatureData(key, feature) {
            return new Promise((resolve, reject) => {
                // Set up a timeout for loading
                const timeout = setTimeout(() => {
                    reject(new Error('Timeout loading feature data'));
                }, 10000); // 10 second timeout
                
                // Calculate which chunk contains this feature
                const chunkIndex = getChunkIndex(feature);
                
                // Remove existing script if any
                const existingScript = document.getElementById('featureChunkScript');
                if (existingScript) {
                    document.head.removeChild(existingScript);
                }
                
                // Reset window.featureChunk
                window.featureChunk = null;
                
                // Create new script element
                const script = document.createElement('script');
                script.id = 'featureChunkScript';
                script.src = `data/${key}/chunk_${chunkIndex}.js`;
                
                script.onload = () => {
                    clearTimeout(timeout);
                    if (window.featureChunk && window.featureChunk[feature]) {
                        resolve(window.featureChunk[feature]);
                    } else {
                        reject(new Error(`Feature ${feature} not found in chunk ${chunkIndex}`));
                    }
                };
                
                script.onerror = () => {
                    clearTimeout(timeout);
                    reject(new Error(`Failed to load chunk ${chunkIndex} for ${key}`));
                };
                
                // Add script to document
                document.head.appendChild(script);
            });
        }
        
        // ===== INITIALIZATION =====
        // Initialize the viewer
        function initViewer() {
            // Check if manifest data is available
            if (!manifestData || !manifestData.keys || !manifestData.features) {
                showError('Failed to load manifest data. Please refresh the page.');
                return;
            }
            
            // Populate the keys dropdown
            manifestData.keys.forEach(key => {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = key;
                keySelect.appendChild(option);
            });
            
            // Set up event listeners
            keySelect.addEventListener('change', handleKeyChange);
            goToFeatureBtn.addEventListener('click', handleGoToFeature);
            featureInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    handleGoToFeature();
                }
            });
            prevFeatureBtn.addEventListener('click', showPreviousFeature);
            nextFeatureBtn.addEventListener('click', showNextFeature);
            
            // Check URL parameters for direct feature loading
            const params = getUrlParams();
            const urlKey = params.key;
            const urlFeature = params.feature;
            
            if (urlKey && urlFeature && 
                manifestData.keys.includes(urlKey) && 
                manifestData.features[urlKey]) {
                
                // Select the key from URL
                keySelect.value = urlKey;
                handleKeyChange();
                
                // Allow a small delay for the feature dropdown to update
                setTimeout(() => {
                    // Try to load the feature directly
                    loadFeature(urlKey, urlFeature);
                }, 100);
            }
            // If no URL params, select first key and feature if available
            else if (manifestData.keys.length > 0) {
                // Select first key
                keySelect.value = manifestData.keys[0];
                handleKeyChange();
                
                                // New code:
                if (manifestData.features[manifestData.keys[0]] && manifestData.features[manifestData.keys[0]].length > 0) {
                    setTimeout(() => {
                        const firstFeature = manifestData.features[manifestData.keys[0]][0];
                        featureInput.value = firstFeature;
                        loadFeature(manifestData.keys[0], firstFeature);
                    }, 100);
                }
            }
        }
        
        // Show error message
        function showError(message) {
            loadingIndicator.classList.remove('hidden');
            loadingIndicator.innerHTML = `<p class="error-panel">${message}</p>`;
            contentDisplay.classList.add('hidden');
        }
        
        // ===== EVENT HANDLERS =====
        // Handle key selection change
        function handleKeyChange() {
            const selectedKey = keySelect.value;
            
            if (selectedKey) {
                // Clear content display
                contentDisplay.classList.add('hidden');
                loadingIndicator.classList.remove('hidden');
                loadingIndicator.innerHTML = '<p>Please enter a feature number to view</p>';
                
                // Reset feature info
                featureInfo.textContent = 'No feature selected';
                
                // Clear panels
                logitLensPanel.innerHTML = '<div class="loading">Enter a feature number to view logit lens</div>';
                histogramPanel.innerHTML = '<div class="loading">Enter a feature number to view histogram</div>';
                connectionsPanel.innerHTML = '';
                
                // Disable navigation buttons
                prevFeatureBtn.disabled = true;
                nextFeatureBtn.disabled = true;
                
                // Reset current feature data
                currentFeatureData = null;
                
                // Load first feature by default
                if (manifestData.features[selectedKey] && manifestData.features[selectedKey].length > 0) {
                    setTimeout(() => {
                        const firstFeature = manifestData.features[selectedKey][0];
                        featureInput.value = firstFeature;
                        loadFeature(selectedKey, firstFeature);
                    }, 100);
                }
            }
        }
        
        // Handle direct feature number input
        function handleGoToFeature() {
            const selectedKey = keySelect.value;
            const featureNum = featureInput.value.trim();
            
            if (!selectedKey) {
                alert('Please select a model/key first');
                return;
            }
            
            if (!featureNum) {
                alert('Please enter a feature number');
                return;
            }
            
            loadFeature(selectedKey, featureNum);
        }
        
        // Load a specific feature by key and feature number
        async function loadFeature(key, feature) {
            // Update feature info
            featureInfo.textContent = `${key} - Feature ${feature}`;
            featureInput.value = feature; // Update input to match
            featureInput.value = feature; // Update input to match
            
            // Show loading indicators
            loadingIndicator.classList.remove('hidden');
            loadingIndicator.innerHTML = '<p>Loading content...</p>';
            contentDisplay.classList.add('hidden');
            
            logitLensPanel.innerHTML = '<div class="loading">Loading logit lens...</div>';
            histogramPanel.innerHTML = '<div class="loading">Loading histogram...</div>';
            connectionsPanel.innerHTML = '<div class="loading">Loading connections...</div>';
            
            // Update URL without reloading page
            const newUrl = new URL(window.location.href);
            newUrl.searchParams.set('key', key);
            newUrl.searchParams.set('feature', feature);
            window.history.pushState({ key, feature }, '', newUrl.href);
            
            try {
                // Load feature data
                currentFeatureData = await loadFeatureData(key, feature);
                
                // Display the data
                displayFeatureData();
                
                // Disable navigation buttons for initial state
                const allFeatures = manifestData.features[key].map(f => parseInt(f));
                // const minFeature = Math.min(...allFeatures);
                const maxFeature = Math.max(...allFeatures);
                
                prevFeatureBtn.disabled = parseInt(feature) <= 0;
                nextFeatureBtn.disabled = parseInt(feature) >= maxFeature;
            } catch (error) {
                console.error('Error loading feature data:', error);
                showError(`Failed to load data for ${key} - Feature ${feature}`);
            }
        }
        
        
        // Display the loaded feature data
        function displayFeatureData() {
            if (currentFeatureData) {
                // Display token activations
                contentDisplay.innerHTML = currentFeatureData.tokenActivations;
                contentDisplay.classList.remove('hidden');
                loadingIndicator.classList.add('hidden');
                
                // Display logit lens
                logitLensPanel.innerHTML = currentFeatureData.logitLens;
                
                // Display histogram
                histogramPanel.innerHTML = currentFeatureData.histogram;
                
                // Display connections
                connectionsPanel.innerHTML = currentFeatureData.connections;
            }
        }
        
        // ===== NAVIGATION CONTROLS =====
        // Show previous feature
        function showPreviousFeature() {
            const selectedKey = keySelect.value;
            const currentFeature = featureInfo.textContent.split(' - Feature ')[1];
            
            if (selectedKey && currentFeature) {
                const prevFeature = parseInt(currentFeature) - 1;
                if (prevFeature >= 0) {
                    loadFeature(selectedKey, prevFeature);
                }
            }
        }
        
        // Show next feature
        function showNextFeature() {
            const selectedKey = keySelect.value;
            const currentFeature = featureInfo.textContent.split(' - Feature ')[1];
            
            if (selectedKey && currentFeature) {
                const nextFeature = parseInt(currentFeature) + 1;
                const maxFeature = Math.max(...manifestData.features[selectedKey].map(f => parseInt(f)));
                
                if (nextFeature <= maxFeature) {
                    loadFeature(selectedKey, nextFeature);
                }
            }
        }
        
        // Update navigation button states
        function updateNavigationButtons(key, feature) {
            if (!key || !feature) return;
            
            const featureNum = parseInt(feature);
            const allFeatures = manifestData.features[key].map(f => parseInt(f));
            const minFeature = Math.min(...allFeatures);
            const maxFeature = Math.max(...allFeatures);
            
            prevFeatureBtn.disabled = featureNum <= minFeature;
            nextFeatureBtn.disabled = featureNum >= maxFeature;
        }
        
        // Handle browser back/forward navigation
        window.addEventListener('popstate', function(event) {
            const params = getUrlParams();
            
            if (params.key && params.feature) {
                // Only update if values actually changed
                if (keySelect.value !== params.key) {
                    keySelect.value = params.key;
                    handleKeyChange();
                    
                    setTimeout(() => {
                        loadFeature(params.key, params.feature);
                    }, 100);
                } else {
                    loadFeature(params.key, params.feature);
                }
            }
        });
        
        // Initialize the viewer when the page loads
        window.addEventListener('DOMContentLoaded', () => {
            initViewer();
            
            // After initialization, if URL doesn't have params, open first feature of first key
            setTimeout(() => {
                const params = getUrlParams();
                if (!params.key && !params.feature && manifestData.keys.length > 0) {
                    const firstKey = manifestData.keys[0];
                    keySelect.value = firstKey;
                    handleKeyChange();
                }
            }, 200);
        });
    </script>
</body>
</html>'''
    
    # Save main viewer HTML file
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(viewer_html)
    
    # print(f"Enhanced token activation viewer created at: {output_dir}")
    # print(f"Open {index_path} in your browser to use the viewer")
    
    # If running in notebook, provide a clickable link
    # try:
    #     from IPython.display import HTML, display
    #     display(HTML(f'<a href="{index_path}" target="_blank">Open Enhanced Token Activation Viewer</a>'))
    # except:
    #     pass
    
    return index_path








