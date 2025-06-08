import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from dictionary_learning.mask_scae import SCAESuite, MergedSCAESuite, SubmoduleName, AutoEncoderTopK, CrosscoderTopK # Added AE an CCK
from transformer_lens import HookedTransformer
import os
import gc
import shutil
from typing import List, Dict, Tuple, Optional, Union, Any
from IPython.display import display, HTML
import numpy as np
import html # Added import
import matplotlib.pyplot as plt
# import matplotlib.colors # No longer needed for token display
# import matplotlib.pyplot as plt # No longer needed for token display, keep for future hist if any


# --- 1. Data Loading ---
def load_tokenized_dataset(
    dataset_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    num_samples: int = 1000,
    split: str = "train",
    text_column: str = "text",
    streaming: bool = False,
    data_files: Optional[Union[str, List[str]]] = None,
) -> Dataset:
    """
    Loads a dataset, tokenizes it, and formats it for processing.
    The tokenizer will be used with return_tensors="pt".
    """
    if streaming and data_files: # Streaming from local files
         dataset = load_dataset("json", data_files=data_files, split=split, streaming=streaming)
    elif streaming: # Streaming from HF
        dataset = load_dataset(dataset_name_or_path, split=split, streaming=streaming)
    elif data_files: # Non-streaming from local files
        dataset = load_dataset("json", data_files=data_files, split=f"{split}[:{num_samples}]")
    else: # Non-streaming from HF
        dataset = load_dataset(dataset_name_or_path, split=f"{split}[:{num_samples}]")

    def tokenize_and_chunk(examples): # examples[text_column] is List[str]
        list_of_1d_token_tensors = []
        for text in examples[text_column]:
            # Tokenize each text individually to avoid cross-text padding from tokenizer
            token_ids_tensor = tokenizer(text, return_attention_mask=False, return_tensors="pt")["input_ids"]
            
            # Squeeze if tokenizer returns (1, N) for single text
            if token_ids_tensor.ndim == 2 and token_ids_tensor.shape[0] == 1:
                token_ids_tensor = token_ids_tensor.squeeze(0)
            
            # Ensure it's 1D; skip if empty (e.g., from empty string)
            if token_ids_tensor.ndim != 1:
                if token_ids_tensor.numel() == 0:
                    continue 
                else:
                    raise ValueError(f"Tokenizer produced unexpected tensor shape {token_ids_tensor.shape} for single text input: '{text[:100]}...'")
            list_of_1d_token_tensors.append(token_ids_tensor)

        if not list_of_1d_token_tensors: # All texts in batch were empty or resulted in no tokens
            return {"input_ids": []} # Return empty list of chunks

        # Concatenate all token tensors. Assuming tokenizer outputs CPU tensors.
        # If they can be on other devices, ensure they are moved to a consistent device (e.g., CPU) before cat.
        try:
            concatenated_tokens_tensor = torch.cat(list_of_1d_token_tensors, dim=0)
        except Exception as e:
            # For debugging if cat fails (e.g. list_of_1d_token_tensors is empty when it shouldn't be, or tensors are on different devices)
            print(f"Error during torch.cat of token tensors: {e}")
            # for i, t in enumerate(list_of_1d_token_tensors):
            #     print(f"Tensor {i}: shape {t.shape}, device {t.device}, dtype {t.dtype}")
            raise

        current_device = concatenated_tokens_tensor.device
        
        current_length = concatenated_tokens_tensor.size(0)
        padding_length = (seq_len - (current_length % seq_len)) % seq_len

        if padding_length > 0:
            if tokenizer.pad_token_id is None:
                # Attempt to use eos_token_id if pad_token_id is None, common for some models like GPT-2
                if tokenizer.eos_token_id is not None:
                    print(f"Warning: tokenizer.pad_token_id is None. Using tokenizer.eos_token_id ({tokenizer.eos_token_id}) for padding.")
                    pad_token_id_to_use = int(tokenizer.eos_token_id)
                else:
                    raise ValueError("Tokenizer does not have a pad_token_id or eos_token_id set, which is required for padding.")
            else:
                pad_token_id_to_use = int(tokenizer.pad_token_id)
            
            pad_values = torch.full((padding_length,), pad_token_id_to_use,
                                    dtype=concatenated_tokens_tensor.dtype, device=current_device)
            concatenated_tokens_tensor = torch.cat([concatenated_tokens_tensor, pad_values], dim=0)

        num_chunks = concatenated_tokens_tensor.size(0) // seq_len
        if num_chunks == 0:
            return {"input_ids": []} 

        chunked_tokens_2d_tensor = concatenated_tokens_tensor.reshape(num_chunks, seq_len)
        # map expects the function to return a dict of lists, where each element in the list is a sample
        list_of_chunk_tensors = [chunk for chunk in chunked_tokens_2d_tensor] 
        
        return {"input_ids": list_of_chunk_tensors}

    if streaming:
        tokenized_dataset = dataset.map(
            tokenize_and_chunk,
            batched=True,
            remove_columns=[text_column] 
        )
    else: # Not streaming
        tokenized_dataset = dataset.map(
            tokenize_and_chunk,
            batched=True,
            remove_columns=[text_column], 
            num_proc=max(1, os.cpu_count() // 2)
        )
        # The number of samples is already limited by the initial load_dataset split.
        # No further sub-selection of chunks based on original num_samples here.
        
        # set_format ensures that when an item is accessed, 'input_ids' is a tensor.
        # With the new tokenize_and_chunk, it's already a list of tensors,
        # so this will ensure individual items are tensors if not already.
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    return tokenized_dataset


# --- 2. Activation Collection & Saving ---
def collect_activations_and_tokens(
    suite: SCAESuite,
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase, # Added tokenizer for consistency, though not directly used in merged_suite.forward
    dataset: Dataset,
    device: Union[str, torch.device],
    output_dir: str,
    run_mode_sparse: bool,
    batch_size: int = 8,
    temperature: float = 1.0, # For learnable masks if used
    max_batches_to_process: Optional[int] = None,
):
    """
    Runs the SCAESuite on the provided data, collects sparse feature activations 
    and tokens, and saves them to disk.

    Args:
        suite: The SCAESuite object.
        model: The HookedTransformer model associated with the suite.
        tokenizer: The tokenizer.
        dataset: A tokenized Hugging Face Dataset (each item is {"input_ids": tensor}).
        device: Torch device.
        output_dir: Directory to save activations and tokens.
        run_mode_sparse: Boolean, if True, runs suite in sparse connection mode.
        batch_size: Batch size for processing.
        temperature: Temperature for learnable masks.
        max_batches_to_process: Optional limit on the number of batches.
    """
    
    mode_str = "sparse_true" if run_mode_sparse else "sparse_false"
    current_output_path = os.path.join(output_dir, mode_str)
    if os.path.exists(current_output_path):
        print(f"Warning: Output path {current_output_path} already exists. Clearing it.")
        shutil.rmtree(current_output_path)
    os.makedirs(current_output_path, exist_ok=True)

    merged_suite = MergedSCAESuite(model, suite).to(device)
    merged_suite.eval()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    print(f"Starting activation collection. Mode: {'Sparse' if run_mode_sparse else 'Non-sparse'}")
    print(f"Saving to: {current_output_path}")

    for batch_idx, batch_data in enumerate(data_loader):
        if max_batches_to_process is not None and batch_idx >= max_batches_to_process:
            print(f"Reached max_batches_to_process: {max_batches_to_process}. Stopping.")
            break

        input_ids_from_batch = batch_data["input_ids"] # Expected: Tensor from DataLoader

        if not isinstance(input_ids_from_batch, torch.Tensor):
            # This should ideally not be hit if data pipeline is correct
            raise TypeError(
                f"batch_data['input_ids'] (type: {type(input_ids_from_batch)}) is not a torch.Tensor as expected. "
                "There might be an issue in the tokenize_and_chunk or DataLoader's collate_fn."
            )
        
        input_ids = input_ids_from_batch.to(device) # Should be (batch_size, seq_len)

        # Ensure input_ids has a batch dimension. DataLoader usually ensures this.
        if input_ids.ndim == 1:
            # This might occur if dataset somehow yields 1D tensors and batch_size=1
            # and collate_fn doesn't add batch dim. merged_suite needs batch dim.
            # Or if somehow input_ids_from_batch was a single 1D tensor before .to(device)
            print(f"Warning: input_ids had 1 dimension (shape {input_ids.shape}). Unsqueezing dim 0 to create batch dimension.")
            input_ids = input_ids.unsqueeze(0)
        
        # Check for empty batches or batches of empty sequences
        if input_ids.shape[0] == 0 : # Dataloader yielded an empty batch (e.g. batch_size > 0 but no data)
            if batch_size > 0: # We expected a batch
                 print(f"Skipping empty batch {batch_idx} (input_ids shape: {input_ids.shape})")
            continue
        
        # This check might be relevant if seq_len could be 0, but our chunking logic ensures seq_len > 0 for chunks.
        # So, if num_chunks > 0, then shape[1] will be seq_len.
        # The main concern is if input_ids.shape[0] (batch dimension from dataloader) is 0.
        # If input_ids.numel() == 0 but input_ids.shape[0] > 0, it means batch_size > 0 but seq_len = 0. (e.g. shape [8,0])
        # This should not happen due to chunking logic.

        with torch.no_grad():
            reconstructions, pruned_features, cache = merged_suite(
                input_ids,
                temperature=temperature,
                runtime_use_sparse_connections_override=run_mode_sparse
            )

        batch_output_dir = os.path.join(current_output_path, f"batch_{batch_idx:05d}")
        os.makedirs(batch_output_dir, exist_ok=True)

        # Save tokens
        torch.save(input_ids.cpu(), os.path.join(batch_output_dir, "tokens.pt"))

        # Save sparse activations
        for module_name, activation_tensor in pruned_features.items():
            # activation_tensor is the scatter_buffer (batch, seq, n_features_module)
            # Find non-zero elements
            non_zero_indices = activation_tensor.nonzero(as_tuple=True) # (batch_indices, seq_indices, feat_indices)
            non_zero_values = activation_tensor[non_zero_indices]
            
            save_path = os.path.join(batch_output_dir, f"activations_{module_name}.pt")
            torch.save({
                'indices': tuple(idx.cpu() for idx in non_zero_indices),
                'values': non_zero_values.cpu(),
                'shape': activation_tensor.shape
            }, save_path)
            del activation_tensor, non_zero_indices, non_zero_values

        del reconstructions, pruned_features, cache, input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if batch_idx % 10 == 0:
            print(f"Processed and saved batch {batch_idx}")

    print(f"Finished activation collection for mode: {'Sparse' if run_mode_sparse else 'Non-sparse'}.")


# --- Helper functions from interp_utils.py (for styling and logit lens) ---

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
        ratio = activation/max_value if max_value != 0 else 1 # Avoid division by zero
        ratio = min(1, max(0, ratio)) # Clamp ratio to [0,1]
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value if min_value != 0 else 1 # Avoid division by zero
        ratio = min(1, max(0, ratio)) # Clamp ratio to [0,1]
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
    else:
        text_color = "0,0,0"
        background_color = f'rgba({white},{white},{white},1)'
    return text_color, background_color

def create_logit_lens_html(top_ind, top_val, bot_ind, bot_val, tokenizer: PreTrainedTokenizerBase, k=10):
    """
    Create an HTML display of top and bottom tokens with their values.
    Adapted from interp_utils.py
    """
    
    # Decode tokens
    def format_token_for_logit_lens(token_id_list, tokenizer_ref):
        # tokenizer.decode can take a list of a single id
        decoded = tokenizer_ref.decode(token_id_list)
        # Escape HTML special characters first to prevent misinterpretation
        escaped = html.escape(decoded)
        # Then, replace spaces with underscores and actual newlines with '\n' string
        return escaped.replace(" ", "_").replace("\n", "\\n")

    top_text = [format_token_for_logit_lens([tok], tokenizer) for tok in top_ind[:k]]
    bot_text = [format_token_for_logit_lens([tok], tokenizer) for tok in bot_ind[:k]]
    
    # Create HTML template with direct background color attributes
    html_template = """
    <div style="margin-top: 20px;">
        <style>
            .logit-lens-table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
                color: #333; /* Default text color for table */
            }
            .logit-lens-table th, .logit-lens-table td {
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .logit-lens-table th {
                background-color: #f2f2f2;
            }
            .logit-lens-title {
                font-size: 18px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 10px;
                color: #ccc; /* Light text color for title if on dark background */
            }
        </style>
        
        <div class="logit-lens-title">Logit Lens</div>
        
        <table class="logit-lens-table">
            <tr>
                <th>Top Token</th>
                <th>Value</th>
                <th>Bottom Token</th>
                <th>Value</th>
            </tr>
    """
    
    for i in range(k):
        html_template += f"""
        <tr>
            <td>{top_text[i]}</td>
            <td><span style="background-color: #e6f7ff; color: #005f80; padding: 2px 4px; border-radius: 3px; display: inline-block;"><b>{top_val[i].item():.3f}</b></span></td>
            <td>{bot_text[i]}</td>
            <td><span style="background-color: #ffe6e6; color: #800000; padding: 2px 4px; border-radius: 3px; display: inline-block;"><b>{bot_val[i].item():.3f}</b></span></td>
        </tr>
        """
    
    html_template += "</table></div>"
    return html_template


# --- 3. Feature Dashboard ---

def _get_context_html(
    tokens: List[int],
    activations_in_context: List[float], 
    tokenizer: PreTrainedTokenizerBase,
    min_act_for_norm: float,
    max_act_for_norm: float,
    positive_threshold: float = 0.01, # Added thresholds
    negative_threshold: float = 0.01
) -> str:
    """Generates HTML for a single context window with highlighted tokens, styled like interp_utils.py."""
    html_parts = []
    # Token string processing similar to interp_utils.py
    # However, interp_utils.py decodes token by token. Here we get a list of token_ids.
    # We should decode them one by one to correctly handle special tokens and spaces.
    
    decoded_tokens = []
    for token_id in tokens:
        # Use decode for single tokens to get the string representation, including prefixes like 'Ġ'
        # or handle special tokens.
        
        # Decode a single token ID. Pass as a list to decode for robustness with some tokenizers.
        tok_str = tokenizer.decode([token_id])

        display_token: str
        if tok_str == tokenizer.eos_token or tok_str == tokenizer.bos_token or tok_str == tokenizer.pad_token:
            # For special tokens, escape them and then wrap in brackets
            display_token = f"[{html.escape(tok_str.upper())}]"
        else:
            # For regular tokens:
            # 1. Handle space prefixes (like 'Ġ' or leading ' ') by converting to '&nbsp;'
            #    and separating the rest of the token.
            # 2. HTML escape the rest of the token.
            # 3. Replace newline characters in the escaped part with '\n'.
            
            temp_tok_str = tok_str
            prefix = ""

            if temp_tok_str.startswith('Ġ'): # GPT2/RoBERTa BPE
                prefix = '&nbsp;'
                temp_tok_str = temp_tok_str[1:]
            elif temp_tok_str.startswith(' '): # SentencePiece / WordPiece (leading space)
                 prefix = '&nbsp;'
                 temp_tok_str = temp_tok_str[1:]
            
            # Escape the main part of the token string
            escaped_token_part = html.escape(temp_tok_str)
            
            # Replace newlines in the (now escaped) token part with '\n'
            processed_token_part = escaped_token_part.replace("\n", "\\n")
            
            display_token = prefix + processed_token_part
            
        decoded_tokens.append(display_token)

    for token_str, act_val in zip(decoded_tokens, activations_in_context):
        text_color_rgb, background_color_rgba = value_to_color(
            act_val, max_act_for_norm, min_act_for_norm,
            positive_threshold=positive_threshold, negative_threshold=negative_threshold
        )
        
        html_parts.append(
            f'<span style="background-color:{background_color_rgba}; color:rgb({text_color_rgb}); margin-right: 0.00em; padding: 1px; border-radius: 3px;" title="Act: {act_val:.4f}">{token_str}</span>'
        )
    return "".join(html_parts)


def _generate_connections_html(
    module_name_str: str,
    feature_idx_in_module: int,
    suite: SCAESuite,
    k_top_connections: int = 10,
) -> str:
    """Generates HTML for the connections section of the dashboard."""
    try:
        # 1. Get all module names and parse them
        parsed_modules = []
        for name in suite.module_dict.keys():
            try:
                parts = name.split('_')
                type = parts[0]
                layer = int(parts[1])
                parsed_modules.append({'name': name, 'type': type, 'layer': layer})
            except (IndexError, ValueError):
                continue

        current_module_info = next((m for m in parsed_modules if m['name'] == module_name_str), None)
        
        if current_module_info is None:
            return "<p>Could not parse current module name.</p>"

        current_module_layer = current_module_info['layer']
        current_module_type = current_module_info['type']
        current_module = suite.module_dict[module_name_str]

        # 2. UPSTREAM connections (current module is DOWNSTREAM)
        all_upstream_connections = []
        down_module = current_module
        
        upstream_modules_info = [m for m in parsed_modules if m['layer'] < current_module_layer]

        for up_module_info in upstream_modules_info:
            up_module_name = up_module_info['name']
            up_module = suite.module_dict[up_module_name]
            
            if up_module_name not in down_module.connection_masks:
                continue
                
            mask = down_module.connection_masks[up_module_name].forward(temperature=1, hard=True)
            vw = down_module.get_virtual_weights(
                up_name=up_module_name,
                up_ae=up_module.ae,
                down_enc=down_module.ae.encoder.weight,
                connection_mask=mask
            )

            if current_module_type == "attn":
                vw = vw.sum(0)

            if vw.ndim == 2 and feature_idx_in_module < vw.shape[0]:
                feature_connections = vw[feature_idx_in_module]
                non_zero_indices = feature_connections.nonzero(as_tuple=False).squeeze(-1)
                
                # Handle case where nonzero returns a single-element tensor that is not 1-D
                if non_zero_indices.dim() == 0 and non_zero_indices.numel() == 1:
                    non_zero_indices = non_zero_indices.unsqueeze(0)

                for ind in non_zero_indices:
                    val = feature_connections[ind].item()
                    if abs(val) > 1e-4:
                        all_upstream_connections.append({'strength': val, 'module': up_module_name, 'feature_idx': ind.item()})
        
        total_upstream = len(all_upstream_connections)
        top_upstream = sorted([c for c in all_upstream_connections if c['strength'] > 0], key=lambda x: x['strength'], reverse=True)[:k_top_connections]
        bottom_upstream = sorted([c for c in all_upstream_connections if c['strength'] < 0], key=lambda x: x['strength'])[:k_top_connections]


        # 3. DOWNSTREAM connections (current module is UPSTREAM)
        all_downstream_connections = []
        up_module = current_module
        up_module_name = module_name_str
        
        downstream_modules_info = [m for m in parsed_modules if m['layer'] > current_module_layer]

        for down_module_info in downstream_modules_info:
            down_module_name = down_module_info['name']
            down_module = suite.module_dict[down_module_name]
            down_module_type = down_module_info['type']
            
            if up_module_name not in down_module.connection_masks:
                continue

            mask = down_module.connection_masks[up_module_name].forward(temperature=1, hard=True)
            vw = down_module.get_virtual_weights(
                up_name=up_module_name,
                up_ae=up_module.ae,
                down_enc=down_module.ae.encoder.weight,
                connection_mask=mask
            )
            
            if down_module_type == "attn":
                vw = vw.sum(0)
            
            if vw.ndim == 2 and feature_idx_in_module < vw.shape[1]:
                feature_connections = vw[:, feature_idx_in_module]
                non_zero_indices = feature_connections.nonzero(as_tuple=False).squeeze(-1)

                if non_zero_indices.dim() == 0 and non_zero_indices.numel() == 1:
                    non_zero_indices = non_zero_indices.unsqueeze(0)

                for ind in non_zero_indices:
                    val = feature_connections[ind].item()
                    if abs(val) > 1e-4:
                         all_downstream_connections.append({'strength': val, 'module': down_module_name, 'feature_idx': ind.item()})

        total_downstream = len(all_downstream_connections)
        top_downstream = sorted([c for c in all_downstream_connections if c['strength'] > 0], key=lambda x: x['strength'], reverse=True)[:k_top_connections]
        bottom_downstream = sorted([c for c in all_downstream_connections if c['strength'] < 0], key=lambda x: x['strength'])[:k_top_connections]

        # 4. Format into HTML
        html_parts = ['<div style="display: flex; flex-direction: row; justify-content: space-around; width: 100%; background-color: #f2f2f2; color: black; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">']
        
        # Column 1: Top Upstream
        html_parts.append(f'<div style="width: 24%;"><h4>Top Upstream ({total_upstream})</h4><ul style="list-style: none; padding-left: 0;">')
        if top_upstream:
            for conn in top_upstream:
                html_parts.append(f"<li>{conn['strength']:.2f}&nbsp;&nbsp;{conn['module']} / {conn['feature_idx']}</li>")
        else:
            html_parts.append("<li>None found</li>")
        html_parts.append('</ul></div>')
        
        # Column 2: Bottom Upstream
        html_parts.append('<div style="width: 24%;"><h4>Bottom Upstream</h4><ul style="list-style: none; padding-left: 0;">')
        if bottom_upstream:
            for conn in bottom_upstream:
                html_parts.append(f"<li>{conn['strength']:.2f}&nbsp;&nbsp;{conn['module']} / {conn['feature_idx']}</li>")
        else:
            html_parts.append("<li>None found</li>")
        html_parts.append('</ul></div>')

        # Column 3: Top Downstream
        html_parts.append(f'<div style="width: 24%;"><h4>Top Downstream ({total_downstream})</h4><ul style="list-style: none; padding-left: 0;">')
        if top_downstream:
            for conn in top_downstream:
                html_parts.append(f"<li>{conn['strength']:.2f}&nbsp;&nbsp;{conn['module']} / {conn['feature_idx']}</li>")
        else:
            html_parts.append("<li>None found</li>")
        html_parts.append('</ul></div>')

        # Column 4: Bottom Downstream
        html_parts.append('<div style="width: 24%;"><h4>Bottom Downstream</h4><ul style="list-style: none; padding-left: 0;">')
        if bottom_downstream:
            for conn in bottom_downstream:
                html_parts.append(f"<li>{conn['strength']:.2f}&nbsp;&nbsp;{conn['module']} / {conn['feature_idx']}</li>")
        else:
            html_parts.append("<li>None found</li>")
        html_parts.append('</ul></div>')


        html_parts.append('</div>')
        return "".join(html_parts)

    except Exception as e:
        # Also print to console for debugging
        print(f"Error generating connections view for {module_name_str}/{feature_idx_in_module}: {e}")
        import traceback
        traceback.print_exc()
        return f"<div style='border: 1px solid #444; padding: 5px; margin-bottom: 10px; color: #ffaaaa;'>Error generating connections display: {e}</div>"


def generate_feature_dashboard(
    module_name_str: str,
    feature_idx_in_module: int,
    activations_base_dir: str, 
    tokenizer: PreTrainedTokenizerBase,
    model: HookedTransformer, # Added model for logit lens
    suite: SCAESuite,         # Added suite for AE decoder weights
    k_top_contexts: int = 10,
    k_top_connections: int = 10,
    context_window_size: int = 20,
    positive_threshold: float = 0.01, # Thresholds for coloring
    negative_threshold: float = 0.01
):
    """
    Generates an HTML dashboard displaying top-k contexts and logit lens for a given feature.
    Styling is adapted from interp_utils.py.
    """
    print(f"Generating dashboard for: {module_name_str}, Feature Index: {feature_idx_in_module}")
    
    all_feature_activations = [] 

    batch_dirs = sorted([os.path.join(activations_base_dir, d) for d in os.listdir(activations_base_dir) if d.startswith("batch_")])

    if not batch_dirs:
        print(f"No batch data found in {activations_base_dir}")
        display(HTML("<p>No batch data found.</p>"))
        return

    min_overall_activation = float('inf')
    max_overall_activation = float('-inf')

    for batch_dir_path in batch_dirs:
        activations_file = os.path.join(batch_dir_path, f"activations_{module_name_str}.pt")
        if not os.path.exists(activations_file):
            continue

        try:
            data = torch.load(activations_file, map_location='cpu')
            indices_tuple, values, shape = data['indices'], data['values'], data['shape']
            
            b_idx_local, s_idx, f_idx = indices_tuple
            
            feature_match_mask = (f_idx == feature_idx_in_module)
            
            vals_for_feat = values[feature_match_mask]
            b_idx_local_for_feat = b_idx_local[feature_match_mask]
            s_idx_for_feat = s_idx[feature_match_mask]

            if vals_for_feat.numel() > 0:
                min_overall_activation = min(min_overall_activation, vals_for_feat.min().item())
                max_overall_activation = max(max_overall_activation, vals_for_feat.max().item())

            for val, b_local, s_local in zip(vals_for_feat.tolist(), b_idx_local_for_feat.tolist(), s_idx_for_feat.tolist()):
                all_feature_activations.append((val, batch_dir_path, b_local, s_local))
        except Exception as e:
            print(f"Error loading or processing {activations_file}: {e}")
            continue
            
    if not all_feature_activations:
        print(f"No activations found for feature {module_name_str}/{feature_idx_in_module} across all batches.")
        display(HTML("<p>No activations found for this feature.</p>"))
        return

    all_feature_activations.sort(key=lambda x: x[0], reverse=True)
    top_k_contexts_info = all_feature_activations[:k_top_contexts]

    if not top_k_contexts_info:
        print(f"No top-k contexts to display.")
        display(HTML("<p>No top-k contexts found.</p>"))
        return

    # Use min/max overall activation for color bar normalization
    # If only one value (or all same), add some buffer for make_colorbar
    if min_overall_activation == max_overall_activation:
        if min_overall_activation == 0:
            min_overall_activation = -0.1
            max_overall_activation = 0.1
        else:
            buffer = abs(min_overall_activation * 0.1) if min_overall_activation != 0 else 0.1
            min_overall_activation -= buffer
            max_overall_activation += buffer
    
    # Ensure min is less than max if they became equal due to buffer logic for zero
    if min_overall_activation >= max_overall_activation:
        max_overall_activation = min_overall_activation + 0.1


    # --- Main HTML Structure ---
    # Wrap in a body style similar to interp_utils.py for the token display part
    html_output_parts = ['<body style="background-color:black; color: white; padding: 10px; font-family: monospace;">']
    
    # --- Connections Section ---
    connections_html_content = _generate_connections_html(
        module_name_str, feature_idx_in_module, suite, k_top_connections=k_top_connections
    )
    html_output_parts.append(connections_html_content)
    
    # removed this to reduce clutter
    # html_output_parts.append(f"<h3 style='color: #eee;'>Top {k_top_contexts} unique contexts for {module_name_str} / Feature {feature_idx_in_module}</h3>")
    
    # Add color bar using the overall min/max activations
    colorbar_html = make_colorbar(min_overall_activation, max_overall_activation, positive_threshold=positive_threshold, negative_threshold=negative_threshold)
    html_output_parts.append(f"<div style='margin-bottom: 10px;'>Token Activations: {colorbar_html}</div>")

    printed_contexts = set()
    displayed_contexts_count = 0

    for rank, (act_val, batch_dir_path, sample_idx_in_batch, token_idx_in_sample) in enumerate(top_k_contexts_info):
        if displayed_contexts_count >= k_top_contexts:
            break

        context_identifier = (batch_dir_path, sample_idx_in_batch)
        if context_identifier in printed_contexts:
            continue

        try:
            tokens_file = os.path.join(batch_dir_path, "tokens.pt")
            all_tokens_in_batch = torch.load(tokens_file, map_location='cpu')
            sample_tokens_full = all_tokens_in_batch[sample_idx_in_batch].tolist() 
            
            half_window = context_window_size // 2
            start_idx = max(0, token_idx_in_sample - half_window)
            end_idx = min(len(sample_tokens_full), token_idx_in_sample + half_window + (context_window_size % 2))
            context_token_ids = sample_tokens_full[start_idx:end_idx]

            current_batch_activations_file = os.path.join(batch_dir_path, f"activations_{module_name_str}.pt")
            act_data = torch.load(current_batch_activations_file, map_location='cpu')
            idx_tuple, val_tensor, shp = act_data['indices'], act_data['values'], act_data['shape']
            
            dense_sample_feature_activations = torch.zeros(shp[1], dtype=val_tensor.dtype) # Use dtype from loaded tensor
            mask_for_sample_and_feature = (idx_tuple[0] == sample_idx_in_batch) & (idx_tuple[2] == feature_idx_in_module)
            seq_indices_for_s_f = idx_tuple[1][mask_for_sample_and_feature]
            vals_for_s_f = val_tensor[mask_for_sample_and_feature]
            
            if seq_indices_for_s_f.numel() > 0:
                 dense_sample_feature_activations.scatter_(0, seq_indices_for_s_f, vals_for_s_f)
            else: # Handle case where there are no activations for this specific sample and feature
                 dense_sample_feature_activations = torch.zeros(shp[1], dtype=val_tensor.dtype) # Still need to define it
            
            activations_for_context_window = dense_sample_feature_activations[start_idx:end_idx].tolist()

            # Use overall min/max for normalization in _get_context_html for consistency with colorbar
            context_html = _get_context_html(
                context_token_ids, activations_for_context_window, tokenizer, 
                min_overall_activation, max_overall_activation,
                positive_threshold, negative_threshold
            )
            html_output_parts.append(f"<div style='border: 1px solid #444; padding: 10px; margin-bottom: 10px; border-radius: 5px;'><b>Max act: {act_val:.4f}</b><br><div style='margin-top: 5px; white-space: pre-wrap; line-height: 1.5; overflow-wrap: break-word;'>{context_html}</div></div>")
            printed_contexts.add(context_identifier)
            displayed_contexts_count += 1

        except Exception as e:
            html_output_parts.append(f"<div style='border: 1px solid #444; padding: 5px; margin-bottom: 10px; color: #ffaaaa;'>Error processing context {rank+1}: {e}</div>")
            print(f"Error processing context {rank+1} for feature {module_name_str}/{feature_idx_in_module}: {e}")
            import traceback
            traceback.print_exc()

    # --- Logit Lens Section ---
    logit_lens_html_content = ""
    try:
        # Correctly access ModuleDict element
        if module_name_str in suite.module_dict:
            scae_module = suite.module_dict[module_name_str]
            if hasattr(scae_module, 'ae'):
                ae_instance = scae_module.ae
                feature_vector_for_logit_lens = None

                if isinstance(ae_instance, AutoEncoderTopK):
                    if feature_idx_in_module < ae_instance.decoder.weight.shape[1]:
                        feature_vector_for_logit_lens = ae_instance.decoder.weight[:, feature_idx_in_module]
                    else:
                        logit_lens_html_content = "<p style='color: #ffcc00;'>Feature index out of bounds for AutoEncoderTopK decoder.</p>"
                
                elif isinstance(ae_instance, CrosscoderTopK):
                    if feature_idx_in_module < ae_instance.decoder_weight.shape[0]:
                        # Sum decoder_weight over the n_outputs dimension for the specific feature
                        # decoder_weight shape: (dict_size, n_outputs, d_model)
                        # Select for feature: (n_outputs, d_model)
                        feature_specific_decoder_weights = ae_instance.decoder_weight[feature_idx_in_module, :, :]
                        feature_vector_for_logit_lens = feature_specific_decoder_weights.sum(dim=0)
                    else:
                        logit_lens_html_content = "<p style='color: #ffcc00;'>Feature index out of bounds for CrosscoderTopK decoder_weight.</p>"
                else:
                    logit_lens_html_content = "<p style='color: #ffcc00;'>Unknown AE type for logit lens.</p>"

                if feature_vector_for_logit_lens is not None:
                    # Ensure feature_vector matches the dtype of model.W_U for matmul
                    feature_vector_for_logit_lens = feature_vector_for_logit_lens.to(dtype=model.W_U.dtype, device=model.W_U.device)
                    with torch.no_grad():
                        # ln_final typically expects float32 or the model's main working dtype
                        # If ln_final itself is bfloat16 and W_U is bfloat16, this is fine.
                        # If ln_final is float32, it's good feature_vector is also float32 (or compatible).
                        logit_lens_logits = model.ln_final(feature_vector_for_logit_lens) @ model.W_U
                    
                    top_val, top_ind = torch.topk(logit_lens_logits, k=10, dim=-1)
                    bot_val, bot_ind = torch.topk(logit_lens_logits, k=10, dim=-1, largest=False)
                    
                    logit_lens_html_content = create_logit_lens_html(top_ind.cpu(), top_val.cpu(), bot_ind.cpu(), bot_val.cpu(), tokenizer)
            else:
                logit_lens_html_content = "<p style='color: #ffcc00;'>Could not find AE module for logit lens (module name not in suite.module_dict).</p>"
        else:
            logit_lens_html_content = "<p style='color: #ffcc00;'>Could not find AE module for logit lens (module name not in suite.module_dict).</p>"
    except Exception as e:
        logit_lens_html_content = f"<p style='color: #ffaaaa;'>Error generating logit lens: {e}</p>"
        print(f"Error generating logit lens for {module_name_str}/{feature_idx_in_module}: {e}")
        import traceback
        traceback.print_exc()
        
    html_output_parts.append(logit_lens_html_content)
    html_output_parts.append('</body>')
    display(HTML("".join(html_output_parts)))


def find_non_dead_features(activations_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Finds non-dead features by scanning saved activation files.

    A feature is considered non-dead if it has at least one non-zero activation
    value across all processed batches for a given mode (sparse/non-sparse).

    Args:
        activations_dir: The base directory where activation batches were saved by
                         `collect_activations_and_tokens`. This directory should contain
                         subdirectories like 'sparse_true' and 'sparse_false'.

    Returns:
        A dictionary with keys 'sparse_true' and 'sparse_false'. Each of these
        contains a dictionary mapping module names to a sorted list of their
        non-dead feature indices.
        Example:
        {
            'sparse_true': {'attn_0': [1, 5, ...], 'cc_1': [10, 23, ...]},
            'sparse_false': {'attn_0': [0, 1, 2, ...], 'cc_1': [5, 12, ...]}
        }
    """
    results = {}
    modes = ['sparse_true', 'sparse_false']

    for mode in modes:
        mode_path = os.path.join(activations_dir, mode)
        if not os.path.isdir(mode_path):
            print(f"Directory for mode '{mode}' not found at {mode_path}. Skipping.")
            continue

        non_dead_features_for_mode = {} 

        batch_dirs = sorted([d for d in os.listdir(mode_path) if d.startswith("batch_") and os.path.isdir(os.path.join(mode_path, d))])
        if not batch_dirs:
            print(f"No batch data found in {mode_path}.")
            results[mode] = {}
            continue
            
        print(f"Processing mode: {mode}...")
        for batch_dir_name in batch_dirs:
            batch_dir_path = os.path.join(mode_path, batch_dir_name)
            
            activation_files = [f for f in os.listdir(batch_dir_path) if f.startswith("activations_") and f.endswith(".pt")]

            for activation_file in activation_files:
                module_name = activation_file.replace("activations_", "").replace(".pt", "")
                
                if module_name not in non_dead_features_for_mode:
                    non_dead_features_for_mode[module_name] = set()

                file_path = os.path.join(batch_dir_path, activation_file)
                try:
                    data = torch.load(file_path, map_location='cpu')
                    feature_indices_with_activation = data['indices'][2]
                    
                    if feature_indices_with_activation.numel() > 0:
                        non_dead_features_for_mode[module_name].update(feature_indices_with_activation.tolist())
                except Exception as e:
                    print(f"Error loading or processing {file_path}: {e}")
        
        sorted_non_dead_features = {
            module: sorted(list(features))
            for module, features in non_dead_features_for_mode.items()
        }
        results[mode] = sorted_non_dead_features
    
    print("Finished finding non-dead features.")
    return results


def plot_alive_feature_percentage(
    non_dead_features_dict: Dict[str, Dict[str, List[int]]],
    suite: SCAESuite
):
    """
    Generates and displays a bar chart showing the percentage of alive features
    for each module, separated by sparse and non-sparse modes.

    Args:
        non_dead_features_dict: The dictionary returned by `find_non_dead_features`.
        suite: The SCAESuite object, used to get the total number of features per module.
    """
    
    # 1. Get module names and sort them: attn_0, attn_1, ..., cc_0, cc_1, ...
    def sort_key(name: str):
        parts = name.split('_')
        module_type = parts[0]
        layer = int(parts[1])
        # 'attn' comes before 'cc'
        type_priority = 0 if module_type == 'attn' else 1
        return (type_priority, layer)

    module_names = sorted(suite.module_dict.keys(), key=sort_key)

    vanilla_percentages = []
    sparsely_connected_percentages = []
    
    # 2. Calculate percentages for each module
    for name in module_names:
        scae_module_wrapper = suite.module_dict[name]
        if not scae_module_wrapper or not hasattr(scae_module_wrapper, 'ae'):
            continue
        
        ae_instance = scae_module_wrapper.ae
        total_features = 0
        if isinstance(ae_instance, AutoEncoderTopK):
            total_features = ae_instance.decoder.weight.shape[1]
        elif isinstance(ae_instance, CrosscoderTopK):
            total_features = ae_instance.decoder_weight.shape[0]

        if total_features == 0:
            vanilla_percentages.append(0)
            sparsely_connected_percentages.append(0)
            continue
            
        # Get alive counts from the input dictionary
        num_alive_sparsely_connected = len(non_dead_features_dict.get('sparse_true', {}).get(name, []))
        num_alive_vanilla = len(non_dead_features_dict.get('sparse_false', {}).get(name, []))
        
        sparsely_connected_percentages.append((num_alive_sparsely_connected / total_features) * 100)
        vanilla_percentages.append((num_alive_vanilla / total_features) * 100)

    # 3. Plotting
    x = np.arange(len(module_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, vanilla_percentages, width, label='Vanilla', color='royalblue')
    rects2 = ax.bar(x + width/2, sparsely_connected_percentages, width, label='Sparsely-connected', color='skyblue')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Percentage of Alive Features (%)')
    ax.set_title('Percentage of Alive Features by Module and Mode')
    ax.set_xticks(x)
    ax.set_xticklabels(module_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.show()


def plot_upstream_connection_histograms(
    suite: SCAESuite,
    non_dead_features_dict: Dict[str, Dict[str, List[int]]],
    mode: str = 'sparse_true'
):
    """
    Computes and plots histograms of the number of upstream connections for each
    alive feature in each module.

    Args:
        suite: The SCAESuite object.
        non_dead_features_dict: Dictionary from find_non_dead_features.
        mode: The mode to analyze ('sparse_true' or 'sparse_false').
              Defaults to 'sparse_true' as connections are most relevant there.
    """
    print(f"Generating upstream connection histograms for mode: {mode}")

    # 1. Parse and sort all module names
    parsed_modules = []
    for name in suite.module_dict.keys():
        try:
            parts = name.split('_')
            parsed_modules.append({'name': name, 'type': parts[0], 'layer': int(parts[1])})
        except (IndexError, ValueError):
            continue
    
    def sort_key(mod):
        return (0 if mod['type'] == 'attn' else 1, mod['layer'])
    
    parsed_modules.sort(key=sort_key)
    
    all_module_counts = {}

    # 2. Iterate through each module as the downstream module to calculate connection counts
    for down_module_info in parsed_modules:
        down_module_name = down_module_info['name']
        down_module = suite.module_dict[down_module_name]
        
        alive_features = non_dead_features_dict.get(mode, {}).get(down_module_name, [])
        if not alive_features:
            all_module_counts[down_module_name] = []
            continue

        ae_instance_down = down_module.ae
        total_features_down = 0
        if isinstance(ae_instance_down, AutoEncoderTopK):
            total_features_down = ae_instance_down.decoder.weight.shape[1]
        elif isinstance(ae_instance_down, CrosscoderTopK):
            total_features_down = ae_instance_down.decoder_weight.shape[0]

        if total_features_down == 0:
            continue
            
        total_upstream_connections = torch.zeros(total_features_down, dtype=torch.int32)
        
        upstream_modules_info = [m for m in parsed_modules if m['layer'] < down_module_info['layer']]

        for up_module_info in upstream_modules_info:
            up_module_name = up_module_info['name']
            up_module = suite.module_dict[up_module_name]

            if up_module_name not in down_module.connection_masks:
                continue
            
            mask = down_module.connection_masks[up_module_name].forward(temperature=1, hard=True)
            vw = down_module.get_virtual_weights(
                up_name=up_module_name,
                up_ae=up_module.ae,
                down_enc=down_module.ae.encoder.weight,
                connection_mask=mask
            )
            if down_module_info['type'] == "attn":
                vw = vw.sum(0)
            
            non_zero_per_row = (vw.abs() > 1e-4).sum(dim=1)
            total_upstream_connections += non_zero_per_row.cpu().int()

        alive_feature_indices = torch.tensor(alive_features, dtype=torch.long)
        counts_for_alive_features = total_upstream_connections[alive_feature_indices].tolist()
        all_module_counts[down_module_name] = counts_for_alive_features
    
    # 3. Plotting Preparation
    all_module_counts.pop('attn_0', None)
    all_module_counts.pop('cc_0', None)  # Assuming mlp_0 is cc_0
    
    attn_module_names = sorted([name for name in all_module_counts.keys() if name.startswith('attn')])
    cc_module_names = sorted([name for name in all_module_counts.keys() if name.startswith('cc')])

    if not attn_module_names and not cc_module_names:
        print("No data to plot after filtering.")
        return

    # 4. Plotting
    nrows = max(len(attn_module_names), len(cc_module_names))
    ncols = 2
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 4), constrained_layout=True, squeeze=False)
    
    fig.suptitle('Histogram of Upstream Counts per Alive Feature', fontsize=16)

    # Plot attn modules in the left column
    for i, module_name in enumerate(attn_module_names):
        ax = axes[i, 0]
        counts = all_module_counts[module_name]
        
        if counts:
            log_counts = np.log10([c + 0.1 for c in counts])
            ax.hist(log_counts, bins=30, color='c', edgecolor='k', alpha=0.7)
            ax.set_title(f"{module_name} (n={len(counts)})")
            ax.set_xlabel("log_10(Num Connections + 0.1)")
            ax.set_ylabel("Number of Features")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xlim(left=-1)
            ax.set_xlim(right=4)
            start, _ = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(np.ceil(start), 5, 1))
        else:
            ax.set_title(f"{module_name}")
            ax.text(0.5, 0.5, "No alive features", ha='center', va='center', transform=ax.transAxes)

    # Plot mlp/cc modules in the right column
    for i, module_name in enumerate(cc_module_names):
        ax = axes[i, 1]
        counts = all_module_counts[module_name]
        
        if counts:
            log_counts = np.log10([c + 0.1 for c in counts])
            ax.hist(log_counts, bins=30, color='m', edgecolor='k', alpha=0.7)
            ax.set_title(f"{module_name} (n={len(counts)})")
            ax.set_xlabel("log_10(Num Connections + 0.1)")
            ax.set_ylabel("Number of Features")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xlim(right=4)
            start, _ = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(np.ceil(start), 5, 1))
        else:
            ax.set_title(f"{module_name}")
            ax.text(0.5, 0.5, "No alive features", ha='center', va='center', transform=ax.transAxes)

    # Hide unused subplots
    for i in range(len(attn_module_names), nrows):
        axes[i, 0].set_visible(False)
    for i in range(len(cc_module_names), nrows):
        axes[i, 1].set_visible(False)

    plt.show()


def calculate_global_connection_stats(
    suite: SCAESuite,
    non_dead_features_dict: Dict[str, Dict[str, List[int]]],
    mode: str = 'sparse_true'
) -> Tuple[float, float]:
    """
    Computes the mean and median number of upstream connections across all
    alive features in all relevant modules.

    Args:
        suite: The SCAESuite object.
        non_dead_features_dict: Dictionary from find_non_dead_features.
        mode: The mode to analyze ('sparse_true' or 'sparse_false').

    Returns:
        A tuple containing (mean, median) of the upstream connection counts.
    """
    print(f"Calculating global connection stats for mode: {mode}...")

    parsed_modules = []
    for name in suite.module_dict.keys():
        try:
            parts = name.split('_')
            parsed_modules.append({'name': name, 'type': parts[0], 'layer': int(parts[1])})
        except (IndexError, ValueError):
            continue
    
    all_connection_counts = []

    for down_module_info in parsed_modules:
        down_module_name = down_module_info['name']

        if down_module_name in ['attn_0', 'cc_0']:
            continue
            
        down_module = suite.module_dict[down_module_name]
        
        alive_features = non_dead_features_dict.get(mode, {}).get(down_module_name, [])
        if not alive_features:
            continue

        ae_instance_down = down_module.ae
        total_features_down = 0
        if isinstance(ae_instance_down, AutoEncoderTopK):
            total_features_down = ae_instance_down.decoder.weight.shape[1]
        elif isinstance(ae_instance_down, CrosscoderTopK):
            total_features_down = ae_instance_down.decoder_weight.shape[0]

        if total_features_down == 0:
            continue
            
        total_upstream_connections = torch.zeros(total_features_down, dtype=torch.int32)
        
        upstream_modules_info = [m for m in parsed_modules if m['layer'] < down_module_info['layer']]

        for up_module_info in upstream_modules_info:
            up_module_name = up_module_info['name']
            up_module = suite.module_dict[up_module_name]

            if up_module_name not in down_module.connection_masks:
                continue
            
            mask = down_module.connection_masks[up_module_name].forward(temperature=1, hard=True)
            vw = down_module.get_virtual_weights(
                up_name=up_module_name,
                up_ae=up_module.ae,
                down_enc=down_module.ae.encoder.weight,
                connection_mask=mask
            )
            if down_module_info['type'] == "attn":
                vw = vw.sum(0)
            
            non_zero_per_row = (vw.abs() > 1e-4).sum(dim=1)
            total_upstream_connections += non_zero_per_row.cpu().int()

        alive_feature_indices = torch.tensor(alive_features, dtype=torch.long)
        counts_for_alive_features = total_upstream_connections[alive_feature_indices].tolist()
        all_connection_counts.extend(counts_for_alive_features)

    if not all_connection_counts:
        print("No alive features with connections found to calculate stats on.")
        return 0.0, 0.0

    mean_connections = np.mean(all_connection_counts)
    median_connections = np.median(all_connection_counts)

    print(f"\n--- Global Connection Statistics ---")
    print(f"Total alive features analyzed (from modules > layer 0): {len(all_connection_counts)}")
    print(f"Mean upstream connections per feature: {mean_connections:.2f}")
    print(f"Median upstream connections per feature: {median_connections:.2f}")
    
    return mean_connections, median_connections
