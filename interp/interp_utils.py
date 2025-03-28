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