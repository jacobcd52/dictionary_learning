import torch as t
from nnsight import LanguageModel
from tqdm import tqdm
from utils import get_modules
from transformer_lens import HookedTransformer, ActivationCache
from collections import namedtuple
import torch as t
from typing import Dict, Iterator, Union, Tuple, Any



class SimpleBuffer:   
    def __init__(
            self,
            data: Iterator[Union[str, t.Tensor]],
            model_name: str,
            ctx_len: int = 128,
            batch_size: int = 512,
            device: str = "cpu",
            dtype: t.dtype = t.float32,
        ):
        # Store the original dataset for reinitialization
        self.original_data = data
        self.data = iter(data)  # Initialize the iterator
        self.model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
        for param in self.model.parameters():
            param.requires_grad = False

        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype        

        self.hook_list = ['blocks.0.hook_resid_pre']
        for layer in range(self.model.cfg.n_layers):
            self.hook_list += [
                f'blocks.{layer}.ln1.hook_scale', 
                f'blocks.{layer}.ln2.hook_scale', 
                f'blocks.{layer}.ln1.hook_normalized', 
                f'blocks.{layer}.ln2.hook_normalized', 
                f'blocks.{layer}.hook_attn_out', 
                f'blocks.{layer}.hook_mlp_out',
                f'blocks.{layer}.attn.hook_pattern'
            ]

        
    def __next__(self) -> Tuple[ActivationCache, t.Tensor]:
        """Return a batch of activations as a named tuple.
        Automatically reinitializes the data iterator when exhausted.
        """
        while True:  # Keep trying until we get a full batch
            try:
                batch = [next(self.data) for _ in range(self.batch_size)]
                tokens = self.model.to_tokens(batch, prepend_bos=True)[:, :self.ctx_len]
                with t.no_grad():
                    loss, cache = self.model.run_with_cache(
                        tokens, 
                        return_type="loss",
                        names_filter=self.hook_list
                    )
                return cache.to(self.dtype), tokens
                
            except StopIteration:
                # Reinitialize the iterator and continue the while loop
                self.data = iter(self.original_data)
    
            
    def __iter__(self):
        return self
    

    @property
    def config(self) -> dict:
        """Return the current configuration of the buffer."""
        return {
            "d_submodule": self.d_submodule,
            "ctx_len": self.ctx_len,
            "batch_size": self.batch_size,
            "device": self.device,
            "needs_tokenization": self.needs_tokenization
        }