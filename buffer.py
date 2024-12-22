import torch as t
from nnsight import LanguageModel
import gc
from tqdm import tqdm
from contextlib import nullcontext

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Tuple
import torch as t
import torch.nn as nn
from tqdm import tqdm
import json
import os
import tempfile
from contextlib import nullcontext
from huggingface_hub import hf_hub_download, HfApi


from config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}



@dataclass
class BufferConfig:
    """Configuration for the activation buffer"""
    n_ctxs: int = 30000
    ctx_len: int = 128
    refresh_batch_size: int = 512
    out_batch_size: int = 8192
    device: str = "cpu"
    dtype: t.dtype = t.float32

    def to_dict(self) -> Dict:
        return {
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
            "dtype": str(self.dtype)
        }
    

class AllActivationBuffer:
    def __init__(
            self,
            data,
            model: LanguageModel,
            submodules,
            d_submodule=None,
            n_ctxs=3e4,
            ctx_len=128,
            refresh_batch_size=512,
            out_batch_size=8192,
            device="cpu",
            dtype=t.float32,
        ):
        # Validate submodules structure
        for name, config in submodules.items():
            if not isinstance(config, dict) or not all(k in config for k in ["input_point", "output_point"]):
                raise ValueError(f"Each submodule config must be a dict with 'input_point' and 'output_point' keys")
            for point in ["input_point", "output_point"]:
                if not isinstance(config[point], tuple) or len(config[point]) != 2:
                    raise ValueError(f"{point} must be a tuple of (submodule, in_or_out)")
                submodule, in_or_out = config[point]
                if in_or_out not in ["in", "out"]:
                    raise ValueError(f"in_or_out must be either 'in' or 'out', got {in_or_out}")

        self.data = data
        self.model = model
        self.submodules = submodules
        self.n_ctxs = int(n_ctxs)
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.dtype = dtype
        
        try:
            first_item = next(iter(data))
            self.needs_tokenization = isinstance(first_item, str)
            self.data = iter(data)
        except StopIteration:
            raise ValueError("Empty data iterator provided")
        
        if d_submodule is None:
            d_submodule = {}
            for name, config in submodules.items():
                try:
                    # Get dimensions from input point
                    input_submodule, input_io = config["input_point"]
                    if input_io == "in":
                        d_in = input_submodule.in_features
                    else:  # input_io == "out"
                        d_in = input_submodule.out_features
                        
                    # Get dimensions from output point
                    output_submodule, output_io = config["output_point"]
                    if output_io == "in":
                        d_out = output_submodule.in_features
                    else:  # output_io == "out"
                        d_out = output_submodule.out_features
                        
                    if d_in != d_out:
                        raise ValueError(f"Input and output dimensions must match for {name}, got {d_in} and {d_out}")
                    d_submodule[name] = d_in
                except:
                    raise ValueError(f"d_submodule cannot be inferred for {name} and must be specified directly")
        elif isinstance(d_submodule, int):
            d_submodule = {name: d_submodule for name in submodules.keys()}
            
        self.d_submodule = d_submodule

        # Initialize activations with specified dtype
        self.activations = {}
        for name in submodules.keys():
            self.activations[name] = t.empty(0, 2, d_submodule[name], device=device, dtype=dtype)

        self.read = t.zeros(0, dtype=t.bool, device=device)
        self.refresh()

    def process_batch(self, batch, batch_size=None):
        """Process a batch of inputs, handling both tokenized and untokenized data."""
        if batch_size is None:
            batch_size = self.refresh_batch_size

        if self.needs_tokenization:
            return self.model.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.ctx_len
            ).input_ids.to(self.device)
        else:
            tokens = t.tensor(batch, device=self.device)
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            return tokens

    def token_batch(self, batch_size=None):
        """Return a batch of tokens, automatically handling tokenization if needed"""
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            batch = [next(self.data) for _ in range(batch_size)]
            return self.process_batch(batch, batch_size)
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return two dictionaries of activations, one for inputs and one for targets.
        Each activation tensor has shape [batch, d].
        """
        with t.no_grad():
            # Check if we need to refresh based on number of unread samples
            n_unread = (~self.read).sum().item()
            if n_unread < self.out_batch_size:
                self.refresh()

            # Set seed for reproducible randomness
            g = t.Generator(device=self.device)
            g.manual_seed(42)
            
            # Get random unread indices using seeded generator
            unreads = (~self.read).nonzero().squeeze()
            perm = t.randperm(len(unreads), device=unreads.device, generator=g)
            idxs = unreads[perm[: self.out_batch_size]]
            self.read[idxs] = True
            
            # Split activations into inputs and targets
            input_acts = {}
            target_acts = {}
            
            for name in self.submodules.keys():
                acts = self.activations[name][idxs]  # [batch, 2, d]
                input_acts[name] = acts[:, 0]  # [batch, d]
                target_acts[name] = acts[:, 1]  # [batch, d]
            
            return input_acts, target_acts

    def _process_states(self, states):
        """Process states to handle tuples and get the main activation tensor"""
        if hasattr(states, "value"):
            states = states.value
        
        if isinstance(states, tuple):
            for item in states:
                if item is not None and hasattr(item, 'shape'):
                    states = item
                    break
        
        return states

    def refresh(self):
        if len(self.read) > 0:
            for name in self.submodules.keys():
                self.activations[name] = self.activations[name][~self.read]

        target_size = self.n_ctxs * self.ctx_len
        while any(len(acts) < target_size for acts in self.activations.values()):
            try:
                tokens = self.token_batch()
                
                with t.no_grad():
                    # Use autocast for mixed precision if using float16 or bfloat16
                    context_manager = (
                        t.cuda.amp.autocast(dtype=self.dtype)
                        if self.dtype in [t.float16, t.bfloat16]
                        else nullcontext()
                    )
                    
                    with context_manager:
                        trace = self.model.trace(tokens)
                        with trace:
                            # Save input and output points for each submodule
                            saved_states = {}
                            for name, config in self.submodules.items():
                                saved_states[name] = {
                                    "input": {
                                        "submodule": config["input_point"][0].input.save(),
                                        "io": config["input_point"][1]
                                    },
                                    "output": {
                                        "submodule": config["output_point"][0].output.save(),
                                        "io": config["output_point"][1]
                                    }
                                }
                            output = trace.output

                    for name, config in self.submodules.items():
                        # Process input point
                        input_states = self._process_states(saved_states[name]["input"]["submodule"])
                        input_io = saved_states[name]["input"]["io"]
                        input_acts = input_states if input_io == "in" else self._process_states(saved_states[name]["input"]["submodule"])
                        
                        # Process output point
                        output_states = self._process_states(saved_states[name]["output"]["submodule"])
                        output_io = saved_states[name]["output"]["io"]
                        output_acts = output_states if output_io == "in" else self._process_states(saved_states[name]["output"]["submodule"])
                        
                        # Convert to specified dtype
                        input_acts = input_acts.to(dtype=self.dtype)
                        output_acts = output_acts.to(dtype=self.dtype)
                        
                        # Reshape
                        batch_size = input_acts.shape[0]
                        seq_len = input_acts.shape[1]
                        hidden_dim = input_acts.shape[2]
                        
                        flat_in = input_acts.reshape(batch_size * seq_len, hidden_dim)
                        flat_out = output_acts.reshape(batch_size * seq_len, hidden_dim)
                        
                        # Stack and concatenate
                        hidden_states = t.stack([flat_in, flat_out], dim=1)
                        self.activations[name] = t.cat(
                            [self.activations[name], hidden_states.to(device=self.device, dtype=self.dtype)], 
                            dim=0
                        )

                    self.read = t.zeros(len(next(iter(self.activations.values()))), dtype=t.bool, device=self.device)
                
            except StopIteration:
                if all(len(acts) == 0 for acts in self.activations.values()):
                    raise StopIteration("No data available to process")
                break
            
    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
            "needs_tokenization": self.needs_tokenization
        }
    