import torch as t
from nnsight import LanguageModel
from tqdm import tqdm
from utils import get_modules


from config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}


from collections import namedtuple
import torch as t

ActivationBatch = namedtuple('ActivationBatch', ['initial', 'src', 'tgt', 'layernorm_scale'])

from collections import namedtuple
import torch as t
from typing import Dict, Iterator, Union, Tuple, Any

ActivationBatch = namedtuple('ActivationBatch', ['initial', 'src', 'tgt', 'layernorm_scale'])

class AllActivationBuffer:
    """Buffer for collecting and managing neural network activations during processing.
    
    Stores activations from specified submodules, including input/output pairs and layernorm scales.
    Handles both tokenized and raw text input, with support for mixed precision training.
    """
    
    def __init__(
            self,
            data: Iterator[Union[str, t.Tensor]],
            model: 'LanguageModel',
            model_name: str,
            n_ctxs: int = int(3e4),
            ctx_len: int = 128,
            refresh_batch_size: int = 512,
            out_batch_size: int = 8192,
            device: str = "cpu",
            dtype: t.dtype = t.float32,
            remove_bos=False
        ):
        """Initialize the activation buffer.
        
        Args:
            data: Iterator of input data (strings or tensors)
            model: Language model to process data
            submodules: Dict mapping names to (submodule, io_type) pairs
            initial_submodule: First submodule for initial activations
            layernorm_submodules: Dict mapping names to layernorm submodules
            d_submodule: Hidden dimensions (int or dict mapping names to dims)
            n_ctxs: Number of contexts to store
            ctx_len: Maximum context length
            refresh_batch_size: Batch size for refreshing buffer
            out_batch_size: Batch size for outputs
            device: Device to store tensors on
            dtype: Data type for stored tensors
        """
        self.data = data
        self.model = model
        initial_submodule, layernorm_submodules, submodules, d_submodule, ln_final, unembed = get_modules(model, model_name)
        self.initial_submodule = initial_submodule
        self.layernorm_submodules = layernorm_submodules
        self.submodules = submodules
        self.ln_final = ln_final
        self.unembed = unembed
    
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.dtype = dtype
        self.start_pos = 1 if remove_bos else 0
        
        # Determine if tokenization is needed
        try:
            first_item = next(iter(data))
            self.needs_tokenization = isinstance(first_item, str)
            self.data = iter(data)
        except StopIteration:
            raise ValueError("Empty data iterator provided")
        
        # Infer or validate hidden dimensions
        self.d_submodule = self._setup_hidden_dims(d_submodule)
        
        # Initialize activation storage
        self.activations = self._init_activations()
        self.initial_activations = t.empty(0, self.d_submodule['initial'], device=device, dtype=dtype)
        self.ln_scales = {
            name: t.empty(0, device=device, dtype=dtype)
            for name in self.layernorm_submodules
        }
        
        self.read = t.zeros(0, dtype=t.bool, device=device)
        self.refresh()
    
    def _setup_hidden_dims(self, d_submodule: Union[Dict[str, int], int, None]) -> Dict[str, int]:
        """Set up hidden dimensions for all submodules."""
        if d_submodule is None:
            d_submodule = {}
            # Infer dimensions from submodules where possible
            for name, (submodule, io) in self.submodules.items():
                try:
                    d_submodule[name] = (
                        submodule.in_features if io == "in" 
                        else submodule.out_features
                    )
                except:
                    raise ValueError(f"d_submodule cannot be inferred for {name}")
            
            try:
                d_submodule['initial'] = self.initial_submodule.in_features
            except:
                raise ValueError("Could not infer input dimension for initial_submodule")
                
        elif isinstance(d_submodule, int):
            d_initial = d_submodule
            d_submodule = {
                'initial': d_initial,
                **{name: d_initial for name in self.submodules}
            }
            
        return d_submodule
    
    def _init_activations(self) -> Dict[str, t.Tensor]:
        """Initialize empty activation tensors for all submodules."""
        return {
            name: t.empty(0, 2, self.d_submodule[name], device=self.device, dtype=self.dtype)
            for name in self.submodules
        }

    def _compute_ln_scale(self, acts: t.Tensor, eps: float=1e-5) -> t.Tensor:
        """Compute layernorm scale factor sqrt(x^2 + eps)."""
        return t.sqrt(acts.var(dim=-1) + eps)

    def process_batch(self, batch: list, batch_size: int = None) -> t.Tensor:
        """Process a batch of inputs, handling tokenization if needed."""
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
            return tokens.unsqueeze(0) if len(tokens.shape) == 1 else tokens

    def token_batch(self, batch_size: int = None) -> t.Tensor:
        """Return a batch of tokens, handling tokenization if needed."""
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            batch = [next(self.data) for _ in range(batch_size)]
            return self.process_batch(batch, batch_size)
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def _process_states(self, states: Any) -> t.Tensor:
        """Extract activation tensor from potentially nested state structure."""
        if hasattr(states, "value"):
            states = states.value
        
        if isinstance(states, tuple):
            for item in states:
                if item is not None and hasattr(item, 'shape'):
                    return item
        
        return states

    def __next__(self) -> ActivationBatch:
        """Return a batch of activations as a named tuple."""
        with t.no_grad():
            # Refresh buffer if needed
            n_unread = (~self.read).sum().item()
            if n_unread < self.out_batch_size:
                self.refresh()

            # Get random unread indices
            g = t.Generator(device=self.device).manual_seed(42)
            unreads = (~self.read).nonzero().squeeze()
            perm = t.randperm(len(unreads), device=unreads.device, generator=g)
            idxs = unreads[perm[: self.out_batch_size]]
            self.read[idxs] = True
            
            # Collect activations for the batch
            initial_act = self.initial_activations[idxs]
            
            input_acts = {}
            target_acts = {}
            for name in self.submodules:
                acts = self.activations[name][idxs]
                input_acts[name] = acts[:, 0]
                target_acts[name] = acts[:, 1]
            
            ln_scales = {
                name: self.ln_scales[name][idxs]
                for name in self.layernorm_submodules
            }
            
            return ActivationBatch(
                initial=initial_act,
                src=input_acts,
                tgt=target_acts,
                layernorm_scale=ln_scales
            )

    def refresh(self):
        """Refresh the activation buffer with new data."""
        # Clear read activations
        if len(self.read) > 0:
            for name in self.submodules:
                self.activations[name] = self.activations[name][~self.read]
            for name in self.layernorm_submodules:
                self.ln_scales[name] = self.ln_scales[name][~self.read]
            self.initial_activations = self.initial_activations[~self.read]

        target_size = self.n_ctxs * self.ctx_len
        while any(len(acts) < target_size for acts in self.activations.values()):
            try:
                tokens = self.token_batch()
                
                with t.no_grad():
                    context_manager = (
                        t.cuda.amp.autocast(dtype=self.dtype)
                        if self.dtype in [t.float16, t.bfloat16]
                        else t.no_grad()
                    )
                    
                    with context_manager:
                        trace = self.model.trace(tokens)
                        with trace:
                            # Save all required activation states
                            initial_input = self.initial_submodule.input.save()
                            saved_inputs = {
                                name: submodule.input.save()
                                for name, (submodule, _) in self.submodules.items()
                            }
                            saved_outputs = {
                                name: submodule.output.save()
                                for name, (submodule, _) in self.submodules.items()
                            }
                            ln_inputs = {
                                name: submodule.input.save()
                                for name, submodule in self.layernorm_submodules.items()
                            }
                            output = trace.output

                    # Process initial activations
                    initial_state = self._process_states(initial_input).to(dtype=self.dtype)[:, self.start_pos:]
                    batch_size, seq_len, hidden_dim = initial_state.shape
                    flat_initial = initial_state.reshape(batch_size * seq_len, hidden_dim)
                    self.initial_activations = t.cat(
                        [self.initial_activations, flat_initial],
                        dim=0
                    )
                    
                    # Process layernorm inputs and compute scales
                    for name, ln_input in ln_inputs.items():
                        ln_state = self._process_states(ln_input).to(dtype=self.dtype)[:, self.start_pos:]
                        flat_ln = ln_state.reshape(batch_size * seq_len, hidden_dim)
                        scale = self._compute_ln_scale(
                            flat_ln, 
                            self.layernorm_submodules[name].eps
                        )
                        self.ln_scales[name] = t.cat(
                            [self.ln_scales[name], scale],
                            dim=0
                        )
                    
                    # Process submodule activations
                    for name, (submodule, io) in self.submodules.items():
                        raw_in = self._process_states(saved_inputs[name]).to(dtype=self.dtype)[:, self.start_pos:]
                        raw_out = self._process_states(saved_outputs[name]).to(dtype=self.dtype)[:, self.start_pos:]
                        
                        flat_in = raw_in.reshape(batch_size * seq_len, hidden_dim)
                        flat_out = raw_out.reshape(batch_size * seq_len, hidden_dim)
                        
                        # Stack inputs/outputs based on IO type
                        hidden_states = t.stack(
                            [flat_in, flat_in] if io == "in"
                            else [flat_out, flat_out] if io == "out"
                            else [flat_in, flat_out],
                            dim=1
                        )
                        
                        self.activations[name] = t.cat(
                            [self.activations[name], hidden_states],
                            dim=0
                        )

                    self.read = t.zeros(
                        len(next(iter(self.activations.values()))),
                        dtype=t.bool,
                        device=self.device
                    )
                
            except StopIteration:
                if all(len(acts) == 0 for acts in self.activations.values()):
                    raise StopIteration("No data available to process")
                break

    def get_seq_activations(self, batch_size: int = 32) -> Tuple[ActivationBatch, t.Tensor]:
            """Return a batch of activations preserving batch and sequence dimensions, along with input tokens.
            
            Args:
                batch_size (int): Number of sequences in each batch
                
            Returns:
                Tuple[ActivationBatch, t.Tensor]: 
                    - ActivationBatch with activations of shape [batch_size, seq_len, hidden_dim]
                    - Input tokens of shape [batch_size, seq_len]
            """
            with t.no_grad():
                # Get fresh tokens and run them through the model
                tokens = self.token_batch(batch_size)
                
                with self.model.trace(tokens) as trace:
                    # Save all required activation states
                    initial_input = self.initial_submodule.input.save()
                    saved_inputs = {
                        name: submodule.input.save()
                        for name, (submodule, _) in self.submodules.items()
                    }
                    saved_outputs = {
                        name: submodule.output.save()
                        for name, (submodule, _) in self.submodules.items()
                    }
                    ln_inputs = {
                        name: submodule.input.save()
                        for name, submodule in self.layernorm_submodules.items()
                    }
                    output = trace.output

                # Process initial activations
                initial_state = self._process_states(initial_input).to(dtype=self.dtype)[:, self.start_pos:]
                
                # Process layernorm inputs and compute scales
                ln_scales = {}
                for name, ln_input in ln_inputs.items():
                    ln_state = self._process_states(ln_input).to(dtype=self.dtype)[:, self.start_pos:]
                    ln_scales[name] = self._compute_ln_scale(
                        ln_state, 
                        self.layernorm_submodules[name].eps
                    )
                
                # Process submodule activations
                input_acts = {}
                target_acts = {}
                for name, (submodule, io) in self.submodules.items():
                    raw_in = self._process_states(saved_inputs[name]).to(dtype=self.dtype)[:, self.start_pos:]
                    raw_out = self._process_states(saved_outputs[name]).to(dtype=self.dtype)[:, self.start_pos:]
                    
                    # Set inputs/outputs based on IO type
                    if io == "in":
                        input_acts[name] = raw_in
                        target_acts[name] = raw_in
                    elif io == "out":
                        input_acts[name] = raw_out
                        target_acts[name] = raw_out
                    else:  # io == "both"
                        input_acts[name] = raw_in
                        target_acts[name] = raw_out

                return (
                    ActivationBatch(
                        initial=initial_state,
                        src=input_acts,
                        tgt=target_acts,
                        layernorm_scale=ln_scales
                    ),
                    tokens[:, self.start_pos:]
                )
            
    def __iter__(self):
        return self

    @property
    def config(self) -> dict:
        """Return the current configuration of the buffer."""
        return {
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
            "needs_tokenization": self.needs_tokenization
        }