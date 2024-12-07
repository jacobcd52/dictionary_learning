import torch as t
from nnsight import LanguageModel
import gc
from tqdm import tqdm


from config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}


class ActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model : LanguageModel, # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 d_submodule=None, # submodule dimension; if None, try to detect automatically
                 io='out', # can be 'in' or 'out'; whether to extract input or output activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu' # device on which to store the activations
                 ):
        
        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        self.activations = t.empty(0, d_submodule, device=device)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):

        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        # Optional progress bar when filling buffer. At larger models / buffer sizes (e.g. gemma-2-2b, 1M tokens on a 4090) this can take a couple minutes.
        # pbar = tqdm(total=self.activation_buffer_size, initial=current_idx, desc="Refreshing activations")

        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                with self.model.trace(
                    self.text_batch(),
                    **tracer_kwargs,
                    invoker_args={"truncation": True, "max_length": self.ctx_len},
                ):
                    if self.io == "in":
                        hidden_states = self.submodule.input[0].save()
                    else:
                        hidden_states = self.submodule.output.save()
                    input = self.model.input.save()
            attn_mask = input.value[1]["attention_mask"]
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = hidden_states[attn_mask != 0]

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]

            self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                self.device
            )
            current_idx += len(hidden_states)

            # pbar.update(len(hidden_states))

        # pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'd_submodule' : self.d_submodule,
            'io' : self.io,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


class HeadActivationBuffer:
    """
    This is specifically designed for training SAEs for individual attn heads in Llama3. 
    Much redundant code; can eventually be merged to ActivationBuffer.
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model : LanguageModel, # LanguageModel from which to extract activations
                 layer, # submodule of the model from which to extract activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 apply_W_O = False,
                 remote = False,
                 ):
        
        self.layer = layer
        self.n_heads = model.config.num_attention_heads
        self.resid_dim = model.config.hidden_size 
        self.head_dim = self.resid_dim //self.n_heads
        self.data = data
        self.model = model
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.apply_W_O = apply_W_O
        self.remote = remote

        self.activations = t.empty(0, self.n_heads, self.head_dim, device=device) # [seq-pos, n_layers, n_head, head_dim]
        self.read = t.zeros(0).bool()
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:
            with t.no_grad():
                with self.model.trace(self.text_batch(), **tracer_kwargs, invoker_args={'truncation': True, 'max_length': self.ctx_len}, remote=self.remote):
                    input = self.model.input.save()
                    hidden_states = self.model.model.layers[self.layer].self_attn.o_proj.input[0][0]#.save()
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]

                    # Reshape by head
                    new_shape = hidden_states.size()[:-1] + (self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
                    hidden_states = hidden_states.view(*new_shape)

                    # Optionally map from head dim to resid dim
                    if self.apply_W_O:
                        hidden_states_W_O_shape = hidden_states.size()[:-1] + (self.model.config.hidden_size,) # (batch_size, seq_len, n_heads, resid_dim)
                        hidden_states_W_O = t.zeros(hidden_states_W_O_shape, device=hidden_states.device)
                        for h in range (self.n_heads):
                            start = h*self.head_dim
                            end = (h+1)*self.head_dim
                            hidden_states_W_O[..., h, start:end] = hidden_states[..., h, :]
                        hidden_states = self.model.model.layers[self.layer].self_attn.o_proj(hidden_states_W_O).save()

            # Apply attention mask
            attn_mask = input.value[1]['attention_mask']
            hidden_states = hidden_states[attn_mask != 0]

            # Save results
            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'layer': self.layer,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


class NNsightActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """

    def __init__(
        self,
        data,  # generator which yields text data
        model: LanguageModel,  # LanguageModel from which to extract activations
        submodule,  # submodule of the model from which to extract activations
        d_submodule=None,  # submodule dimension; if None, try to detect automatically
        io="out",  # can be 'in' or 'out'; whether to extract input or output activations, "in_and_out" for transcoders
        n_ctxs=3e4,  # approximate number of contexts to store in the buffer
        ctx_len=128,  # length of each context
        refresh_batch_size=512,  # size of batches in which to process the data when adding to buffer
        out_batch_size=8192,  # size of batches in which to yield activations
        device="cpu",  # device on which to store the activations
    ):

        if io not in ["in", "out", "in_and_out"]:
            raise ValueError("io must be either 'in' or 'out' or 'in_and_out'")

        if d_submodule is None:
            try:
                if io == "in":
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        
        if io in ["in", "out"]:
            self.activations = t.empty(0, d_submodule, device=device)
        elif io == "in_and_out":
            self.activations = t.empty(0, 2, d_submodule, device=device)

        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]


    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts, return_tensors="pt", max_length=self.ctx_len, padding=True, truncation=True
        )

    def token_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return t.tensor([next(self.data) for _ in range(batch_size)], device=self.device)
        except StopIteration:
            raise StopIteration("End of data stream reached")
        
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        # if batch_size is None:
        #     batch_size = self.refresh_batch_size
        # try:
        #     return [next(self.data) for _ in range(batch_size)]
        # except StopIteration:
        #     raise StopIteration("End of data stream reached")
        return self.token_batch(batch_size)

    def _reshaped_activations(self, hidden_states):
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * seq_len, d_model)
        return hidden_states

    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:

            with t.no_grad(), self.model.trace(
                self.token_batch(),
                **tracer_kwargs,
                invoker_args={"truncation": True, "max_length": self.ctx_len},
            ):
                if self.io in ["in", "in_and_out"]:
                    hidden_states_in = self.submodule.input[0].save()
                if self.io in ["out", "in_and_out"]:
                    hidden_states_out = self.submodule.output.save()

            if self.io == "in":
                hidden_states = self._reshaped_activations(hidden_states_in)
            elif self.io == "out":
                hidden_states = self._reshaped_activations(hidden_states_out)
            elif self.io == "in_and_out":
                hidden_states_in = self._reshaped_activations(hidden_states_in).unsqueeze(1)
                hidden_states_out = self._reshaped_activations(hidden_states_out).unsqueeze(1)
                hidden_states = t.cat([hidden_states_in, hidden_states_out], dim=1)
            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "io": self.io,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


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
            dtype=t.float32,  # Add dtype parameter
        ):
        for name, (submodule, io) in submodules.items():
            if io not in ["in", "out", "in_and_out"]:
                raise ValueError(f"io must be either 'in' or 'out' or 'in_and_out', got {io}")

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
            for name, (submodule, io) in submodules.items():
                try:
                    if io == "in":
                        d_submodule[name] = submodule.in_features
                    else:
                        d_submodule[name] = submodule.out_features
                except:
                    raise ValueError(f"d_submodule cannot be inferred for {name} and must be specified directly")
        elif isinstance(d_submodule, int):
            d_submodule = {name: d_submodule for name in submodules.keys()}
            
        self.d_submodule = d_submodule

        # Initialize activations with specified dtype
        self.activations = {}
        for name, (submodule, _) in submodules.items():
            # TODO: there's no point storing input and output acts in the same [batch, 2, d] tensor,
            # since we just split them into input_acts and output_acts in __next__.
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
                hidden_states_dict = {}
                
                with t.no_grad():
                    # Use autocast for mixed precision if using float16 or bfloat16
                    if self.dtype in [t.float16, t.bfloat16]:
                        with t.cuda.amp.autocast(dtype=self.dtype):
                            trace = self.model.trace(tokens)
                            with trace:
                                saved_inputs = {}
                                saved_outputs = {}
                                for name, (submodule, _) in self.submodules.items():
                                    saved_inputs[name] = submodule.input.save()
                                    saved_outputs[name] = submodule.output.save()
                                output = trace.output
                    else:
                        trace = self.model.trace(tokens)
                        with trace:
                            saved_inputs = {}
                            saved_outputs = {}
                            for name, (submodule, _) in self.submodules.items():
                                saved_inputs[name] = submodule.input.save()
                                saved_outputs[name] = submodule.output.save()
                            output = trace.output

                    for name, (submodule, io) in self.submodules.items():
                        raw_in = self._process_states(saved_inputs[name])
                        raw_out = self._process_states(saved_outputs[name])
                        
                        # Convert to specified dtype
                        raw_in = raw_in.to(dtype=self.dtype)
                        raw_out = raw_out.to(dtype=self.dtype)
                        
                        batch_size = raw_in.shape[0]
                        seq_len = raw_in.shape[1]
                        hidden_dim = raw_in.shape[2]
                        
                        flat_in = raw_in.reshape(batch_size * seq_len, hidden_dim)
                        flat_out = raw_out.reshape(batch_size * seq_len, hidden_dim)
                        
                        if io == "in":
                            hidden_states = t.stack([flat_in, flat_in], dim=1)
                        elif io == "out":
                            hidden_states = t.stack([flat_out, flat_out], dim=1)
                        else:  # io == "in_and_out"
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