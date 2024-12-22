import torch as t
from nnsight import LanguageModel
import gc
from tqdm import tqdm


from config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}


class AllActivationBuffer:
    def __init__(
            self,
            data,
            model: LanguageModel,
            submodules,
            initial_submodule,
            d_submodule=None,
            n_ctxs=3e4,
            ctx_len=128,
            refresh_batch_size=512,
            out_batch_size=8192,
            device="cpu",
            dtype=t.float32,
        ):
        for name, (submodule, io) in submodules.items():
            if io not in ["in", "out", "in_and_out"]:
                raise ValueError(f"io must be either 'in' or 'out' or 'in_and_out', got {io}")

        self.data = data
        self.model = model
        self.submodules = submodules
        self.initial_submodule = initial_submodule
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
                
                try:
                    d_initial = initial_submodule.in_features
                except:
                    raise ValueError("Could not infer input dimension for initial_submodule")
                
        elif isinstance(d_submodule, int):
            d_initial = d_submodule
            d_submodule = {name: d_submodule for name in submodules.keys()}
            
        self.d_submodule = d_submodule

        # Initialize activations with specified dtype
        self.activations = {}
        for name, (submodule, _) in submodules.items():
            # Store as [batch, 2, d] for input/output pairs
            self.activations[name] = t.empty(0, 2, d_submodule[name], device=device, dtype=dtype)
        

        self.initial_activations = t.empty(0, d_initial, device=device, dtype=dtype)
        
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
        Return a tuple (initial_act, input_acts, target_acts) where:
        - initial_act is a tensor of shape [batch, d] from the initial submodule
        - input_acts and target_acts are dictionaries of tensors of shape [batch, d]
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
            
            # Get initial activations
            initial_act = self.initial_activations[idxs]  # [batch, d]
            
            # Split module activations into inputs and targets
            input_acts = {}
            target_acts = {}
            
            for name in self.submodules.keys():
                acts = self.activations[name][idxs]  # [batch, 2, d]
                input_acts[name] = acts[:, 0]  # [batch, d]
                target_acts[name] = acts[:, 1]  # [batch, d]
            
            return initial_act, input_acts, target_acts

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
            self.initial_activations = self.initial_activations[~self.read]

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
                                initial_input = self.initial_submodule.input.save()
                                saved_inputs = {}
                                saved_outputs = {}
                                for name, (submodule, _) in self.submodules.items():
                                    saved_inputs[name] = submodule.input.save()
                                    saved_outputs[name] = submodule.output.save()
                                output = trace.output
                    else:
                        trace = self.model.trace(tokens)
                        with trace:
                            initial_input = self.initial_submodule.input.save()
                            saved_inputs = {}
                            saved_outputs = {}
                            for name, (submodule, _) in self.submodules.items():
                                saved_inputs[name] = submodule.input.save()
                                saved_outputs[name] = submodule.output.save()
                            output = trace.output

                    # Process initial input
                    initial_state = self._process_states(initial_input)
                    initial_state = initial_state.to(dtype=self.dtype)
                    batch_size = initial_state.shape[0]
                    seq_len = initial_state.shape[1]
                    hidden_dim = initial_state.shape[2]
                    flat_initial = initial_state.reshape(batch_size * seq_len, hidden_dim)
                    
                    # Store initial activations
                    self.initial_activations = t.cat(
                        [self.initial_activations, flat_initial], 
                        dim=0
                    )
                    
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