import torch as t
from transformer_lens import HookedTransformer, ActivationCache
from typing import Iterator, Union, Tuple
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset

from multiprocessing import cpu_count

# from gpt_neo import GPTNeoModel

def chunk_and_tokenize(
    data,
    tokenizer: PreTrainedTokenizerBase,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    # load_from_cache_file: bool = True,
):
    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        batch_encoding = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            x[text_key],  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            padding=True,
            truncation=True,
        )
        tokens = batch_encoding["input_ids"]
        mask = ~(tokens == tokenizer.pad_token_id).any(dim=1)
        tokens = tokens[mask][:2]

        return tokens

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        # load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])

# class Buffer: 
#     def __init__(
#         self,
#         device: str = "cuda:0",
#         dataset_id: str = "roneneldan/TinyStories",
#         cache_ctx_len: int = 1024,
#         n_tokens: int = 10_000_000,
#         torch_dtype: t.dtype = t.bfloat16,
#     ):

#         model = GPTNeoModel.from_pretrained("roneneldan/TinyStories-33M", device=device)
#         self.model = t.compile(model)

#         self.tokens = self.tokenize(dataset_id)

#     def tokenize(self, dataset_id: str) -> t.Tensor:
#         dataset = load_dataset(dataset_id, split="train")





class SimpleBuffer:   
    def __init__(
            self,
            data: Iterator[Union[str, t.Tensor]],
            model_name: str,
            ctx_len: int = 128,
            batch_size: int = 512,
            prepend_bos: bool = True,
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
        self.prepend_bos = prepend_bos   

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
                batch = []
                for _ in range(self.batch_size):
                    found=False
                    while not found:
                        text = next(self.data)
                        tokens = self.model.to_tokens([text], prepend_bos=self.prepend_bos)
                        if tokens.shape[1] >= self.ctx_len:
                            batch.append(text)
                            found=True
                tokens = self.model.to_tokens(batch, prepend_bos=self.prepend_bos)[:, :self.ctx_len]
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