# %%
from transformers import AutoTokenizer
from dictionary_learning.gpt_neo import GPTNeoModel
import torch as t
from datasets import load_dataset

t.set_grad_enabled(False)

model = GPTNeoModel.from_pretrained(
    "roneneldan/TinyStories-33M",
    torch_dtype=t.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:1%]")

batch = tokenizer(
    dataset["text"][:200], return_tensors="pt", padding=True, truncation=True
).to("cuda")
tokens = batch["input_ids"]
mask = ~(tokens == tokenizer.pad_token_id).any(dim=1)
tokens = tokens[mask][:2]
print(tokens.shape)

# %%

outputs = model(tokens)


# %%

def chunk_and_tokenize(
    data,
    tokenizer,
    format: str = "torch",
    num_proc: int = 3,
    text_key: str = "text",
    max_seq_len: int = 2048,
    # load_from_cache_file: bool = True,
):
    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        batch_encoding = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            x[text_key],  # start with an eos token
            max_length=1024,
            return_attention_mask=False,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"]
        mask = ~(tokens == tokenizer.pad_token_id).any(dim=1)
        tokens = tokens[mask][:2]

        return {
            "input_ids" : tokens.tolist()
        }

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


from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:20%]")
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer.pad_token = tokenizer.eos_token

chunk_and_tokenize(dataset, tokenizer)
