# %%
from transformers import AutoTokenizer, GPTNeoModel
from dictionary_learning.dictionary_learning.gpt_neo import GPTNeoModel as Custom
import torch as t
from datasets import load_dataset

t.set_grad_enabled(False)

transformer = GPTNeoModel.from_pretrained(
    "roneneldan/TinyStories-33M",
    torch_dtype=t.bfloat16,
    attn_implementation="eager",
    use_cache=False,
).to("cuda")

model = Custom.from_pretrained(
    "roneneldan/TinyStories-33M",
    attn_implementation="eager",
    use_cache=False,
    torch_dtype=t.bfloat16,
)
# model.fold_ln()
model.to("cuda")
model = t.compile(model)

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:1%]")

batch = tokenizer(
    dataset["text"][:200], return_tensors="pt", padding=True, truncation=True
).to("cuda")
tokens = batch["input_ids"]
mask = ~(tokens == tokenizer.pad_token_id).any(dim=1)
tokens = tokens[mask][0].unsqueeze(0)
print(tokens.shape)


#  %%

transformer_out = transformer(tokens)
custom_out = model(tokens)

percent_equal = t.sum(transformer_out.last_hidden_state[0, -1, :] == custom_out[0][0, -1, :]) / transformer_out.last_hidden_state.shape[2]
print(percent_equal)

# %%

transformer_out.last_hidden_state[0, -1, :]

# %%

custom_out[0, -1, :]

# %%

from transformer_lens import HookedTransformer

tl_model = HookedTransformer.from_pretrained("roneneldan/TinyStories-33M", device="cuda", dtype=t.bfloat16)


# %%

def time(which_model):
    import time
    start = time.time()
    for _ in range(100):
        _ = which_model(tokens)
    end = time.time()
    return end - start


print(time(tl_model))
print(time(model))

# %%
