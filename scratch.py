# %%
from transformers import AutoTokenizer
from dictionary_learning.gpt_neo import GPTNeoForCausalLM
import torch as t
from datasets import load_dataset

t.set_grad_enabled(False)

model = GPTNeoForCausalLM.from_pretrained(
    "roneneldan/TinyStories-33M",
    torch_dtype=t.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
)
# model.fold_ln("cuda", t.bfloat16)



# %%

from transformers import GPTNeoXConfig


# %%

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

