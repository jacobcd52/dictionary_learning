# %%
from transformers import AutoTokenizer, GPTNeoModel
from gptneo import GPTNeoModel as Custom
import torch as t
from datasets import load_dataset

t.set_grad_enabled(False)

transformer = GPTNeoModel.from_pretrained(
    "roneneldan/TinyStories-33M",
    torch_dtype=t.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
).to("cuda")

model = Custom.from_pretrained(
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
tokens = tokens[mask][0].unsqueeze(0)
print(tokens.shape)

transformer_out = transformer(tokens)
custom_out = model(tokens)

percent_equal = t.sum(transformer_out.last_hidden_state[0, -1, :] == custom_out[0, -1, :]) / transformer_out.last_hidden_state.shape[2]
print(percent_equal)



