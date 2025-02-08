from datasets import load_dataset
import zstandard as zstd
import io
import json
from nnsight import LanguageModel
import torch as t

def hf_dataset_to_generator(dataset_name, split='train', streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    def gen():
        for x in iter(dataset):
            yield x['text']
    
    return gen()

def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a 'text' field
    """
    compressed_file = open(data_path, 'rb')
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
    def generator():
        for line in text_stream:
            yield json.loads(line)['text']
    return generator()


def load_model_with_folded_ln2(
        model_name,
        device='cuda',
        torch_dtype=t.bfloat16
    ):
    model = LanguageModel(model_name, device_map=device, torch_dtype=torch_dtype, dispatch=True)

    # test_input = t.randint(0, model.config.vocab_size, (2, 8)).to(device)
    # with model.trace(test_input):
    #     logits_no_fold = model.output.save()

    # W(g(x-xbar) + c) + b = Wg(x-xbar) + Wc + b

    if model_name == "gpt2":
        for layer in range(model.config.n_layer):
            g = model.transformer.h[layer].ln_2.weight.data.clone()
            c = model.transformer.h[layer].ln_2.bias.data.clone()
            W = model.transformer.h[layer].mlp.c_fc.weight.data.clone()
            b = model.transformer.h[layer].mlp.c_fc.bias.data.clone()

            model.transformer.h[layer].ln_2.weight.data = t.ones_like(g)
            model.transformer.h[layer].ln_2.bias.data = t.zeros_like(c)

            model.transformer.h[layer].mlp.c_fc.weight.data = W * g.unsqueeze(1)
            model.transformer.h[layer].mlp.c_fc.bias.data = b + c @ W

        # with model.trace(test_input):
        #     logits_with_fold = model.output.save() 
        
        # print(logits_no_fold.value.logits[0,:5])
        # print(logits_with_fold.value.logits[0,:5])

        # assert t.allclose(logits_no_fold.value.logits, logits_with_fold.value.logits)

    elif model_name == "roneneldan/TinyStories-33M":
        for layer in range(model.config.num_layers):
            g = model.transformer.h[layer].ln_2.weight.data.clone()
            c = model.transformer.h[layer].ln_2.bias.data.clone()
            W = model.transformer.h[layer].mlp.c_fc.weight.data.clone()
            b = model.transformer.h[layer].mlp.c_fc.bias.data.clone()

            model.transformer.h[layer].ln_2.weight.data = t.ones_like(g)
            model.transformer.h[layer].ln_2.bias.data = t.zeros_like(c)

            model.transformer.h[layer].mlp.c_fc.weight.data = (W.T * g.unsqueeze(1)).T
            model.transformer.h[layer].mlp.c_fc.bias.data = b + W @ c

  
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model


def load_iterable_dataset(
        hf_name,
        streaming=True
        ):
    dataset = load_dataset(
        hf_name, 
        split='train', 
        streaming=streaming,
        trust_remote_code=True
        )

    class CustomData():
        def __init__(self, dataset):
            self.data = iter(dataset)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.data)['text']

    return  CustomData(dataset)


def get_modules(model, model_name : str):
    if model_name == "gpt2":
        initial_submodule = model.transformer.h[0]
        submodules = {}
        layernorm_submodules = {}
        for layer in range(model.config.n_layer):
            submodules[f"mlp_{layer}"] = (model.transformer.h[layer].mlp, "in_and_out")
            submodules[f"attn_{layer}"] = (model.transformer.h[layer].attn, "out")

            layernorm_submodules[f"mlp_{layer}"] = model.transformer.h[layer].ln_2

        d_submodule = model.config.n_embd

        # TODO: need ln_final and unebed

    elif model_name == "roneneldan/TinyStories-33M":
        initial_submodule = model.transformer.h[0]
        submodules = {}
        layernorm_submodules = {}
        for layer in range(model.config.num_layers):
            submodules[f"mlp_{layer}"] = (model.transformer.h[layer].mlp, "in_and_out")
            submodules[f"attn_{layer}"] = (model.transformer.h[layer].attn, "out")

            layernorm_submodules[f"mlp_{layer}"] = model.transformer.h[layer].ln_2

        ln_final = model.transformer.ln_f
        unembed = model.lm_head

        d_submodule = model.config.hidden_size

        # match gpt2 for convenience
        model.config.n_embd = model.config.hidden_size
        model.config.n_layer = model.config.num_layers

    else:
        raise ValueError(f"Model {model_name} not supported")

    return initial_submodule, layernorm_submodules, submodules, d_submodule, ln_final, unembed
