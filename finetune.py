from datasets import load_dataset
import torch as t
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from dictionary_learning.buffer import chunk_and_tokenize
from dictionary_learning.trainer import train, SCAEConfig


PATH = "/root/dictionary_learning/pythia_connections/Copy of top_connections_20.pkl"

CFG = SCAEConfig(
    wb_project="dictionary_learning",
    lr=2e-5,
    warmup_ratio=0.05,
    epochs=1,
    batch_size=32,
    k=64,
    expansion_factor=4,
    sample_length=512,
    connections_path=PATH,
)

N_TOKENS = 200_000_000


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        "/root/dictionary_learning/pile-uncopyrighted", split="train"
    )
    # Compute the number of rows to get from the
    # Pile depending on a desired number of tokens.
    # Buffer a little bc not all rows might have enough tokens.
    buffered_row_count = int(N_TOKENS / CFG.sample_length * 1.5)
    dataset = dataset.select(range(buffered_row_count))
    dataset = chunk_and_tokenize(dataset, tokenizer, "text", CFG.sample_length)

    world_size = t.cuda.device_count()

    mp.spawn(
        train,
        args=(world_size, t.bfloat16, dataset, CFG),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
