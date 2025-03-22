from datasets import load_dataset
import torch as t
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from dictionary_learning.buffer import chunk_and_tokenize
from dictionary_learning.trainer import SCAETrainer, SCAEConfig


PATH = "/root/dictionary_learning/pythia_connections/Copy of top_connections_20.pkl"
PATH_TO_PILE = "/root/dictionary_learning/pile-uncopyrighted"
N_TOKENS = 25_000_000
CFG = SCAEConfig(
    wb_project="dictionary_learning",
    warmup_ratio=0.00,
    epochs=1,
    batch_size=64,
    k=64,
    expansion_factor=4,
    sample_length=256,
    track_dead_features=True,
    connections_path=None,
)


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        PATH_TO_PILE,
        split="train[:10%]",
        num_proc=10,
    )
    dataset = chunk_and_tokenize(dataset, tokenizer, "text", CFG.sample_length)
    dataset = dataset.select(range(N_TOKENS // CFG.sample_length))

    world_size = t.cuda.device_count()

    mp.spawn(
        SCAETrainer,
        args=(world_size, t.bfloat16, CFG, dataset),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
