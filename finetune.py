from datasets import load_dataset
import torch as t
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from dictionary_learning.buffer import chunk_and_tokenize
from dictionary_learning.trainer import train, TrainerConfig, SCAEConfig

n_connections = 20
path = f"/root/dictionary_learning/pythia_connections/Copy of top_connections_{n_connections}.pkl"

SCAE_CFG = SCAEConfig(
    k=64,
    expansion_factor=4,
    connections_path=path,
)

TRAIN_CFG = TrainerConfig(
    wb_project="dictionary_learning",
    wb_name="finetune",
    lr=2e-5,
    warmup_ratio=0.05,
    epochs=1,
    batch_size=64,
)


def main():
    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("kh4dien/fineweb-100m-sample", split="train[:50%]")
    dataset = chunk_and_tokenize(dataset, tokenizer, "text", 256)

    world_size = t.cuda.device_count()

    mp.spawn(
        train,
        args=(world_size, t.bfloat16, dataset, TRAIN_CFG, SCAE_CFG),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
