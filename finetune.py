from datasets import load_dataset
import torch as t
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from dictionary_learning.buffer import chunk_and_tokenize
from dictionary_learning.trainer import SCAETrainer, SCAEConfig

connections = "20"
PATH = f"/root/dictionary_learning/pythia_connections/Copy of top_connections_{connections}.pkl"
PATH_TO_PILE = "/root/dictionary_learning/pile-uncopyrighted"
N_TOKENS = 1_000_000
CFG = SCAEConfig(
    model_name="EleutherAI/pythia-70m",
    wb_project="pythia_scae_official",
    wb_run_name=f"I'll add this below", # Used as name for saving to hf
    save_to_hf=True,
    hf_username="jacobcd52",
    warmup_ratio=0.00,
    epochs=1,
    batch_size=128,
    k=64,
    expansion_factor=4,
    sample_length=256,
    track_dead_features=True,
    connections_path=PATH,
    auxk_alpha=1e-2
)

CFG.wb_run_name = f"lr{CFG.base_lr}_bs{CFG.batch_size}_auxk{CFG.auxk_alpha}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
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
