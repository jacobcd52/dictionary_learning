from datasets import load_dataset
import torch as t
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from dictionary_learning.buffer import chunk_and_tokenize
from dictionary_learning.trainer import SCAETrainer, SCAEConfig

PATH_TO_PILE = "/root/dictionary_learning/pile-uncopyrighted"
N_TOKENS = 10_000_000
CFG = SCAEConfig(
    model_name="EleutherAI/pythia-70m",
    wb_project="pythia_scae_official",
    save_to_hf=True,
    hf_username="jacobcd52",
    warmup_ratio=0.00,
    decay_start_ratio=0.7, # this has a significant effect!
    epochs=1,
    batch_size=256,
    k=64,
    expansion_factor=4,
    sample_length=256,
    track_dead_features=True,
    connections_path="", # set in sweep()
    auxk_alpha=0,
    base_lr=1e-3 # OpenAI default of 2e-4 is too low for us
)


if __name__ == "__main__":

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
    print(f"Using {world_size} GPUs")
    
    for connections in [100]:
        PATH_TO_CONNECTIONS = f"/root/dictionary_learning/pythia_connections/top_connections_{connections}.pkl"
        CFG.connections_path = PATH_TO_CONNECTIONS
        CFG.wb_run_name = f"c{connections} lr{CFG.base_lr}_bs{CFG.batch_size}_auxk{CFG.auxk_alpha}"

        mp.spawn(
            SCAETrainer,
            args=(world_size, t.bfloat16, CFG, dataset),
            nprocs=world_size,
            join=True,
        )