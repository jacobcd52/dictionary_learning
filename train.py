import sys
print(sys.executable)
from datasets import load_dataset
import torch as t
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from transformer_lens import utils

from dictionary_learning.buffer import chunk_and_tokenize
from dictionary_learning.mask_trainer import SCAETrainer, SCAEConfig

from utils import set_seed
set_seed(42)


N_CPUS = mp.cpu_count() // 2

PATH_TO_PILE = "/root/dictionary_learning/pile-uncopyrighted"
N_TOKENS = 200_000_000
CFG = SCAEConfig(
    model_name="EleutherAI/pythia-70m",
    wb_project="pythia_scae_cc_sweep",
    save_to_hf=True,
    hf_username="jacobcd52",
    warmup_ratio=0.00,
    decay_start_ratio=0.9,
    epochs=1,
    batch_size=128,
    k=16,
    expansion_factor=8,
    sample_length=128,
    track_dead_features=True,
    base_lr=5e-4,
    target_C=0,
    # auxk_alpha=0.0,
    mask_type="simple",
    ce_loss_coeff=0,
    ce_loss_sparse_coeff=0,
    fvu_loss_coeff = 1.0,
    fvu_loss_sparse_coeff=0.,
    feature_act_fvu_coeff=0.,
    mask_loss_coeff=1e-4,
    sparse_warmup=0.3,
)

if __name__ == "__main__":
    # t.manual_seed(42)
    # t.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    tokenizer = utils.get_tokenizer_with_bos(tokenizer)
    dataset = load_dataset(
        PATH_TO_PILE,
        split="train[:50%]",
        num_proc=N_CPUS,
    )
    dataset = dataset.shuffle(seed=42)

    dataset = chunk_and_tokenize(dataset, tokenizer, "text", CFG.sample_length, num_proc=N_CPUS)
    dataset = dataset.select(range(N_TOKENS // CFG.sample_length))

    world_size = t.cuda.device_count()
    print(f"Using {world_size} GPUs")

    mask_loss_coeffs_sweep = [0]

    for mask_loss_val in mask_loss_coeffs_sweep:
        CFG.mask_loss_coeff = mask_loss_val
        CFG.wb_run_name = f"warmup{CFG.sparse_warmup} k{CFG.k} mask{CFG.mask_loss_coeff} fact_fvu{CFG.feature_act_fvu_coeff} fvu_sparse{CFG.fvu_loss_sparse_coeff} fvu{CFG.fvu_loss_coeff} lr{CFG.base_lr}"

        mp.spawn(
            SCAETrainer,
            args=(world_size, t.bfloat16, CFG, dataset),
            nprocs=world_size,
            join=True,
        )