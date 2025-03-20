from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from tqdm import tqdm

from transformer_lens import HookedTransformer
from .scae import SCAESuite, MergedSCAESuite

import wandb as wb


@dataclass
class TrainerConfig:
    wb_project: str = "dictionary_learning"
    wb_name: str = "scae"
    lr: float = 2e-5

    warmup_ratio: float = 0.05
    n_steps: int = 10
    batch_size: int = 16
    quantize_optimizer: bool = False


def prepare_optim_and_scheduler(model, cfg: TrainerConfig):
    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam
        print("Using Adam8bit optimizer")

    else:
        from torch.optim import Adam

    adam = Adam(model.get_trainable_params(), lr=cfg.lr)

    warmup_steps = int(cfg.warmup_ratio * cfg.n_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        adam, warmup_steps, cfg.n_steps
    )

    return adam, lr_scheduler


def train_scae(
    scae: SCAESuite,
    model: HookedTransformer,
    dataset: Dataset,
    accelerator: Accelerator,
    cfg: TrainerConfig,
):

    merged_scae = MergedSCAESuite(model, scae)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer, lr_scheduler = prepare_optim_and_scheduler(merged_scae, cfg)

    merged_scae, loader, optimizer, lr_scheduler = accelerator.prepare(
        merged_scae, loader, optimizer, lr_scheduler
    )

    for batch in tqdm(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        _, loss = merged_scae(input_ids, return_loss=True)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
