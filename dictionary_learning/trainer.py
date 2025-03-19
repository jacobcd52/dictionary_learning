from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator

from .scae import MergedSCAESuite

import wandb as wb


@dataclass
class TrainerConfig:
    wb_project: str = "dictionary_learning"
    wb_name: str = "scae"
    lr: float = 2e-5

    lr_warmup_steps: int = 100
    n_steps: int = 10
    batch_size: int = 16
    quantize_optimizer: bool = False


def prepare_optim_and_scheduler(model, cfg: TrainerConfig):
    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam

    else:
        from torch.optim import Adam

    adam = Adam(model.get_trainable_params(), lr=cfg.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        adam, cfg.lr_warmup_steps, cfg.n_steps
    )

    return adam, lr_scheduler


def train_scae(model: MergedSCAESuite, dataset: Dataset, cfg: TrainerConfig):
    # wb.init(project=cfg.wb_project, name=cfg.wb_name)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    from tqdm import tqdm

    accelerator = Accelerator()

    optimizer, lr_scheduler = prepare_optim_and_scheduler(model, cfg)

    model, loader, optimizer, lr_scheduler = accelerator.prepare(
        model, loader, optimizer, lr_scheduler
    )

    for batch in tqdm(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to("cuda")
        _, loss = model(input_ids, return_loss=True)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
