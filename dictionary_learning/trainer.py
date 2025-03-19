from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass

from .scae import MergedSCAESuite

import wandb as wb


@dataclass
class TrainerConfig:
    wb_project: str
    wb_name: str

    lr_warmup_steps: int
    num_batches: int
    quantize_optimizer: bool = False


def prepare_optim_and_scheduler(model, cfg: TrainerConfig):
    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam

    else:
        from torch.optim import Adam

    adam = Adam(model.get_trainable_params(), lr=cfg.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        adam, cfg.lr_warmup_steps, cfg.num_batches
    )

    return adam, lr_scheduler

def train_scae(model: MergedSCAESuite, cfg: TrainerConfig):

    wb.init(project=cfg.wb_project, name=cfg.wb_name)

