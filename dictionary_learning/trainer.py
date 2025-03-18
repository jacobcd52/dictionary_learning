from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass

import wandb as wb


@dataclass
class TrainerConfig:
    wb_project: str
    wb_name: str

    lr_warmup_steps: int
    num_batches: int
    quantize_optimizer: bool = False


def train_scae(cfg: TrainerConfig):

    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam

    else:
        from torch.optim import Adam

    adam = Adam(model.parameters(), lr=cfg.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        adam, cfg.lr_warmup_steps, cfg.num_batches
    )

    wb.init(project="scae", name=cfg.wandb_name)

