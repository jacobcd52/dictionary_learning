from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

# from accelerate import Accelerator

from .scae import SCAESuite
from .buffer import Buffer

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


def train_scae(model: SCAESuite, buffer: Buffer, cfg: TrainerConfig):
    # wb.init(project=cfg.wb_project, name=cfg.wb_name)

    # loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    from tqdm import tqdm

    # accelerator = Accelerator()

    optimizer, lr_scheduler = prepare_optim_and_scheduler(model, cfg)

    # model, loader, optimizer, lr_scheduler = accelerator.prepare(
    #     model, loader, optimizer, lr_scheduler
    # )

    for batch, tokens in tqdm(buffer):
        # Clear all gradients in the model
        model.zero_grad()  # Clear gradients on the whole model
        optimizer.zero_grad()  # Clear gradients on optimizer params
        
        reconstructions = model(batch)
        loss = model.get_ce_loss(batch, reconstructions, tokens)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()