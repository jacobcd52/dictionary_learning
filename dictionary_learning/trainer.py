import os
import pickle
from dataclasses import dataclass

import torch as t
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from transformer_lens import HookedTransformer
from tqdm import tqdm
import wandb as wb

from .scae import SCAESuite, MergedSCAESuite


@dataclass
class TrainerConfig:
    wb_project: str = "dictionary_learning"
    wb_name: str = "scae"
    lr: float = 2e-5

    warmup_ratio: float = 0.05
    n_steps: int = 10
    batch_size: int = 16
    quantize_optimizer: bool = False


@dataclass
class SCAEConfig:
    k: int
    expansion_factor: int
    connections_path: str


def prepare_optim_and_scheduler(model, cfg: TrainerConfig):
    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam

        print("Using Adam8bit optimizer")

    else:
        from torch.optim import Adam

    adam = Adam(model.module.get_trainable_params(), lr=cfg.lr)

    warmup_steps = int(cfg.warmup_ratio * cfg.n_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        adam, warmup_steps, cfg.n_steps
    )

    return adam, lr_scheduler


def prepare_dataloader(
    dataset: Dataset, world_size: int, rank: int, cfg: TrainerConfig
):
    # Create distributed samplers
    train_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # Create data loaders
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        pin_memory=True,
    )

    return train_dataloader


def load_model(device, dtype, cfg: SCAEConfig):
    transformer = (
        HookedTransformer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
        )
        .to(device)
        .to(dtype)
    )

    for p in transformer.parameters():
        p.requires_grad = False

    with open(cfg.connections_path, "rb") as f:
        connections = pickle.load(f)

    n_features = transformer.cfg.d_model * cfg.expansion_factor
    scae = SCAESuite(
        transformer,
        cfg.k,
        n_features,
        device=device,
        dtype=dtype,
        connections=connections,
    )

    model = MergedSCAESuite(transformer, scae)

    return model


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()


def train(
    rank: int,
    world_size: int,
    dtype: t.dtype,
    dataset: Dataset,
    train_cfg: TrainerConfig,
    scae_cfg: SCAEConfig,
):
    setup(rank, world_size)

    # Load model and dispatch to correct device
    device = f"cuda:{rank}"
    t.cuda.set_device(device)

    scaler = GradScaler()
    model = load_model(device, dtype, scae_cfg)
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
    )

    # Prepare optimizer and distributed dataloader
    optimizer, scheduler = prepare_optim_and_scheduler(model, train_cfg)
    loader = prepare_dataloader(dataset, world_size, rank, train_cfg)

    loader.sampler.set_epoch(0)

    for batch in tqdm(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        _, loss = model(input_ids, return_loss=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    dist.barrier()
    cleanup()
