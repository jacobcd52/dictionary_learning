import os
import pickle
from dataclasses import dataclass

import torch as t
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from transformer_lens import HookedTransformer
from tqdm import tqdm
import wandb as wb

from .scae import SCAESuite, MergedSCAESuite


@dataclass
class SCAEConfig:
    # SCAE Arguments
    k: int
    expansion_factor: int
    connections_path: str

    wb_project: str = "dictionary_learning"
    wb_run_name: str = "scae"
    wb_entity: str = "steering-finetuning"
    lr: float = None

    warmup_ratio: float = 0.05
    epochs: int = 1
    batch_size: int = 16
    quantize_optimizer: bool = False
    sample_length: int = 512

    @property
    def wb_cfg(self):
        return {
            "k": self.k,
            "expansion_factor": self.expansion_factor,
            "lr": self.lr,
            "warmup_ratio": self.warmup_ratio,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "quantize_optimizer": self.quantize_optimizer,
            "sample_length": self.sample_length,
        }

def prepare_optim_and_scheduler(model: MergedSCAESuite, n_steps: int, cfg: SCAEConfig):
    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam
        print("Using Adam8bit optimizer")

    else:
        from torch.optim import Adam

    if cfg.lr is None:
        n_features = model.transformer.cfg.d_model * cfg.expansion_factor
        cfg.lr = 2e-4 / (n_features / 2**14)**0.5

    adam = Adam(model.get_trainable_params(), lr=cfg.lr)

    warmup_steps = int(cfg.warmup_ratio * n_steps)
    lr_scheduler = get_linear_schedule_with_warmup(adam, warmup_steps, n_steps)

    return adam, lr_scheduler


def prepare_dataloader(
    dataset: Dataset, world_size: int, rank: int, cfg: SCAEConfig
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
    cfg: SCAEConfig,
):
    setup(rank, world_size)
    device = f"cuda:{rank}"

    # Prepare optimizer and distributed dataloader
    model = load_model(device, dtype, cfg)
    loader = prepare_dataloader(dataset, world_size, rank, cfg)
    optimizer, scheduler = prepare_optim_and_scheduler(
        model, len(loader), cfg
    )

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
    )

    if rank == 0 and cfg.wb_project is not None:
        wb.init(
            project=cfg.wb_project,
            name=cfg.wb_run_name,
            config=cfg.wb_cfg,
            entity=cfg.wb_entity,
        )

    for epoch in range(cfg.epochs):
        loader.sampler.set_epoch(epoch)
        for batch in tqdm(loader, disable=rank != 0):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            _, loss = model(input_ids, return_loss=True)

            loss.backward()
            # model.module.clip_grad_norm()
            optimizer.step()
            scheduler.step()

            if rank == 0:
                wb.log({"loss": loss.item()})

    cleanup()
