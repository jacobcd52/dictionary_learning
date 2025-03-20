import os
import pickle
from dataclasses import dataclass
from typing import Dict

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


def prepare_optim_and_scheduler(
    model: MergedSCAESuite, n_steps: int, cfg: SCAEConfig
):
    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam

        print("Using Adam8bit optimizer")

    else:
        from torch.optim import Adam

    if cfg.lr is None:
        n_features = model.transformer.cfg.d_model * cfg.expansion_factor
        cfg.lr = 2e-4 / (n_features / 2**14) ** 0.5

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


def get_ce_loss(
    model: MergedSCAESuite,
    cache,
    input_ids: t.Tensor,
    reconstructions,
):
    resid_final = sum(reconstructions.values())
    resid_final = resid_final + cache["blocks.0.hook_resid_pre"]

    unembed = model.module.transformer.unembed
    ln_final = model.module.transformer.ln_final

    logits = unembed(ln_final(resid_final))

    # Shift sequences by 1
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]

    logits = logits.reshape(-1, logits.size(-1))
    input_ids = input_ids.reshape(-1)

    loss = t.nn.functional.cross_entropy(logits, input_ids, reduction="mean")
    return loss


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()


class SCAETrainer:
    def __init__(self, rank, world_size, dtype, cfg, dataset):
        setup(rank, world_size)
        self.device = f"cuda:{rank}"

        self.cfg = cfg
        self.rank = rank

        # Prepare optimizer and distributed dataloader
        self.model = self.load_model(self.device, dtype, cfg)
        self.loader = prepare_dataloader(dataset, world_size, rank, cfg)

    def load_model(self, device, dtype, cfg: SCAEConfig):
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

        # Create dead feature tracker
        if cfg.track_dead_features:
            for name in model.module_dict:
                self.num_tokens_since_fired[name] = t.zeros(
                    n_features, device="cpu"
                )

        return model

    def update_dead_features(
        self, pruned_features: Dict[str, t.Tensor], num_tokens: int
    ):
        for name, features in pruned_features.items():
            active_feature_indices = features.nonzero().flatten()
            self.num_tokens_since_fired[name][active_feature_indices] = 0
            self.num_tokens_since_fired[name][~active_feature_indices] += (
                num_tokens
            )

    def train_step(self, model: MergedSCAESuite, input_ids: t.Tensor):
        reconstructions, pruned_features, cache = model(input_ids)

        # Update dead feature tracker
        if self.cfg.track_dead_features:
            self.update_dead_features(pruned_features, input_ids.numel())

        # Get cross entropy loss
        ce_loss = get_ce_loss(model, cache, input_ids, reconstructions)

        # Compute FVU loss for each module
        for name, recon in reconstructions.items():
            module, layer = name.split("_")
            target = cache[f"blocks.{layer}.hook_{module}_out"]

            # Compute FVU loss
            total_variance = t.var(target, dim=0).sum()
            residual_variance = t.var(target - recon, dim=0).sum()
            fvu_loss = residual_variance / total_variance

            # Add FVU loss to total loss
            # TODO: Finish

        total_loss = ce_loss

        return total_loss

    def eval_step(self):
        pass

    def train(self):
        optimizer, scheduler = prepare_optim_and_scheduler(
            self.model, len(self.loader), self.cfg
        )

        self.model = DDP(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank,
        )

        if self.rank == 0 and self.cfg.wb_project is not None:
            wb.init(
                project=self.cfg.wb_project,
                name=self.cfg.wb_run_name,
                config=self.cfg.wb_cfg,
                entity=self.cfg.wb_entity,
            )

        for epoch in range(self.cfg.epochs):
            self.loader.sampler.set_epoch(epoch)
            for batch in tqdm(self.loader, disable=self.rank != 0):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)

                loss = self.train_step(self.model, input_ids)
                loss.backward()

                self.model.module.clip_grad_norm()
                optimizer.step()
                scheduler.step()

                if self.rank == 0:
                    wb.log({"loss": loss.item()})

        cleanup()
