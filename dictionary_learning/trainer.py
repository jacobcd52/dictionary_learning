import os
import pickle
import signal
import sys
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

    track_dead_features: bool = False
    compute_fvu_loss: bool = False

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


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()

def signal_handler(sig, frame):
    print('Keyboard interrupt detected. Cleaning up...')
    cleanup()
    sys.exit(0)

# Register signal handler for keyboard interrupts
signal.signal(signal.SIGINT, signal_handler)


class SCAETrainer:
    def __init__(self, rank, world_size, dtype, cfg, dataset):
        setup(rank, world_size)
        self.device = f"cuda:{rank}"

        self.cfg = cfg
        self.rank = rank

        # Prepare optimizer and distributed dataloader
        self.model = self.load_model(self.device, dtype, cfg)
        self.loader = self.prepare_dataloader(dataset, world_size, rank, cfg)

        self.global_step = 0

        self.train()

    def prepare_optim_and_scheduler(
        self, model: MergedSCAESuite, n_steps: int, cfg: SCAEConfig
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
        lr_scheduler = get_linear_schedule_with_warmup(
            adam, warmup_steps, n_steps
        )

        return adam, lr_scheduler

    def prepare_dataloader(
        self, dataset: Dataset, world_size: int, rank: int, cfg: SCAEConfig
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
        self,
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

        loss = t.nn.functional.cross_entropy(
            logits, input_ids, reduction="mean"
        )
        return loss

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
            n_modules = len(model.scae_suite.module_dict)
            self.num_tokens_since_fired = t.zeros(
                (n_modules, n_features), device="cpu"
            )

        return model

    def update_dead_features(
        self, pruned_features: Dict[str, t.Tensor], num_tokens: int
    ):
        did_fire = t.stack(
            [
                firing_features.sum(dim=(0, 1))
                for firing_features in pruned_features.values()
            ]
        )

        dist.all_reduce(did_fire, op=dist.ReduceOp.MAX)
        did_fire = did_fire.bool()

        row_indices = range(self.num_tokens_since_fired.shape[0])
        for name, row_idx in zip(pruned_features.keys(), row_indices):
            fire_mask = did_fire[row_idx].to("cpu")
            self.num_tokens_since_fired[row_idx][fire_mask] = 0
            self.num_tokens_since_fired[row_idx][~fire_mask] += num_tokens

            if self.rank == 0:
                have_not_fired_mask = self.num_tokens_since_fired[row_idx] > 100_000
                pct_dead = have_not_fired_mask.sum() / have_not_fired_mask.numel()
                wb.log(
                    {f"{name}_dead": pct_dead},
                    step=self.global_step,
                )

    def get_fvu_loss(
        self, reconstructions: Dict[str, t.Tensor], cache: Dict[str, t.Tensor]
    ):
        # Compute FVU loss for each module
        fvu_loss = 0
        for name, recon in reconstructions.items():
            module, layer = name.split("_")
            target = cache[f"blocks.{layer}.hook_{module}_out"]

            total_variance = t.var(target, dim=0).sum()
            residual_variance = t.var(target - recon, dim=0).sum()
            component_fvu = residual_variance / total_variance
            fvu_loss += component_fvu

            if self.rank == 0:
                wb.log(
                    {f"{name}_fvu": component_fvu.item()}, step=self.global_step
                )

        return fvu_loss

    def train_step(self, model: MergedSCAESuite, input_ids: t.Tensor):
        reconstructions, pruned_features, cache = model(input_ids)

        # Update dead feature tracker
        if self.cfg.track_dead_features:
            self.update_dead_features(pruned_features, input_ids.numel())

        total_loss = self.get_ce_loss(model, cache, input_ids, reconstructions)

        if self.cfg.compute_fvu_loss:
            fvu_loss = self.get_fvu_loss(reconstructions, cache)
            total_loss += fvu_loss

        return total_loss

    def eval_step(self):
        pass

    def train(self):
        optimizer, scheduler = self.prepare_optim_and_scheduler(
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
                    wb.log({"loss": loss.item()}, step=self.global_step)

                self.global_step += 1

        cleanup()
