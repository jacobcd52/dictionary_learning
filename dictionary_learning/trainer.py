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
from torch.optim.lr_scheduler import LambdaLR
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

    model_name: str ="EleutherAI/pythia-70m-deduped"

    wb_project: str = "pythia_scae_caden"
    wb_run_name: str = "scae_bae"
    wb_entity: str = "training-saes"
    base_lr: float = 2e-4
    save_to_hf: bool = False
    hf_username: str = None

    track_dead_features: bool = False
    compute_fvu_loss: bool = False
    auxk_alpha: float = 0.0

    warmup_ratio: float = 0.05
    decay_start_ratio: float = 0.7
    epochs: int = 1
    batch_size: int = 16
    quantize_optimizer: bool = False
    sample_length: int = 512

    @property
    def wb_cfg(self):
        return {
            "k": self.k,
            "expansion_factor": self.expansion_factor,
            "base_lr": self.base_lr,
            "warmup_ratio": self.warmup_ratio,
            "decay_start_ratio": self.decay_start_ratio,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "quantize_optimizer": self.quantize_optimizer,
            "sample_length": self.sample_length,
            "auxk_alpha": self.auxk_alpha,
        }


def prepare_optim_and_scheduler(
    model: MergedSCAESuite, n_steps: int, cfg: SCAEConfig
):
    if cfg.quantize_optimizer:
        from bitsandbytes.optim import Adam8bit as Adam

        print("Using Adam8bit optimizer")

    else:
        from torch.optim import Adam

    
    n_features = model.transformer.cfg.d_model * cfg.expansion_factor
    lr = cfg.base_lr / (n_features / 2**14) ** 0.5

    adam = Adam(model.get_trainable_params(), lr=lr)

    warmup_steps = int(cfg.warmup_ratio * n_steps)
    decay_start_step = int(cfg.decay_start_ratio * n_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < decay_start_step:
            # Constant LR after warmup until decay starts
            return 1.0
        else:
            # Linear decay from decay_start_step to n_steps
            return max(
                0.0,
                float(n_steps - current_step) / float(max(1, n_steps - decay_start_step))
            )

    lr_scheduler = LambdaLR(adam, lr_lambda=lr_lambda)

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
    print("Keyboard interrupt detected. Cleaning up...")
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
        self.loader = prepare_dataloader(dataset, world_size, rank, cfg)

        self.global_step = 0

        self.train()

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

        if self.rank == 0:
            ce_loss_diff =  loss - cache["loss"]
            wb.log(
                {"train/ce_loss_diff": ce_loss_diff.item()},
                step=self.global_step,
            )

        return loss

    def load_model(self, device, dtype, cfg: SCAEConfig):
        transformer = (
            HookedTransformer.from_pretrained(
                self.cfg.model_name,
            )
            .to(device)
            .to(dtype)
        )

        for p in transformer.parameters():
            p.requires_grad = False

        if cfg.connections_path is not None:
            with open(cfg.connections_path, "rb") as f:
                connections = pickle.load(f)
        else:
            connections = None

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

    def _get_ae(self, name: str):
        """Helper function to get the autoencoder for a given module."""
        return self.model.module.scae_suite.module_dict[name].ae

    def _compute_losses(self, y, sae_out, ae, dead_mask=None):
        """Compute fvu and auxk loss.
        From: https://github.com/EleutherAI/sparsify/blob/main/sparsify/sparse_coder.py

        Args:
            y: Target activations
            sae_out: Current reconstruction
            ae: Autoencoder
            dead_mask: Boolean mask indicating dead features
        """

        # Compute the residual
        e = y - sae_out

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # We're autoencoding, so x = y
            orig_shape = y.shape
            x_flat = y.flatten(0, 1)
            pre_acts = ae.encoder(x_flat - ae.b_dec)
            pre_acts = pre_acts.reshape(orig_shape[0], orig_shape[1], -1)

            # Don't include living latents in this loss
            auxk_latents = t.where(dead_mask[None], pre_acts, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = ae.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return auxk_loss, fvu

    def get_losses(
        self,
        pruned_features: Dict[str, t.Tensor],
        reconstructions: Dict[str, t.Tensor],
        cache: Dict[str, t.Tensor],
    ):
        total_loss = 0
        for module_idx, name in enumerate(pruned_features.keys()):
            module, layer = name.split("_")
            y = cache[f"blocks.{layer}.hook_{module}_out"]

            # Only set dead mask if computing AuxK loss
            dead_mask = (
                self.num_tokens_since_fired[module_idx]
                if self.cfg.auxk_alpha > 0
                else None
            )
            
            aux_k_loss, fvu = self._compute_losses(
                y,
                reconstructions[name],
                self._get_ae(name),
                dead_mask,
            )

            component_loss = fvu + self.cfg.auxk_alpha * aux_k_loss
            total_loss = total_loss + component_loss

            if self.rank == 0:
                wb.log({f"fvu/{name}": fvu.item()}, step=self.global_step)

        return total_loss

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
                have_not_fired_mask = (
                    self.num_tokens_since_fired[row_idx] > 1_000_000
                )
                pct_dead = (
                    have_not_fired_mask.sum() / have_not_fired_mask.numel()
                )
                wb.log(
                    {f"dead_pct/{name}": pct_dead},
                    step=self.global_step,
                )

    def train_step(self, model: MergedSCAESuite, input_ids: t.Tensor):
        reconstructions, pruned_features, cache = model(input_ids)

        # Update dead feature tracker
        if self.cfg.track_dead_features:
            self.update_dead_features(pruned_features, input_ids.numel())

        total_loss = self.get_ce_loss(model, cache, input_ids, reconstructions)

        if self.cfg.compute_fvu_loss:
            reconstruction_loss = self.get_losses(
                pruned_features, reconstructions, cache
            )
            total_loss += reconstruction_loss

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
                    wb.log({"train/loss": loss.item()}, step=self.global_step)

                self.global_step += 1

        if self.cfg.save_to_hf:
            model_save_name = self.cfg.model_name.split("/")[-1]
            hf_repo_save_id = f"{self.cfg.hf_username}/{model_save_name}_{self.cfg.wb_run_name}"
            self.model.module.scae_suite.upload_to_hf(repo_id=hf_repo_save_id)
        cleanup()
