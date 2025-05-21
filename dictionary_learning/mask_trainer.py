import os
import signal
import sys
from dataclasses import dataclass
from typing import Dict, Union, Optional

import torch as t
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from transformer_lens import HookedTransformer
from tqdm import tqdm
import wandb as wb

# Added imports
import matplotlib.pyplot as plt
import numpy as np

from .mask_scae import SCAESuite, MergedSCAESuite
from .top_k import AutoEncoderTopK, CrosscoderTopK

from utils import set_seed

set_seed(42)


def set_seed(seed=42):
    """
    Set seed for reproducibility across multiple libraries.

    Args:
        seed (int): Seed value to use. Default is 42.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For some operations in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class SCAEConfig:
    # SCAE Arguments
    k: int
    expansion_factor: int
    target_C: int = 100
    mask_type: str = "simple"

    model_name: str = "EleutherAI/pythia-70m-deduped"

    wb_project: str = "pythia_scae_caden"
    wb_run_name: str = "scae_bae"
    wb_entity: str = "training-saes"
    base_lr: float = 2e-4
    save_to_hf: bool = False
    hf_username: str = None

    track_dead_features: bool = False
    fvu_loss_coeff: float = 1.0
    ce_loss_coeff: float = 1.0

    # New loss coefficients
    ce_loss_sparse_coeff: float = 0.0
    fvu_loss_sparse_coeff: float = 1.0
    feature_act_fvu_coeff: float = 0.0
    mask_loss_coeff: float = 1e-5

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
            "target_C": self.target_C,
            "mask_type": self.mask_type,
            "ce_loss_coeff": self.ce_loss_coeff,
            "fvu_loss_coeff": self.fvu_loss_coeff,
            "ce_loss_sparse_coeff": self.ce_loss_sparse_coeff,
            "fvu_loss_sparse_coeff": self.fvu_loss_sparse_coeff,
            "feature_act_fvu_coeff": self.feature_act_fvu_coeff,
            "mask_loss_coeff": self.mask_loss_coeff,
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
                float(n_steps - current_step)
                / float(max(1, n_steps - decay_start_step)),
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
        self.world_size = world_size

        # Histogram directory setup
        self.histogram_dir = "feature_activation_histograms"
        if self.rank == 0:
            os.makedirs(self.histogram_dir, exist_ok=True)

        # Prepare optimizer and distributed dataloader
        self.dataset = dataset
        self.model = self.load_model(self.device, dtype, cfg)
        self.loader = prepare_dataloader(dataset, world_size, rank, cfg)

        self.global_step = 0

        if self.rank == 0:
            print("SCAE Suite Module Dictionary Sizes:")
            for name, module_instance in self.model.scae_suite.module_dict.items():
                if hasattr(module_instance, 'ae') and hasattr(module_instance.ae, 'dict_size'):
                    print(f"  Module {name}: Dict Size = {module_instance.ae.dict_size}")
                else:
                    print(f"  Module {name}: Dict Size not found or AE not as expected.")

        self.train()

    def _get_temperature(self):
        # Start with high temperature and gradually decrease
        max_steps = len(self.loader)
        return max(0.1, 2.0 * (1 - self.global_step / max_steps))

    def get_ce_loss(
        self,
        model: MergedSCAESuite,
        cache: Dict[str, t.Tensor],
        input_ids: t.Tensor,
        reconstructions: Dict[str, t.Tensor],
    ):
        # model is DDP, so access actual model via model.module
        scae_suite = model.module.scae_suite
        
        total_reconstruction_for_ce = t.zeros_like(cache["blocks.0.hook_resid_pre"])

        for name, recon_val in reconstructions.items():
            module_type = name.split("_")[0]
            if module_type == "cc":
                # recon_val for cc is [B, S, N_outputs, D_model]
                # Sum over N_outputs dimension for CE loss contribution
                total_reconstruction_for_ce += recon_val.sum(dim=2)
            elif module_type == "attn":
                # recon_val for attn is [B, S, D_model]
                total_reconstruction_for_ce += recon_val
            else:
                # Should not happen
                print(f"Warning: Unknown module type {module_type} in get_ce_loss")


        resid_final = total_reconstruction_for_ce + cache["blocks.0.hook_resid_pre"]

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
            ce_loss_diff = loss - cache["loss"]
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

        n_features = transformer.cfg.d_model * cfg.expansion_factor
        scae = SCAESuite(
            transformer,
            cfg.k,
            cfg.target_C,
            n_features,
            mask_type=cfg.mask_type,
            device=device,
            dtype=dtype,
        )

        # --- New Encoder Bias Initialization based on Mean SCAE Feature Activations ---
        temp_merged_model = MergedSCAESuite(transformer, scae)
        num_batches_for_init = 10
        final_mean_F_vectors_synced = {}
        mean_F_values_rank0_cpu = {}

        if self.rank == 0:
            print("Rank 0: Starting NEW encoder bias initialization based on mean SCAE feature activations...")
            temp_init_dataloader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            module_feature_means_accumulator = {name: [] for name in scae.module_dict.keys()}
            batches_processed_count = 0
            initial_temperature = 1.0

            print(f"Rank 0: Processing up to {num_batches_for_init} batches for SCAE feature activation collection...")
            for i, batch_data in enumerate(temp_init_dataloader):
                if i >= num_batches_for_init: break
                current_input_ids = batch_data["input_ids"].to(device)
                
                with t.no_grad():
                    _, pruned_features_output, _ = temp_merged_model(
                        current_input_ids, 
                        temperature=initial_temperature, 
                        runtime_use_sparse_connections_override=False
                    )
                
                for module_name, features_tensor in pruned_features_output.items():
                    if module_name in module_feature_means_accumulator and features_tensor.numel() > 0:
                        mean_batch_seq_features = features_tensor.mean(dim=(0, 1)).cpu()
                        module_feature_means_accumulator[module_name].append(mean_batch_seq_features)
                batches_processed_count += 1
            
            print(f"Rank 0: Processed {batches_processed_count} batches for SCAE feature activations.")

            for module_name, means_list in module_feature_means_accumulator.items():
                if means_list:
                    final_F_cpu = t.stack(means_list).mean(dim=0)
                    mean_F_values_rank0_cpu[module_name] = final_F_cpu
                else:
                    module_ae_dict_size = scae.module_dict[module_name].ae.dict_size
                    mean_F_values_rank0_cpu[module_name] = t.zeros(module_ae_dict_size, dtype=dtype) # CPU tensor
                    if batches_processed_count > 0: # Only warn if we expected data
                       print(f"Rank 0: Warning - No SCAE features collected for module {module_name}. Using zeros for F.")
            if batches_processed_count == 0:
                print("Rank 0: Warning - No batches processed for SCAE features. All F values will be zero.")

        ordered_module_names = list(scae.module_dict.keys())
        broadcast_F_list = []
        if self.rank == 0:
            for name in ordered_module_names:
                cpu_tensor = mean_F_values_rank0_cpu.get(name)
                if cpu_tensor is None: # Fallback, though should be populated by prior logic
                    module_ae_dict_size = scae.module_dict[name].ae.dict_size
                    cpu_tensor = t.zeros(module_ae_dict_size, dtype=dtype)
                broadcast_F_list.append(cpu_tensor.to(device=device, dtype=dtype))
        else:
            for name in ordered_module_names:
                module_ae_dict_size = scae.module_dict[name].ae.dict_size
                broadcast_F_list.append(t.zeros(module_ae_dict_size, device=device, dtype=dtype))
        
        if self.world_size > 1:
            if self.rank == 0: print("Rank 0: Broadcasting mean SCAE feature F vectors...")
            for i_tensor in range(len(broadcast_F_list)):
                dist.broadcast(broadcast_F_list[i_tensor], src=0)
            if self.rank == 0: print("Rank 0: Broadcast complete.")

        for i_name, name in enumerate(ordered_module_names):
            final_mean_F_vectors_synced[name] = broadcast_F_list[i_name]

        if final_mean_F_vectors_synced:
            if self.rank == 0: print("Applying synchronized mean SCAE F vectors to encoder biases...")
            for module_idx, (module_name, scae_module_instance) in enumerate(scae.module_dict.items()):
                if module_name in final_mean_F_vectors_synced:
                    mean_F_for_module = final_mean_F_vectors_synced[module_name]
                    actual_ae_instance = scae_module_instance.ae
                    
                    if hasattr(actual_ae_instance, 'encoder') and actual_ae_instance.encoder.bias is not None:
                        with t.no_grad():
                            actual_ae_instance.encoder.bias.copy_(-mean_F_for_module)
                        if self.rank == 0 and module_idx < 5:
                             print(f"Rank 0: Initialized bias for {module_name} using -F method.")
                    elif self.rank == 0:
                        print(f"Rank 0: Module {module_name} AE has no encoder with bias, skipping -F init for it.")
                elif self.rank == 0:
                     print(f"Rank 0: Mean SCAE F vector for module {module_name} not found after sync, skipping bias init.")
        elif self.rank == 0:
            print("Rank 0: Skipped -F encoder bias initialization as no mean SCAE F vectors were effectively computed/synced.")
        # --- End of New Bias Initialization Logic ---

        final_model_to_return = MergedSCAESuite(transformer, scae)

        # Create dead feature tracker
        if cfg.track_dead_features:
            n_modules = len(final_model_to_return.scae_suite.module_dict)
            self.num_tokens_since_fired = t.zeros(
                (n_modules, n_features), device="cpu"
            )

        return final_model_to_return

    def _get_module(self, name: str):
        """Helper function to get the autoencoder for a given module."""
        # Accessing model.module because self.model is DDP wrapped
        return self.model.module.scae_suite.module_dict[name]

    def _compute_single_auxk_loss(self, y: t.Tensor, sae_out: t.Tensor, ae: Union[AutoEncoderTopK, CrosscoderTopK], dead_mask: Optional[t.Tensor] = None):
        """
        Compute auxk loss for a single autoencoder.
        ae is the actual AutoEncoderTopK or CrosscoderTopK instance.
        y is the target activation [B, S, D_model].
        sae_out is the reconstruction from this ae for y, also [B, S, D_model].
        """
        # Compute the residual
        e = y - sae_out
        total_variance = (y - y.mean(dim=(0,1), keepdim=True)).pow(2).sum() + 1e-8


        if dead_mask is not None and ae.encoder.bias is not None and (num_dead := int(dead_mask.sum())) > 0 :
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            orig_shape = y.shape
            x_flat = y.flatten(0, 1) # Shape: [B*S, D_model]
            
            # AuxK for AutoEncoderTopK (assumes ae.b_dec is single vector or None)
            # AuxK for CrosscoderTopK needs careful handling of b_dec if it's per output.
            # For cc_L's own AuxK, we care about its 0-th head.
            # AutoEncoderTopK.encode does not subtract b_dec. CrosscoderTopK.encode also does not.
            # So, direct encoding of x_flat is fine.
            pre_acts_flat = ae.encoder(x_flat) # Shape [B*S, ae.dict_size]
            
            pre_acts = pre_acts_flat.reshape(orig_shape[0], orig_shape[1], -1)

            auxk_latents = t.where(
                dead_mask[None, None, :].to(pre_acts.device), pre_acts, -t.inf
            )
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, dim=-1, sorted=False)

            buffer_BF = t.zeros_like(pre_acts)
            encoded_acts_BF = buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=t.nn.functional.relu(auxk_acts)
            )
            
            # For decoding, CrosscoderTopK needs features of its dict_size.
            # Its decode method returns [B,S,N,D]. We need [B,S,D] for e_hat.
            # If ae is CrosscoderTopK, we should use its 0-th output head for this auxk.
            if isinstance(ae, CrosscoderTopK):
                # encoded_acts_BF is [B,S, ae.dict_size]
                decoded_all_outputs = ae.decode(encoded_acts_BF) # [B,S,N,D]
                e_hat = decoded_all_outputs[:,:,0,:] # Use 0-th head for its "own" reconstruction
            else: # AutoEncoderTopK
                e_hat = ae.decode(encoded_acts_BF) # [B,S,D]
            
            auxk_loss_val = (e_hat - e.detach()).pow(2).sum()
            auxk_loss_val = scale * auxk_loss_val / total_variance
        else:
            auxk_loss_val = sae_out.new_tensor(0.0)
        
        return auxk_loss_val

    def get_auxk_loss(
        self,
        pruned_features: Dict[str, t.Tensor],
        reconstructions: Dict[str, t.Tensor],
        cache: Dict[str, t.Tensor],
    ) -> t.Tensor:
        total_auxk_loss = 0.0
        
        # Note: self.num_tokens_since_fired is indexed based on the order of module_dict
        # We need to ensure this order matches pruned_features.keys() if we iterate that.
        # Or, better, iterate module_dict keys and get items from pruned_features and reconstructions.
        
        module_names_ordered = list(self.model.module.scae_suite.module_dict.keys())

        for module_idx, name in enumerate(module_names_ordered):
            if name not in reconstructions or name not in pruned_features:
                continue # Skip if this module didn't produce output (e.g. if handling of last layer cc is special)

            module_meta = self._get_module(name) # This is the SCAEModule (e.g. SCAEAttention, SCAECrossCoder)
            actual_ae = module_meta.ae # This is AutoEncoderTopK or CrosscoderTopK

            layer_str = name.split("_")[1]
            module_type = name.split("_")[0]
            
            y_target = None
            sae_direct_reconstruction = None

            if module_type == "attn":
                y_target = cache.get(f"blocks.{layer_str}.hook_attn_out")
                sae_direct_reconstruction = reconstructions.get(name)
            elif module_type == "cc":
                y_target = cache.get(f"blocks.{layer_str}.hook_mlp_out")
                # For a CC module's own AuxK, we use its 0-th output head reconstruction
                # against its primary target (MLP output of the same layer).
                cc_reconstruction_all_outputs = reconstructions.get(name) # [B,S,N,D]
                if cc_reconstruction_all_outputs is not None:
                    sae_direct_reconstruction = cc_reconstruction_all_outputs[:,:,0,:] # [B,S,D]
            
            if y_target is None or sae_direct_reconstruction is None:
                if self.rank == 0:
                    print(f"Warning: Missing target or reconstruction for AuxK loss for module {name}")
                continue

            dead_mask = None
            if self.cfg.track_dead_features and self.cfg.auxk_alpha > 0 and hasattr(self, 'num_tokens_since_fired'):
                 if module_idx < self.num_tokens_since_fired.shape[0]: # Check bounds
                    dead_mask = self.num_tokens_since_fired[module_idx] > 1_000_000 # shape [n_features_for_this_module]
            
            current_auxk_loss = self._compute_single_auxk_loss(
                y_target,
                sae_direct_reconstruction,
                actual_ae,
                dead_mask
            )
            total_auxk_loss += current_auxk_loss
            
            if self.rank == 0:
                 wb.log({f"auxk_loss_module/{name}": current_auxk_loss.item()}, step=self.global_step)
        
        return total_auxk_loss

    def get_fvu_loss(
        self,
        reconstructions: Dict[str, t.Tensor],
        cache: Dict[str, t.Tensor],
        log_prefix: str,
    ) -> t.Tensor:
        # Initialize sum_of_individual_fvus on the correct device and dtype
        # Try to get a template tensor from reconstructions or cache, otherwise fallback.
        example_tensor = None
        if reconstructions:
            example_tensor = next(iter(reconstructions.values()), None)
        
        if example_tensor is None and cache:
             # Fallback to cache if reconstructions is empty
            example_tensor = next(iter(cache.values()), None)

        if example_tensor is not None:
            sum_of_individual_fvus = example_tensor.new_tensor(0.0)
        else:
            # Fallback if both are empty: use model's dtype and trainer's device
            # This case should ideally not happen if FVU loss is being meaningfully computed.
            sum_of_individual_fvus = t.tensor(0.0, device=self.device, dtype=self.model.module.transformer.dtype) # Assumes transformer.dtype is accessible

        n_layers = self.model.module.transformer.cfg.n_layers

        # ATTN FVU
        for l in range(n_layers):
            attn_module_name = f"attn_{l}"
            if attn_module_name in reconstructions:
                y_attn = cache.get(f"blocks.{l}.hook_attn_out")
                recon_attn = reconstructions[attn_module_name]

                if y_attn is None:
                    if self.rank == 0: print(f"Warning: Target y_attn for {attn_module_name} not in cache.")
                    continue
                
                l2_loss_attn = (y_attn - recon_attn).pow(2).sum()
                variance_attn = (y_attn - y_attn.mean(dim=(0,1), keepdim=True)).pow(2).sum() + 1e-8
                
                current_fvu_attn = l2_loss_attn / variance_attn
                sum_of_individual_fvus += current_fvu_attn
                
                if self.rank == 0:
                    # wb.log({f"fvu_l2_loss/attn_{l}": l2_loss_attn.item()}, step=self.global_step) # Already removed by user
                    wb.log({f"{log_prefix}fvu_variance/attn_{l}": variance_attn.item()}, step=self.global_step)
                    wb.log({f"{log_prefix}fvu_contrib/attn_{attn_module_name}": current_fvu_attn.item()}, step=self.global_step)


        # MLP FVU (from CCs)
        for j in range(n_layers): # j is the target mlp_out layer index
            target_mlp_hook_name = f"blocks.{j}.hook_mlp_out"
            y_mlp_j = cache.get(target_mlp_hook_name)

            if y_mlp_j is None:
                if self.rank == 0: print(f"Warning: Target y_mlp_j for layer {j} not in cache.")
                continue

            accumulated_recon_for_mlp_j = t.zeros_like(y_mlp_j)
            
            # Sum contributions from all upstream CC modules cc_i (i < j)
            # and also the cc_j module itself (its 0-th head)
            for i in range(j + 1): # i is the layer of the CC module
                cc_module_name = f"cc_{i}"
                if cc_module_name in reconstructions:
                    cc_i_all_outputs = reconstructions[cc_module_name] # [B,S,N,D]
                    
                    # The output head of cc_i that targets mlp_out at layer j
                    output_head_idx = j - i 
                    
                    if 0 <= output_head_idx < cc_i_all_outputs.shape[2]:
                        contrib_from_cc_i = cc_i_all_outputs[:, :, output_head_idx, :]
                        accumulated_recon_for_mlp_j += contrib_from_cc_i


            l2_loss_mlp_j = (y_mlp_j - accumulated_recon_for_mlp_j).pow(2).sum()
            variance_mlp_j = (y_mlp_j - y_mlp_j.mean(dim=(0,1), keepdim=True)).pow(2).sum() + 1e-8
            
            current_fvu_mlp = l2_loss_mlp_j / variance_mlp_j
            sum_of_individual_fvus += current_fvu_mlp

            if self.rank == 0:
                # wb.log({f"fvu_l2_loss/mlp_{j}": l2_loss_mlp_j.item()}, step=self.global_step) # Already removed by user
                wb.log({f"{log_prefix}fvu_variance/mlp_{j}": variance_mlp_j.item()}, step=self.global_step)
                wb.log({f"{log_prefix}fvu_contrib/mlp_{j}": current_fvu_mlp.item()}, step=self.global_step)

        final_fvu = sum_of_individual_fvus
        
        if self.rank == 0:
            wb.log({f"{log_prefix}fvu/total_fvu": final_fvu.item()}, step=self.global_step)
            
        return final_fvu

    def get_feature_act_fvu_loss(
        self,
        features_sparse_dict: Dict[str, t.Tensor],
        features_non_sparse_dict: Dict[str, t.Tensor],
    ) -> t.Tensor:
        total_feature_fvu = features_sparse_dict.get(next(iter(features_sparse_dict.keys()))).new_tensor(0.0) if features_sparse_dict else t.tensor(0.0)

        for name, features_sparse in features_sparse_dict.items():
            if name not in features_non_sparse_dict:
                if self.rank == 0: print(f"Warning: Module {name} features missing in non-sparse dict for feature FVU.")
                continue
            
            features_non_sparse = features_non_sparse_dict[name]

            # Ensure shapes match
            if features_sparse.shape != features_non_sparse.shape:
                if self.rank == 0: print(f"Warning: Shape mismatch for module {name} features for feature FVU. Sparse: {features_sparse.shape}, Non-sparse: {features_non_sparse.shape}")
                continue

            l2_loss = (features_non_sparse - features_sparse).pow(2).sum()
            variance = (features_non_sparse - features_non_sparse.mean(dim=(0, 1), keepdim=True)).pow(2).sum() + 1e-8
            
            current_fvu = l2_loss / variance
            total_feature_fvu += current_fvu
            
            if self.rank == 0:
                # wb.log({f"feature_act_fvu_l2_loss/{name}": l2_loss.item()}, step=self.global_step) # Optional: log L2
                # wb.log({f"feature_act_fvu_variance/{name}": variance.item()}, step=self.global_step) # Optional: log variance
                wb.log({f"feature_act_fvu_contrib/{name}": current_fvu.item()}, step=self.global_step)
        
        if self.rank == 0:
            wb.log({f"feature_act_fvu/total": total_feature_fvu.item()}, step=self.global_step)
            
        return total_feature_fvu

    def _log_feature_activation_histograms(
        self,
        pruned_features_dict: Dict[str, t.Tensor],
        current_global_step: int,
        batch_size: int,
        seq_len: int,
        log_suffix: str = "non_sparse"
    ):
        if self.rank != 0:
            return

        total_tokens_in_batch = batch_size * seq_len
        if total_tokens_in_batch == 0:
            print("Rank 0: Skipping histogram logging due to zero tokens in batch.")
            return

        for module_name, feature_acts_tensor in pruned_features_dict.items():
            if not isinstance(feature_acts_tensor, t.Tensor) or feature_acts_tensor.numel() == 0:
                print(f"Rank 0: Skipping histogram for {module_name} due to empty or invalid features tensor.")
                continue

            # feature_acts_tensor shape: [batch_size, seq_len, n_features_for_module]
            # Move to CPU for numpy operations
            feature_acts_cpu = feature_acts_tensor.detach().cpu()

            # Count how many tokens each feature was active on
            # A feature is active if its value > 1e-7 (to handle float precision)
            num_active_tokens_per_feature = (feature_acts_cpu > 1e-7).sum(dim=(0, 1))

            # Calculate proportion
            proportions = num_active_tokens_per_feature / total_tokens_in_batch
            
            # log10(proportion + epsilon)
            log_proportions_np = np.log10(proportions.numpy() + 1e-10)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(log_proportions_np, bins=200, alpha=0.75, color='skyblue', edgecolor='black')
            
            # Add horizontal dotted line at y=64
            ax.axhline(y=64, color='red', linestyle=':', alpha=0.7)
            
            title_suffix = "Sparse Connections: False" if log_suffix == "non_sparse" else "Sparse Connections: True"
            ax.set_title(f"Feature Activation Frequencies: {module_name}\\nStep: {current_global_step} ({title_suffix})", fontsize=12)
            ax.set_xlabel("log10(Proportion of Tokens Feature is Active)", fontsize=10)
            ax.set_ylabel("Number of Features", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_yscale('log')  # Set y-axis to log scale
            
            # Set x and y axis limits
            ax.set_xlim(-10, 0)
            ax.set_ylim(1, 4000)
            
            # Improve layout
            fig.tight_layout()

            plot_filename = f"{module_name}_step{current_global_step}_{log_suffix}_act_hist.png"
            plot_filepath = os.path.join(self.histogram_dir, plot_filename)
            
            try:
                fig.savefig(plot_filepath)
            except Exception as e:
                print(f"Rank 0: Failed to save histogram {plot_filepath}. Error: {e}")
            plt.close(fig) # Close the figure to free memory
        if self.rank == 0:
            print(f"Rank 0: Saved feature activation histograms for step {current_global_step} to {self.histogram_dir}")

    def get_mask_loss_and_log_c_metric(self, temperature: float) -> t.Tensor:
        total_mask_loss = t.tensor(0.0, device=self.device) # Ensure tensor is on correct device
        module_names_ordered = list(self.model.module.scae_suite.module_dict.keys())

        for name in module_names_ordered:
            module_meta = self._get_module(name) # SCAEModule instance
            
            # Mask Loss
            if hasattr(module_meta, 'connection_masks') and module_meta.connection_masks: # Check if connection_masks exist and populated
                current_mask_loss = module_meta.get_mask_loss(temperature)
                if isinstance(current_mask_loss, t.Tensor):
                    total_mask_loss += current_mask_loss.to(self.device)
                else: # if it's a float from get_mask_loss returning 0.
                    total_mask_loss += t.tensor(current_mask_loss, device=self.device)
                
                if self.rank == 0:
                    log_value = current_mask_loss.item() if isinstance(current_mask_loss, t.Tensor) else current_mask_loss
                    wb.log({f"mask_loss_module/{name}": log_value}, step=self.global_step)

            # C-Metric
            if hasattr(module_meta, 'connection_masks') and module_meta.connection_masks and self.rank == 0:
                C_total_for_module = 0.0
                num_mask_components = 0
                for _up_mask_name, learnable_mask_instance in module_meta.connection_masks.items():
                    # Ensure learnable_mask_instance is on the correct device before calling it
                    # This might not be necessary if they are already correctly moved during model setup.
                    # learnable_mask_instance = learnable_mask_instance.to(self.device) 
                    mask = learnable_mask_instance(temperature, hard=True) # Get hard mask for C metric
                    if mask.numel() > 0 and mask.shape[0] > 0 : # n_features_down > 0 and mask is not empty
                         c_contribution = mask.sum().item() / mask.shape[0]
                         C_total_for_module += c_contribution
                         num_mask_components += 1
                avg_C_for_module = C_total_for_module / num_mask_components if num_mask_components > 0 else 0.0
                wb.log({f"C_metric/{name}": avg_C_for_module}, step=self.global_step)
        
        if self.rank == 0:
            wb.log({"train/total_mask_loss_unscaled": total_mask_loss.item()}, step=self.global_step)
            
        return total_mask_loss

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
        temperature = self._get_temperature()
        
        # First Pass: sparse_connections = False
        reconstructions_non_sparse, pruned_features_non_sparse, cache_non_sparse = model(
            input_ids, temperature, runtime_use_sparse_connections_override=False
        )

        # Calculate losses for non-sparse pass
        ce_loss_non_sparse_val = self.get_ce_loss(model, cache_non_sparse, input_ids, reconstructions_non_sparse)
        fvu_loss_non_sparse_val = self.get_fvu_loss(reconstructions_non_sparse, cache_non_sparse, log_prefix="non_sparse_")

        if self.rank == 0:
            wb.log({"train/ce_loss_non_sparse_unscaled": ce_loss_non_sparse_val.item()}, step=self.global_step)
            wb.log({"train/fvu_loss_non_sparse_unscaled": fvu_loss_non_sparse_val.item()}, step=self.global_step)

            # Log histograms for non-sparse features
            if self.global_step in [0, 10, 20, 50, 100, 150, 200, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200]:
                b_size, s_len = input_ids.shape[0], input_ids.shape[1]
                self._log_feature_activation_histograms(
                    pruned_features_non_sparse,
                    self.global_step,
                    b_size,
                    s_len,
                    log_suffix="non_sparse"
                )

        # Second Pass: sparse_connections = True
        # Note: This reuses the input_ids and temperature. If the model or cache needs to be reset or handled differently,
        # adjustments would be needed. The current MergedSCAESuite.forward re-calculates cache.
        reconstructions_sparse, pruned_features_sparse, cache_sparse = model(
            input_ids, temperature, runtime_use_sparse_connections_override=True
        )

        # Calculate losses for sparse pass
        ce_loss_sparse_val = self.get_ce_loss(model, cache_sparse, input_ids, reconstructions_sparse)
        fvu_loss_sparse_val = self.get_fvu_loss(reconstructions_sparse, cache_sparse, log_prefix="sparse_")
        
        if self.rank == 0:
            wb.log({"train/ce_loss_sparse_unscaled": ce_loss_sparse_val.item()}, step=self.global_step)
            wb.log({"train/fvu_loss_sparse_unscaled": fvu_loss_sparse_val.item()}, step=self.global_step)

        # Calculate Feature Activation FVU Loss
        feature_act_fvu_val = self.get_feature_act_fvu_loss(pruned_features_sparse, pruned_features_non_sparse)
        if self.rank == 0:
             wb.log({"train/feature_act_fvu_unscaled": feature_act_fvu_val.item()}, step=self.global_step)

        # Mask Loss (calculated once per step, relevant to sparse structure)
        mask_loss_val = self.get_mask_loss_and_log_c_metric(temperature)

        # Update dead feature tracker (using sparse features as decided)
        if self.cfg.track_dead_features: # AuxK is ignored, but dead feature tracking might still be useful for observation
            self.update_dead_features(pruned_features_sparse, input_ids.numel())

        # Combine losses with coefficients
        total_loss = 0.0
        
        # Non-sparse losses
        scaled_ce_loss_non_sparse = self.cfg.ce_loss_coeff * ce_loss_non_sparse_val
        total_loss += scaled_ce_loss_non_sparse
        if self.rank == 0: wb.log({"train/ce_loss_non_sparse_scaled": scaled_ce_loss_non_sparse.item()}, step=self.global_step)

        scaled_fvu_loss_non_sparse = self.cfg.fvu_loss_coeff * fvu_loss_non_sparse_val
        total_loss += scaled_fvu_loss_non_sparse
        if self.rank == 0: wb.log({"train/fvu_loss_non_sparse_scaled": scaled_fvu_loss_non_sparse.item()}, step=self.global_step)

        # Sparse losses
        scaled_ce_loss_sparse = self.cfg.ce_loss_sparse_coeff * ce_loss_sparse_val
        total_loss += scaled_ce_loss_sparse
        if self.rank == 0: wb.log({"train/ce_loss_sparse_scaled": scaled_ce_loss_sparse.item()}, step=self.global_step)
        
        scaled_fvu_loss_sparse = self.cfg.fvu_loss_sparse_coeff * fvu_loss_sparse_val
        total_loss += scaled_fvu_loss_sparse
        if self.rank == 0: wb.log({"train/fvu_loss_sparse_scaled": scaled_fvu_loss_sparse.item()}, step=self.global_step)

        # Feature Activation FVU loss
        scaled_feature_act_fvu = self.cfg.feature_act_fvu_coeff * feature_act_fvu_val
        total_loss += scaled_feature_act_fvu
        if self.rank == 0: wb.log({"train/feature_act_fvu_scaled": scaled_feature_act_fvu.item()}, step=self.global_step)
        
        # Mask Loss
        scaled_mask_loss = self.cfg.mask_loss_coeff * mask_loss_val
        total_loss += scaled_mask_loss
        if self.rank == 0: wb.log({"train/total_mask_loss_scaled": scaled_mask_loss.item()}, step=self.global_step)

        # AuxK is currently ignored as per instruction.
        # If it were to be added, its calculation would need to consider which pass (sparse/non-sparse)
        # it applies to, or if it applies to both, and have its own coefficient.

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
                    # log number of tokens trained so far
                    wb.log(
                        {
                            "train/tokens": self.global_step
                            * self.cfg.batch_size
                            * self.cfg.sample_length
                        },
                        step=self.global_step,
                    )

                self.global_step += 1

        if self.cfg.save_to_hf:
            model_save_name = self.cfg.model_name.split("/")[-1]
            hf_repo_save_id = f"{self.cfg.hf_username}/{model_save_name}_{self.cfg.wb_run_name}"
            self.model.module.scae_suite.upload_to_hf(repo_id=hf_repo_save_id)
        cleanup()
