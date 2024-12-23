"""
Training dictionaries
"""

import json
import multiprocessing as mp
import os
from queue import Empty

import torch as t
from tqdm import tqdm

import wandb

from dictionary import AutoEncoder
from trainers.standard import StandardTrainer
from trainers.scae import SubmoduleConfig, SCAESuite, TrainerSCAESuite, TrainerConfig


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except Empty:
            continue
    wandb.finish()


# def log_stats(
#     trainers,
#     step: int,
#     act: t.Tensor,
#     activations_split_by_head: bool,
#     transcoder: bool,
#     log_queues: list=[],
# ):
#     with t.no_grad():
#         # quick hack to make sure all trainers get the same x
#         z = act.clone()
#         for i, trainer in enumerate(trainers):
#             log = {}
#             act = z.clone()
#             if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
#                 act = act[..., i, :]
#             if not transcoder:
#                 act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

#                 # L0
#                 l0 = (f != 0).float().sum(dim=-1).mean().item()
#                 # fraction of variance explained
#                 total_variance = t.var(act, dim=0).sum()
#                 residual_variance = t.var(act - act_hat, dim=0).sum()
#                 frac_variance_explained = 1 - residual_variance / total_variance
#                 log["FVU"] = 1 - frac_variance_explained.item()
#             else:  # transcoder
#                 x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

#                 # L0
#                 l0 = (f != 0).float().sum(dim=-1).mean().item()

#             # log parameters from training
#             log.update({f"{k}": v for k, v in losslog.items()})
#             log[f"l0"] = l0
#             trainer_log = trainer.get_logging_parameters()
#             for name, value in trainer_log.items():
#                 log[f"{name}"] = value

#             if log_queues:
#                 log_queues[i].put(log)


from typing import Union, Dict, Optional, TypeVar
from dataclasses import asdict
import json
import os
import tempfile
import torch as t
from tqdm.auto import tqdm

ModuleSpecs = Union[
    Dict[str, SubmoduleConfig],  # For fresh initialization
    Dict[str, Dict[str, str]],   # For pretrained (keys: repo_id, filename)
]

def train_scae_suite(
    buffer,
    module_specs: ModuleSpecs,
    trainer_config: TrainerConfig,
    connections: Optional[Dict[str, Dict[str, t.Tensor]]] = None,
    steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    use_wandb: bool = False,
    hf_repo_id: Optional[str] = None,
    dtype: t.dtype = t.float32,
    device: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Train a Sparse Connected Autoencoder Suite.
    
    Args:
        buffer: Dataset iterator providing (input_acts, target_acts) pairs
        module_specs: Either:
            - Dictionary mapping names to SubmoduleConfigs for fresh initialization, or
            - Dictionary mapping names to pretrained info with 'repo_id' and 'filename' keys
        trainer_config: Training configuration
        connections: Optional sparse connections between modules. For each downstream
            feature, specifies up to C upstream features it can receive input from.
            Shape: [num_down_features, C] with -1 padding for unused connections.
        steps: Number of training steps (None for full dataset)
        save_steps: Steps between checkpoints (None for no intermediate saves)
        save_dir: Directory to save checkpoints and final model
        log_steps: Steps between logging (None for no logging)
        use_wandb: Whether to use Weights & Biases logging
        hf_repo_id: Optional HuggingFace repo ID to upload trained models
        dtype: Data type for model parameters
        device: Device to train on
        seed: Random seed for reproducibility
        
    Returns:
        Trained TrainerSCAESuite instance
    """
    device = device or ('cuda' if t.cuda.is_available() else 'cpu')
    
    # Check if we're loading pretrained or initializing fresh
    sample_value = next(iter(module_specs.values()))
    if isinstance(sample_value, SubmoduleConfig):
        # Fresh initialization
        suite = SCAESuite(
            submodule_configs=module_specs,
            connections=connections,
            dtype=dtype,
            device=device,
        )
        # Store config for saving
        config_dict = {
            "submodule_configs": {
                name: asdict(cfg) for name, cfg in module_specs.items()
            },
            "is_pretrained": False
        }
    else:
        # Verify pretrained config format
        for name, cfg in module_specs.items():
            required_keys = {'repo_id', 'filename'}
            if not required_keys.issubset(cfg.keys()):
                raise ValueError(
                    f"Pretrained config for {name} missing required keys: "
                    f"{required_keys - set(cfg.keys())}"
                )
        
        # Load pretrained
        suite = SCAESuite.from_pretrained(
            pretrained_configs=module_specs,
            connections=connections,
            device=device
        )
        # Store config for saving
        config_dict = {
            "pretrained_configs": module_specs,
            "is_pretrained": True
        }
    
    # Initialize trainer
    trainer = TrainerSCAESuite(
        suite=suite,
        config=trainer_config,
        seed=seed,
        wandb_name="gpt2_suite_folded_ln" if use_wandb else None,
    )
    
    # Add remaining config information
    config_dict.update({
        "trainer_config": asdict(trainer_config),
        "dtype": str(dtype),
        "buffer_config": {
            "ctx_len": buffer.ctx_len,
            "refresh_batch_size": buffer.refresh_batch_size,
            "out_batch_size": buffer.out_batch_size,
        }
    })
    
    # Save initial config if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
    
    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(
            project="gpt2_suite_folded_ln",
            config=config_dict,
        )
    
    # Training loop
    pbar = tqdm(buffer, total=steps) if steps is not None else tqdm(buffer)
    for step, (initial_acts, input_acts, target_acts, layernorm_scales) in enumerate(pbar):
        if steps is not None and step >= steps:
            break
        
        # Training step with optional logging
        loss = trainer.update(
            step=step,
            initial_acts=initial_acts,
            input_acts=input_acts,
            target_acts=target_acts,
            layernorm_scales=layernorm_scales,
            log_metrics=(log_steps is not None and step % log_steps == 0)
        )
        
        pbar.set_description(f"Loss: {loss:.4f}")
        
        # Save checkpoints if requested
        if save_steps is not None and step % save_steps == 0 and save_dir is not None:
            checkpoint_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save suite state
            t.save(
                {
                    'suite_state': suite.state_dict(),
                    'optimizer_state': trainer.optimizer.state_dict(),
                    'scheduler_state': trainer.scheduler.state_dict(),
                    'step': step,
                },
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
            
            # Save connections separately (if present)
            if connections is not None:
                t.save(
                    connections,
                    os.path.join(checkpoint_dir, "connections.pt")
                )
    
    # Save final model
    if save_dir is not None:
        final_dir = os.path.join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        t.save(
            {
                'suite_state': suite.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'scheduler_state': trainer.scheduler.state_dict(),
                'step': step if step is not None else -1,
            },
            os.path.join(final_dir, "checkpoint.pt")
        )
        
        if connections is not None:
            t.save(
                connections,
                os.path.join(final_dir, "connections.pt")
            )
    
    # Upload to HuggingFace if requested
    if hf_repo_id is not None:
        try:
            from huggingface_hub import HfApi
            
            print(f"\nUploading models to HuggingFace repo: {hf_repo_id}")
            api = HfApi()
            
            # Upload configuration
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                json.dump(config_dict, f, indent=4)
                f.flush()
                api.upload_file(
                    path_or_fileobj=f.name,
                    path_in_repo="config.json",
                    repo_id=hf_repo_id,
                    repo_type="model",
                )
            
            # Upload suite checkpoint
            with tempfile.NamedTemporaryFile(suffix='.pt') as f:
                t.save({
                    'suite_state': suite.state_dict(),
                    'step': step if step is not None else -1,
                }, f.name)
                api.upload_file(
                    path_or_fileobj=f.name,
                    path_in_repo="checkpoint.pt",
                    repo_id=hf_repo_id,
                    repo_type="model",
                )
            
            # Upload connections if present
            if connections is not None:
                with tempfile.NamedTemporaryFile(suffix='.pt') as f:
                    t.save(connections, f.name)
                    api.upload_file(
                        path_or_fileobj=f.name,
                        path_in_repo="connections.pt",
                        repo_id=hf_repo_id,
                        repo_type="model",
                    )
            
            print("Successfully uploaded all models to HuggingFace!")
            
        except ImportError:
            print("Warning: huggingface_hub package not found. Skipping HuggingFace upload.")
            print("To upload to HuggingFace, install with: pip install huggingface_hub")
        except Exception as e:
            print(f"Error uploading to HuggingFace: {str(e)}")
    
    if use_wandb:
        wandb.finish()
    
    return trainer