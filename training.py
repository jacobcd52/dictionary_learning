"""
Training dictionaries
"""

import json
import os
from queue import Empty
import torch as t
from typing import Union, Dict, Optional, TypeVar
from dataclasses import asdict
import tempfile
from tqdm.auto import tqdm
import wandb

from trainers.scae import SubmoduleConfig, SCAESuite

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


def initialize_optimizers(suite, base_lr):
    """Initialize optimizers with custom learning rates based on dictionary sizes"""
    lrs = {
        name: base_lr / (ae.dict_size / 2**14)**0.5
        for name, ae in suite.aes.items()
    }
    optimizer = t.optim.Adam([
        {'params': ae.parameters(), 'lr': lrs[name]}
        for name, ae in suite.aes.items()
    ], betas=(0.9, 0.999))
    return optimizer, lrs

def get_lr_scheduler(optimizer, steps, lr_decay_start_proportion):
    """Create learning rate scheduler"""
    def lr_fn(step):
        if step < lr_decay_start_proportion * steps:
            return 1.0
        return (steps - step) / (steps - lr_decay_start_proportion * steps)
    
    return t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

def compute_mse_loss(suite, initial_acts, input_acts, target_acts, layernorm_scales):
    """Compute MSE loss using FVU (Fraction of Variance Unexplained)"""
    reconstructions = suite.pruned_forward(
        initial_acts=initial_acts,
        inputs=input_acts,
        layernorm_scales=layernorm_scales,
        return_topk=False
    )
    
    total_loss = 0
    losses = {}
    
    for name, ae in suite.aes.items():
        if name not in input_acts:
            continue
        
        tgt = target_acts[name]
        x_hat = reconstructions[name]
        
        # Compute FVU loss
        total_variance = t.var(tgt, dim=0).sum()
        residual_variance = t.var(tgt - x_hat, dim=0).sum()
        fvu_loss = residual_variance / total_variance
        layer = int(name.split('_')[1])
        total_loss = total_loss + fvu_loss * 2**(9-3*layer)
        losses[name] = fvu_loss.item()
        
    return total_loss, losses, reconstructions

def train_scae_suite(
    buffer,
    loss_type: str = "mse",
    base_lr: float = 2e-4,
    steps: Optional[int] = None,
    submodule_configs: Optional[Dict[str, SubmoduleConfig]]=None,
    connections: Optional[Dict[str, Dict[str, t.Tensor]]] = None,
    save_steps: Optional[int] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    use_wandb: bool = False,
    repo_id_in: Optional[str] = None,
    repo_id_out: Optional[str] = None,
    dtype: t.dtype = t.float32,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    wandb_project_name: str = "scae",
    lr_decay_start_proportion: float = 0.8,
    ln_final: Optional[t.nn.Module] = None,
    unembed: Optional[t.nn.Module] = None,
):
    """
    Train a Sparse Connected Autoencoder Suite.
    
    Args:
        buffer: Dataset iterator providing training data
        submodule_configs: Dictionary mapping names to SubmoduleConfigs for fresh initialization
        connections: Optional sparse connections between modules
        steps: Number of training steps
        save_steps: Steps between checkpoints (None for no intermediate saves)
        save_dir: Directory to save checkpoints and final model
        log_steps: Steps between logging (None for no logging)
        use_wandb: Whether to use Weights & Biases logging
        repo_id_in: Optional HuggingFace repo ID to load pretrained model from
        repo_id_out: Optional HuggingFace repo ID to upload trained models to
        dtype: Data type for model parameters
        device: Device to train on
        seed: Random seed for reproducibility
        wandb_project_name: Name of the wandb project
        loss_type: Type of loss to use ("mse" or "ce")
        base_lr: Base learning rate
        lr_decay_start_proportion: When to start learning rate decay
        ln_final: Final layer norm for CE loss (required if loss_type="ce")
        unembed: Unembedding matrix for CE loss (required if loss_type="ce")
    """
    if loss_type not in ["mse", "ce"]:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'mse' or 'ce'")
        
    if loss_type == "ce":
        if any(x is None for x in [ln_final, unembed]):
            raise ValueError("ln_final and unembed are required when loss_type='ce'")
            
    device = device or ('cuda' if t.cuda.is_available() else 'cpu')
    
    # Set random seed if provided
    if seed is not None:
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    
    # Move CE components to device if needed
    if loss_type == "ce":
        ln_final.to(device)
        unembed.to(device)
    
    # Initialize or load pretrained suite
    if repo_id_in is None:
        suite = SCAESuite(
            submodule_configs=submodule_configs,
            dtype=dtype,
            device=device,
            connections=connections
        )
        config_dict = {
            "submodule_configs": {
                name: asdict(cfg) for name, cfg in submodule_configs.items()
            },
            "is_pretrained": False
        }
    else:      
        suite = SCAESuite.from_pretrained(
            repo_id=repo_id_in,
            connections=connections,
            device=device,
            dtype=dtype,
        )
        config_dict = {
            "submodule_configs": {
                name: asdict(config) for name, config in suite.configs.items()
            },
            "is_pretrained": True
        }
    print("W_dec attn_0:", suite.aes['attn_0'].decoder.weight.shape, suite.aes['attn_0'].decoder.weight.dtype)
    print("W_dec mlp_0:", suite.aes['mlp_0'].decoder.weight.shape)
    print(suite.aes.keys())
    
    # Initialize optimizer and scheduler
    optimizer, lrs = initialize_optimizers(suite, base_lr)
    scheduler = get_lr_scheduler(optimizer, steps, lr_decay_start_proportion)
    
    # Add remaining config information
    config_dict.update({
        "base_lr": base_lr,
        "lr_decay_start_proportion": lr_decay_start_proportion,
        "steps": steps,
        "dtype": str(dtype),
        "buffer_config": {
            "ctx_len": buffer.ctx_len,
            "refresh_batch_size": buffer.refresh_batch_size,
            "out_batch_size": buffer.out_batch_size,
            "ce_batch_size": buffer.ce_batch_size,
        },
        "loss_type": loss_type
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
            project=wandb_project_name,
            config=config_dict,
        )
    
    # Training loop
    from itertools import count
    pbar = tqdm(range(steps)) if steps is not None else count()
    
    for step in pbar:
        if steps is not None and step >= steps:
            break
            
        # Ensure decoder norms are unit
        for ae in suite.aes.values():
            ae.set_decoder_norm_to_unit_norm()
            
        optimizer.zero_grad()
        
        if loss_type == "mse":
            batch = next(buffer)
            initial_acts, input_acts, target_acts, layernorm_scales = batch
            loss, losses, _ = compute_mse_loss(
                suite=suite,
                initial_acts=initial_acts,
                input_acts=input_acts,
                target_acts=target_acts,
                layernorm_scales=layernorm_scales
            )
            
            # Log metrics if requested
            if log_steps is not None and step % log_steps == 0 and use_wandb:
                wandb.log({
                    **{f"loss/fvu/{name}": value for name, value in losses.items()},
                    **{f"lr/{name}": param_group['lr'] 
                       for name, param_group in zip(suite.aes.keys(), optimizer.param_groups)},
                    "loss/total": loss.item()
                }, step=step)
                
        else:  # ce loss
            acts, tokens = buffer.get_seq_activations()
            tokens = tokens.to(device)
            initial_acts, input_acts, target_acts, layernorm_scales = acts
            reconstructions = suite.pruned_forward(
                initial_acts=initial_acts,
                inputs=input_acts,
                layernorm_scales=layernorm_scales,
                return_topk=False
            )
            loss = suite.get_ce_loss(
                tokens=tokens,
                initial_acts=initial_acts,
                reconstructions=reconstructions,
                ln_final=ln_final,
                unembed=unembed
            )
            
            # Log metrics if requested
            if log_steps is not None and step % log_steps == 0 and use_wandb:
                wandb.log({
                    "loss/ce": loss.item(),
                    **{f"lr/{name}": param_group['lr'] 
                       for name, param_group in zip(suite.aes.keys(), optimizer.param_groups)}
                }, step=step)
        
        # Backward pass and optimization
        loss.backward()
        for ae in suite.aes.values():
            if ae.decoder.weight.grad is not None:
                t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                ae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        # Save checkpoints if requested
        if save_steps is not None and step % save_steps == 0 and save_dir is not None:
            checkpoint_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            t.save(
                {
                    'suite_state': suite.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'step': step,
                },
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
            
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
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
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
    if repo_id_out is not None:
        try:
            from huggingface_hub import HfApi
            
            print(f"\nUploading models to HuggingFace repo: {repo_id_out}")
            api = HfApi()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                json.dump(config_dict, f, indent=4)
                f.flush()
                api.upload_file(
                    path_or_fileobj=f.name,
                    path_in_repo="config.json",
                    repo_id=repo_id_out,
                    repo_type="model",
                )
            
            with tempfile.NamedTemporaryFile(suffix='.pt') as f:
                t.save({
                    'suite_state': suite.state_dict(),
                    'step': step if step is not None else -1,
                }, f.name)
                api.upload_file(
                    path_or_fileobj=f.name,
                    path_in_repo="checkpoint.pt",
                    repo_id=repo_id_out,
                    repo_type="model",
                )
            
            if connections is not None:
                with tempfile.NamedTemporaryFile(suffix='.pt') as f:
                    t.save(connections, f.name)
                    api.upload_file(
                        path_or_fileobj=f.name,
                        path_in_repo="connections.pt",
                        repo_id=repo_id_out,
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
    
    return suite, optimizer, scheduler