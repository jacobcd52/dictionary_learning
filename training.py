"""
Training for Sparsely-Connected AutoEncoder Suite
"""

import json
import os
from queue import Empty
import torch as t
from typing import Union, Dict, Optional, TypeVar
import tempfile
from tqdm.auto import tqdm
import wandb
from transformer_lens import HookedTransformer

from trainers.scae import SCAESuite

def initialize_optimizers(suite, base_lr):
    """Initialize optimizers with custom learning rates based on feature counts"""
    lrs = {
        name: base_lr / (suite.n_features / 2**14)**0.5
        for name in suite.aes.keys()
    }
    optimizer = t.optim.Adam([
        {'params': ae.parameters(), 'lr': lrs[name]}
        for name, ae in suite.aes.items()
    ], betas=(0.9, 0.999))
    return optimizer, lrs

def get_lr_scheduler(optimizer, steps, lr_decay_start_proportion):
    """Create learning rate scheduler with linear decay"""
    def lr_fn(step):
        if step < lr_decay_start_proportion * steps:
            return 1.0
        return (steps - step) / (steps - lr_decay_start_proportion * steps)
    
    return t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

def train_scae_suite(
    buffer,
    model_name,
    k: int,
    n_features: int,
    loss_type: str = "mse",
    base_lr: float = 2e-4,
    steps: Optional[int] = None,
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
):
    """
    Train a Sparse Connected Autoencoder Suite.
    
    Args:
        buffer: Dataset iterator providing training data
        model: TransformerLens model
        k: Number of features to select for each autoencoder
        n_features: Dictionary size for each autoencoder
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
    """
    if loss_type not in ["mse", "ce"]:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'mse' or 'ce'")
        
    device = device or ('cuda' if t.cuda.is_available() else 'cpu')
    
    # Set random seed if provided
    if seed is not None:
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    
    # Initialize or load pretrained suite
    if repo_id_in is None:
        suite = SCAESuite(
            model=HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype),
            k=k,
            n_features=n_features,
            connections=connections,
            dtype=dtype,
            device=device,
        )
        config_dict = {
            "k": k,
            "n_features": n_features,
            "is_pretrained": False
        }
    else:      
        suite = SCAESuite.from_pretrained(
            repo_id=repo_id_in,
            model=HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )
        config_dict = {
            "k": suite.k,
            "n_features": suite.n_features,
            "is_pretrained": True
        }
    
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
            "batch_size": buffer.batch_size,
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
            
        optimizer.zero_grad()
        
        if loss_type == "mse":
            cache, tokens = next(buffer)
            reconstructions = suite.forward_pruned(cache)
            
            # Compute MSE loss using FVU
            total_loss = 0
            losses = {}
            
            for name, recon in reconstructions.items():
                ln_name = 'ln1' if 'attn' in name else 'ln2'
                layer = int(name.split('_')[1])
                target = cache[f'blocks.{layer}.{ln_name}.hook_normalized']
                
                # Compute FVU loss
                total_variance = t.var(target, dim=0).sum()
                residual_variance = t.var(target - recon, dim=0).sum()
                fvu_loss = residual_variance / total_variance
                total_loss = total_loss + fvu_loss * 2**(9-3*layer)
                losses[name] = fvu_loss.item()
            
            loss = total_loss
            
            # Log metrics if requested
            if log_steps is not None and step % log_steps == 0 and use_wandb:
                wandb.log({
                    **{f"fvu/{name}": value for name, value in losses.items()},
                    "loss": loss.item()
                }, step=step)
                
        else:  # ce loss
            cache, tokens = next(buffer)
            tokens = tokens.to(device)
            reconstructions = suite.forward_pruned(cache)
            loss = suite.get_ce_loss(reconstructions, tokens)
            
            # Log metrics if requested
            if log_steps is not None and step % log_steps == 0 and use_wandb:
                wandb.log({
                    "ce": loss.item(),
                }, step=step)
        
        # Backward pass and optimization
        loss.backward()
        for ae in suite.aes.values():
            if ae.decoder.weight.grad is not None:
                t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
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
    
    # Upload to HuggingFace if requested
    if repo_id_out is not None:
        suite.upload_to_hf(repo_id_out)
    
    if use_wandb:
        wandb.finish()
    
    return suite, optimizer, scheduler