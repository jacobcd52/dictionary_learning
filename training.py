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

def get_module(model):
    return model.module if isinstance(model, t.nn.DataParallel) else model

def initialize_optimizers(suite, base_lr):
    # If suite is wrapped, use the underlying module.
    module = suite.module if isinstance(suite, t.nn.DataParallel) else suite
    lrs = {
        name: base_lr / (module.n_features / 2**14)**0.5
        for name in module.aes.keys()
    }
    optimizer = t.optim.Adam([
        {'params': ae.parameters(), 'lr': lrs[name]}
        for name, ae in module.aes.items()
    ], betas=(0.9, 0.999))
    return optimizer, lrs

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

def get_lr_scheduler(optimizer, steps, lr_decay_start_proportion, lr_warmup_end_proportion):
    """Create learning rate scheduler with linear decay"""
    def lr_fn(step):
        if step < lr_warmup_end_proportion * steps:
            return step / (lr_warmup_end_proportion * steps)
        elif step < lr_decay_start_proportion * steps:
            return 1.0
        else:
            return (steps - step) / (steps - lr_decay_start_proportion * steps)
    
    return t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

def train_scae_suite(
    buffer,
    model_name,
    k: Optional[int] = None,
    expansion: Optional[int] = None, 
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
    wandb_run_name: Optional[str] = None,
    lr_decay_start_proportion: float = 0.8,
    lr_warmup_end_proportion: float = 0.0,
    stagger_steps: Optional[int] = None,
    track_dead_features: bool = True,
    dead_feature_threshold: int = 10_000_000,
    auxk_alpha: float = 0.0,  # Added parameter for auxk loss weight
):
    """
    Train a Sparse Connected Autoencoder Suite.
    
    Args:
        buffer: Dataset iterator providing training data
        model_name: Name of the transformer model to use
        k: Number of features to select for each autoencoder (required if repo_id_in not provided)
        expansion: Factor to multiply model.cfg.d_model by to get n_features (required if repo_id_in not provided)
        loss_type: Type of loss to use, either "mse" or "ce"
        base_lr: Base learning rate for training
        steps: Total number of training steps
        connections: Optional dictionary specifying sparse connections
        save_steps: Interval for saving checkpoints
        save_dir: Directory to save checkpoints and final model
        log_steps: Interval for logging metrics
        use_wandb: Whether to use Weights & Biases for logging
        repo_id_in: HuggingFace repository ID to load a pretrained model from
        repo_id_out: HuggingFace repository ID to upload the trained model to
        dtype: Data type for model parameters
        device: Device to use for training
        seed: Random seed for reproducibility
        wandb_project_name: Name of the Weights & Biases project
        wandb_run_name: Name of the Weights & Biases run
        lr_decay_start_proportion: Proportion of training steps after which learning rate decay starts
        lr_warmup_end_proportion: Proportion of training steps after which learning rate warmup ends
        stagger_steps: If provided, number of steps to wait before introducing each new loss term
        track_dead_features: Whether to track and log dead features
        dead_feature_threshold: Number of tokens after which a feature is considered dead if it hasn't fired
        auxk_alpha: Weight for auxk loss term (0.0 disables auxk loss)
    """
    if loss_type not in ["mse", "ce"]:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'mse' or 'ce'")
        
    device = device or ('cuda' if t.cuda.is_available() else 'cpu')
    
    # Set random seed if provided
    if seed is not None:
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    
    # Initialize or load pretrained suite
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    
    if repo_id_in is None:
        # New training initialization
        if k is None or expansion is None:
            raise ValueError("k and expansion must be provided when not loading a pretrained model")
        
        n_features = expansion * model.cfg.d_model
        suite = SCAESuite(
            model=model,
            k=k,
            n_features=n_features,
            connections=connections,
            dtype=dtype,
            device=device,
        )
        config_dict = {
            "k": k,
            "expansion": expansion,
            "n_features": n_features,
            "is_pretrained": False
        }
    else:      
        # Simplified loading - let SCAESuite.from_pretrained handle connections
        suite = SCAESuite.from_pretrained(
            repo_id=repo_id_in,
            model=model,
            connections=connections,  # Let class handle override logic
            device=device,
            dtype=dtype
        )
        
        # Calculate expansion from loaded n_features
        config_dict = {
            "k": suite.k,
            "expansion": suite.n_features // model.cfg.d_model,
            "n_features": suite.n_features,
            "is_pretrained": True
        }

    # --- Wrap with DataParallel if multiple GPUs are available ---
    print(f"Using {t.cuda.device_count()} GPUs for training")
    if device.startswith("cuda") and t.cuda.device_count() > 1:
        suite = t.nn.DataParallel(suite)
    
    # Initialize optimizer and scheduler
    optimizer, lrs = initialize_optimizers(suite, base_lr)
    scheduler = get_lr_scheduler(optimizer, steps, lr_decay_start_proportion, lr_warmup_end_proportion)
    
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
        "loss_type": loss_type,
        "stagger_steps": stagger_steps,  # Add stagger_steps to config
        "track_dead_features": track_dead_features,  # Add dead feature tracking to config
        "dead_feature_threshold": dead_feature_threshold,  # Add dead feature threshold to config
        "auxk_alpha": auxk_alpha,  # Add auxk_alpha to config
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
            name=wandb_run_name,
        )
    
    # Define helper function to determine if a component should be included in the loss
    def get_loss_scale(name, step):
        if stagger_steps is None:
            return 1.0  # No staggering, use all components
        
        # Determine the expected ordering of components for staggering
        # First, extract layer number and type (attn or mlp)
        parts = name.split('_')
        layer_num = int(parts[1])
        is_attn = 'attn' in name
        
        # Calculate the position in the sequence
        pos = 2 * layer_num + (0 if is_attn else 1)
        
        # If the current step is beyond the step where this component should be included
        if step >= pos * stagger_steps:
            return 1.0  # Include this component
        else:
            return 0.0  # Zero out this component
    
    # Helper function to compute AuxK loss
    def compute_auxk_loss(name, features, reconstruction, target, dead_mask, module, device, dtype):
        """
        Compute AuxK loss for the given features and target.
        
        Args:
            name: Name of the autoencoder
            features: Feature activations (can be sparse or dense)
            reconstruction: Current reconstruction
            target: Target activations to reconstruct
            dead_mask: Boolean mask indicating dead features
            module: SCAE module reference
            device: Device to use
            dtype: Data type to use
        
        Returns:
            AuxK loss value
        """
        # Skip if no dead features
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            return t.tensor(0.0, device=device, dtype=dtype)
        
        # Heuristic: use k_aux as half the input dimension
        input_dim = target.shape[-1]
        k_aux = input_dim // 2
        
        # Scale based on number of dead features
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)
        
        # Get shape information and flatten if needed
        if len(features.shape) == 3:  # [batch, seq, features]
            batch_size, seq_len, _ = features.shape
            features_flat = features.reshape(-1, features.shape[-1])  # [batch*seq, features]
            target_flat = target.reshape(-1, target.shape[-1])  # [batch*seq, dim]
            reconstruction_flat = reconstruction.reshape(-1, reconstruction.shape[-1])  # [batch*seq, dim]
        else:  # [batch*seq, features]
            features_flat = features
            target_flat = target
            reconstruction_flat = reconstruction
        
        # Compute residual
        residual_flat = target_flat - reconstruction_flat
        
        # Get the autoencoder
        ae = module.aes[name]
        
        # Create full non-sparse feature representation if needed
        dict_size = ae.dict_size if hasattr(ae, 'dict_size') else ae.code_size
        
        if features_flat.shape[-1] != dict_size:  # If we have sparse features
            # Create full feature tensor
            full_features = t.zeros(
                features_flat.shape[0], dict_size,
                device=features_flat.device, dtype=features_flat.dtype
            )
            
            # Get indices of active features
            if hasattr(features_flat, 'indices') and hasattr(features_flat, 'values'):
                # If features is a sparse tensor
                full_features.scatter_(-1, features_flat.indices, features_flat.values)
            else:
                # If features contains top-k values
                _, top_indices = t.topk(features_flat, k=module.k, dim=-1)
                for i in range(features_flat.shape[0]):
                    full_features[i, top_indices[i]] = features_flat[i, range(module.k)]
        else:
            full_features = features_flat
        
        # Expand dead mask to match features
        dead_mask_expanded = dead_mask.unsqueeze(0).expand(full_features.shape[0], -1)
        
        # Apply dead mask to features
        auxk_features = t.where(
            dead_mask_expanded, 
            full_features, 
            t.tensor(-float('inf'), device=device, dtype=dtype)
        )
        
        # Get top k_aux dead features
        if k_aux > 0:
            auxk_vals, auxk_indices = auxk_features.topk(k_aux, dim=-1, sorted=False)
            
            # Create sparse representation for auxk
            auxk_features_sparse = t.zeros_like(full_features)
            batch_indices = t.arange(full_features.shape[0], device=device).unsqueeze(1).expand(-1, k_aux)
            auxk_features_sparse[batch_indices.flatten(), auxk_indices.flatten()] = auxk_vals.flatten()
            
            # Decode to get residual prediction
            residual_pred = ae.decode(auxk_features_sparse)
            
            # Compute AuxK loss (MSE on residual prediction normalized by total variance)
            total_variance = t.var(target_flat, dim=0).sum()
            auxk_loss = (residual_pred - residual_flat).pow(2).sum() / total_variance
            
            return scale * auxk_loss
        else:
            return t.tensor(0.0, device=device, dtype=dtype)
    
    # Initialize dead feature tracking if enabled
    dead_feature_tracking = {}
    if track_dead_features:
        # Get the module reference
        module = suite.module if isinstance(suite, t.nn.DataParallel) else suite
        
        # Initialize counters for each autoencoder
        for name, ae in module.aes.items():
            # Create a tensor to track tokens since each feature fired
            dict_size = ae.code_size if hasattr(ae, 'code_size') else ae.dict_size
            dead_feature_tracking[name] = {
                'num_tokens_since_fired': t.zeros(dict_size, dtype=t.long, device=device),
                'dead_features': 0  # Count of dead features
            }
    
    # Training loop
    from itertools import count
    pbar = tqdm(range(steps)) if steps is not None else count()
    
    for step in pbar:
        if steps is not None and step >= steps:
            break
            
        optimizer.zero_grad()
        
        if loss_type == "mse":
            cache, tokens = next(buffer)
            module = get_module(suite)
            
            # Call forward_pruned with return_features=True to get activations
            reconstructions, pruned_features = module.forward_pruned(cache, return_features=True)
            
            # Track dead features and compute auxk loss components if enabled
            auxk_losses = {}
            
            if track_dead_features:
                # Get batch size for this step
                num_tokens_in_step = tokens.flatten().shape[0]
                
                # Update counters for each autoencoder
                for name, features in pruned_features.items():
                    # Skip if not using this component yet in staggered training
                    if stagger_steps is not None:
                        scale = get_loss_scale(name, step)
                        if scale == 0.0:
                            continue
                    
                    # Get top-k indices
                    if features.shape[-1] == module.k:  # If features are already sparse
                        # These are the indices of active features
                        if hasattr(features, 'indices'):
                            top_indices = features.indices
                        else:
                            # Find non-zero indices
                            top_indices = t.nonzero(features, as_tuple=True)[1].reshape(features.shape[0], -1)
                    else:
                        # Get top-k indices
                        _, top_indices = t.topk(features, k=module.k, dim=-1)
                    
                    # Update the counters
                    did_fire = t.zeros_like(dead_feature_tracking[name]['num_tokens_since_fired'], dtype=t.bool)
                    did_fire[top_indices.flatten()] = True
                    dead_feature_tracking[name]['num_tokens_since_fired'] += num_tokens_in_step
                    dead_feature_tracking[name]['num_tokens_since_fired'][did_fire] = 0
                    
                    # Compute the number of dead features
                    dead_mask = dead_feature_tracking[name]['num_tokens_since_fired'] > dead_feature_threshold
                    dead_feature_tracking[name]['dead_features'] = int(dead_mask.sum().item())
                    
                    # Compute AuxK loss if enabled and there are dead features
                    if auxk_alpha > 0 and dead_feature_tracking[name]['dead_features'] > 0:
                        layer = int(name.split('_')[1])
                        if 'attn' in name:  
                            target = cache[f'blocks.{layer}.hook_attn_out']
                        elif 'mlp' in name:
                            target = cache[f'blocks.{layer}.hook_mlp_out']
                        else:
                            raise RuntimeError(f"Invalid layer name: {name}")
                        
                        # Compute AuxK loss
                        auxk_losses[name] = compute_auxk_loss(
                            name=name,
                            features=features,
                            reconstruction=reconstructions[name],
                            target=target,
                            dead_mask=dead_mask,
                            module=module,
                            device=device,
                            dtype=dtype
                        )
            
            # Compute MSE loss using FVU
            total_loss = 0
            losses = {}
            scales = {}
            
            for name, recon in reconstructions.items():
                layer = int(name.split('_')[1])
                if 'attn' in name:  
                    target = cache[f'blocks.{layer}.hook_attn_out']
                elif 'mlp' in name:
                    target = cache[f'blocks.{layer}.hook_mlp_out']
                else:
                    raise RuntimeError(f"Invalid layer name: {name}")
                
                # Compute FVU loss
                total_variance = t.var(target, dim=0).sum()
                residual_variance = t.var(target - recon, dim=0).sum()
                fvu_loss = residual_variance / total_variance
                
                # Apply scaling factor based on step and component
                scale = get_loss_scale(name, step)
                
                # Add auxk loss if available
                if name in auxk_losses:
                    component_loss = fvu_loss + auxk_alpha * auxk_losses[name]
                else:
                    component_loss = fvu_loss
                
                total_loss = total_loss + scale * component_loss
                
                losses[name] = fvu_loss.item()
                scales[name] = scale
            
            loss = total_loss
           
            # Log metrics if requested
            if log_steps is not None and step % log_steps == 0 and use_wandb:
                log_dict = {
                    **{f"FVU/{name}": value for name, value in losses.items()},
                    **{f"Scale/{name}": value for name, value in scales.items()},
                    "loss": loss.item()
                }
                
                # Add auxk loss metrics to logging
                if auxk_alpha > 0:
                    for name, auxk_loss in auxk_losses.items():
                        log_dict[f"AuxKLoss/{name}"] = auxk_loss.item()
                
                # Add dead feature metrics to logging if enabled
                if track_dead_features:
                    for name, tracking in dead_feature_tracking.items():
                        log_dict[f"DeadFeatures/{name}"] = tracking['dead_features']
                        # Calculate percentage of dead features
                        total_features = len(tracking['num_tokens_since_fired'])
                        if total_features > 0:
                            log_dict[f"DeadFeaturesPct/{name}"] = (tracking['dead_features'] / total_features) * 100
                
                wandb.log(log_dict, step=step)
                
        else:  # ce loss
            cache, tokens = next(buffer)
            tokens = tokens.to(device)
            module = get_module(suite)
            
            # Call forward_pruned with return_features=True to get activations
            reconstructions, pruned_features = module.forward_pruned(cache, return_features=True)
            
            # Track dead features and compute auxk loss if enabled
            auxk_loss_total = 0
            
            if track_dead_features:
                # Get batch size for this step
                num_tokens_in_step = tokens.size(0)
                
                # Update counters for each autoencoder
                for name, features in pruned_features.items():
                    # Skip if not using this component yet in staggered training
                    if stagger_steps is not None:
                        scale = get_loss_scale(name, step)
                        if scale == 0.0:
                            continue
                    
                    # Get top-k indices
                    if features.shape[-1] == module.k:  # If features are already sparse
                        # These are the indices of active features
                        if hasattr(features, 'indices'):
                            top_indices = features.indices
                        else:
                            # Find non-zero indices
                            top_indices = t.nonzero(features, as_tuple=True)[1].reshape(features.shape[0], -1)
                    else:
                        # Get top-k indices
                        _, top_indices = t.topk(features, k=module.k, dim=-1)
                    
                    # Update the counters
                    did_fire = t.zeros_like(dead_feature_tracking[name]['num_tokens_since_fired'], dtype=t.bool)
                    did_fire[top_indices.flatten()] = True
                    dead_feature_tracking[name]['num_tokens_since_fired'] += num_tokens_in_step
                    dead_feature_tracking[name]['num_tokens_since_fired'][did_fire] = 0
                    
                    # Compute the number of dead features
                    dead_mask = dead_feature_tracking[name]['num_tokens_since_fired'] > dead_feature_threshold
                    dead_feature_tracking[name]['dead_features'] = int(dead_mask.sum().item())
                    
                    # Calculate AuxK loss contribution if enabled and there are dead features
                    # Note: For CE loss, a complete implementation would require additional model-specific code
                    # This is a placeholder that can be expanded if needed
                    if auxk_alpha > 0 and dead_feature_tracking[name]['dead_features'] > 0:
                        # Placeholder for CE-based AuxK loss implementation
                        pass
            
            # Get CE loss
            ce_loss = module.get_ce_loss(cache, reconstructions, tokens)
            
            # Add auxk contribution if any
            loss = ce_loss + auxk_loss_total
            
            # Log metrics if requested
            if log_steps is not None and step % log_steps == 0 and use_wandb:
                log_dict = {}
                
                # Calculate CE-diff
                original_loss = suite.model(tokens, return_type="loss")
                log_dict.update({
                    "ce_diff": loss.item() - original_loss.item(),
                    "ce": loss.item(),
                })
                
                if auxk_alpha > 0:
                    log_dict["auxk_loss"] = auxk_loss_total.item()
                
                # Add dead feature metrics to logging if enabled
                if track_dead_features:
                    for name, tracking in dead_feature_tracking.items():
                        log_dict[f"DeadFeatures/{name}"] = tracking['dead_features']
                        # Calculate percentage of dead features
                        total_features = len(tracking['num_tokens_since_fired'])
                        if total_features > 0:
                            log_dict[f"DeadFeaturesPct/{name}"] = (tracking['dead_features'] / total_features) * 100
                
                wandb.log(log_dict, step=step)
        
        # Backward pass and optimization
        loss.backward()
        module = suite.module if isinstance(suite, t.nn.DataParallel) else suite
        for ae in module.aes.values():
            if ae.decoder.weight.grad is not None:
                t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Update the description with dead feature information if enabled
        if track_dead_features:
            total_dead_features = sum(tracking['dead_features'] for tracking in dead_feature_tracking.values())
            pbar.set_description(f"Loss: {loss.item():.4f}, Dead Features: {total_dead_features}")
        else:
            pbar.set_description(f"Loss: {loss.item():.4f}")
        
        # Save checkpoints if requested
        if save_steps is not None and step % save_steps == 0 and save_dir is not None:
            checkpoint_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Unwrap suite if needed
            state_dict = suite.module.state_dict() if isinstance(suite, t.nn.DataParallel) else suite.state_dict()
            
            # Save dead feature tracking data if enabled
            if track_dead_features:
                # Convert tensor values to serializable format
                serializable_tracking = {}
                for name, tracking in dead_feature_tracking.items():
                    serializable_tracking[name] = {
                        'num_tokens_since_fired': tracking['num_tokens_since_fired'].cpu().tolist(),
                        'dead_features': tracking['dead_features']
                    }
                
                t.save(
                    {
                        'suite_state': state_dict,
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'step': step,
                        'dead_feature_tracking': serializable_tracking
                    },
                    os.path.join(checkpoint_dir, "checkpoint.pt")
                )
            else:
                t.save(
                    {
                        'suite_state': state_dict,
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
        
        # Unwrap suite if needed for final save
        state_dict = suite.module.state_dict() if isinstance(suite, t.nn.DataParallel) else suite.state_dict()
        
        # Save dead feature tracking data if enabled
        if track_dead_features:
            # Convert tensor values to serializable format
            serializable_tracking = {}
            for name, tracking in dead_feature_tracking.items():
                serializable_tracking[name] = {
                    'num_tokens_since_fired': tracking['num_tokens_since_fired'].cpu().tolist(),
                    'dead_features': tracking['dead_features']
                }
            
            t.save(
                {
                    'suite_state': state_dict,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'step': step if step is not None else -1,
                    'dead_feature_tracking': serializable_tracking
                },
                os.path.join(final_dir, "checkpoint.pt")
            )
        else:
            t.save(
                {
                    'suite_state': state_dict,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'step': step if step is not None else -1,
                },
                os.path.join(final_dir, "checkpoint.pt")
            )
    
    # Upload to HuggingFace if requested
    if repo_id_out is not None:
        module = suite.module if isinstance(suite, t.nn.DataParallel) else suite
        module.upload_to_hf(repo_id_out)
    
    if use_wandb:
        wandb.finish()
    
    return suite, optimizer, scheduler