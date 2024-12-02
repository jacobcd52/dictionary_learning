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
from evaluation import evaluate
from trainers.standard import StandardTrainer


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


def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            # log parameters from training
            log.update({f"{k}": v for k, v in losslog.items()})
            log[f"l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def trainSAE(
    data,
    trainer_configs,
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    steps=None,
    save_steps=None,
    save_dir=None,
    log_steps=None,
    activations_split_by_head=False,
    transcoder=False,
    run_cfg={},
):
    """
    Train SAEs using the given trainers
    """
    trainers = []
    for config in trainer_configs:
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    wandb_processes = []
    log_queues = []

    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break

        # logging
        if log_steps is not None and step % log_steps == 0:
            log_stats(
                trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues
            )

        # saving
        if save_steps is not None and step % save_steps == 0:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:
                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))
                    t.save(
                        trainer.ae.state_dict(),
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                    )

        # training
        for trainer in trainers:
            trainer.update(step, act)

    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()


def trainSCAE(
    buffer,
    trainer_cfg,
    steps=None,
    save_steps=None,
    save_dir=None,
    log_steps=None,
    use_wandb=False,
    hf_repo_id=None,  # New parameter for HuggingFace repo ID
):
    # Convert lists to dictionaries if necessary
    if isinstance(trainer_cfg.get("submodules", []), list):
        submodule_names = trainer_cfg["submodule_names"]
        trainer_cfg["submodules"] = {
            name: (module, io_type) 
            for name, (module, io_type) in zip(submodule_names, trainer_cfg["submodules"])
        }

    # Ensure all dictionary keys match
    required_keys = ["activation_dims", "dict_sizes", "ks", "submodules"]
    dicts = {key: trainer_cfg.get(key, {}) for key in required_keys}

    # Validate all keys are the same
    all_keys = set(dicts["submodules"].keys())
    for key, d in dicts.items():
        if set(d.keys()) != all_keys:
            raise ValueError(f"Mismatched keys in {key}. Expected {all_keys}, got {set(d.keys())}")

    # Initialize trainer
    trainer_class = trainer_cfg.pop("trainer")
    trainer = trainer_class(**trainer_cfg)

    # Create save directory if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        simple_config = {
            "activation_dims": trainer_cfg["activation_dims"],
            "dict_sizes": trainer_cfg["dict_sizes"],
            "ks": trainer_cfg["ks"],
            "layers": trainer_cfg.get("layers", []),
            "lm_name": trainer_cfg.get("lm_name", ""),
            "submodule_names": list(trainer_cfg["submodules"].keys()),
            "connection_sparsity_coeff": trainer_cfg.get("connection_sparsity_coeff", 0),
            "use_sparse_connections": trainer_cfg.get("use_sparse_connections", True),
            "buffer_config": {
                "ctx_len": buffer.ctx_len,
                "refresh_batch_size": buffer.refresh_batch_size,
                "out_batch_size": buffer.out_batch_size,
            }
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(simple_config, f, indent=4)

    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(
            project="sae_training",
            config=simple_config,
        )

    # Training loop
    pbar = tqdm(buffer, total=steps) if steps is not None else tqdm(buffer)
    for step, (input_acts, target_acts) in enumerate(pbar):
        if steps is not None and step >= steps:
            break
                
        # Log statistics
        if log_steps is not None and step % log_steps == 0:
            loss_log = trainer.loss(input_acts, target_acts, step=step, logging=True)
            
            # Create log dictionary
            log_dict = {}
            for key, value in loss_log.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        log_dict[f"{key}_{subkey}"] = subvalue
                else:
                    log_dict[key] = value
            
            if use_wandb:
                wandb.log(log_dict, step=step)
            
            # Update progress bar
            pbar.set_description(f"Loss: {log_dict.get('total_loss', 0):.4f}")
        
        # Save checkpoint
        if save_steps is not None and step % save_steps == 0 and save_dir is not None:
            checkpoint_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            for name, ae in trainer.aes.items():
                t.save(
                    ae.state_dict(),
                    os.path.join(checkpoint_dir, f"ae_{name}.pt")
                )
        
        # Training step
        loss = trainer.update(step, input_acts, target_acts)

    # Save final models
    if save_dir is not None:
        final_dir = os.path.join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        for name, ae in trainer.aes.items():
            t.save(
                ae.state_dict(),
                os.path.join(final_dir, f"ae_{name}.pt")
            )

    # Upload to HuggingFace if repo_id is provided
    if hf_repo_id is not None:
        try:
            from huggingface_hub import HfApi, upload_file
            import tempfile
            
            print(f"\nUploading models to HuggingFace repo: {hf_repo_id}")
            
            # Create HuggingFace API client
            api = HfApi()
            
            # Upload configuration
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                json.dump(simple_config, f, indent=4)
                f.flush()
                api.upload_file(
                    path_or_fileobj=f.name,
                    path_in_repo="config.json",
                    repo_id=hf_repo_id,
                    repo_type="model",
                )
            
            # Upload each autoencoder
            for name, ae in trainer.aes.items():
                with tempfile.NamedTemporaryFile(suffix='.pt') as f:
                    t.save(ae.state_dict(), f.name)
                    api.upload_file(
                        path_or_fileobj=f.name,
                        path_in_repo=f"ae_{name}.pt",
                        repo_id=hf_repo_id,
                        repo_type="model",
                    )
                print(f"Uploaded ae_{name}.pt")
            
            print("Successfully uploaded all models to HuggingFace!")
            
        except ImportError:
            print("Warning: huggingface_hub package not found. Skipping HuggingFace upload.")
            print("To upload to HuggingFace, install with: pip install huggingface_hub")
        except Exception as e:
            print(f"Error uploading to HuggingFace: {str(e)}")

    if use_wandb:
        wandb.finish()

    return trainer