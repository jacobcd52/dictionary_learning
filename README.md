Overview

This is a fork of Sam Marks and Aaron Mueller's Dictionary Learning library, modified for training Sparsely Connected Autoencoders (SCAEs).

Given SAEs at various points in a model, we'd like each feature to depend on only a small, fixed set of upstream features. We operationalize this as follows:

1. Train attn_out SAEs and mlp_in -> mlp_out transcoders at each layer, using standard TopK SAE training.
2. Use some method to select, for each feature $F$, a small set of upstream features that will be "connected" to $F$. E.g. we might choose the top 100 upstream features that had the largest average direct-path contribution to the downstream one over a large dataset.
3. Define a *pruned forward pass* of the suite, where the input to a given MLP transcoder neuron is the sum of contributions from all upstream features it connects to (plus a contribution from the token embedding).
4. Finetune the suite to do good reconstruction using the pruned forward pass.

Note: attn_out SAEs are currently leaf nodes, in the sense that we do not prune the connections going into them. They still change during finetuning though, since they are encouraged to learn features that the downstream MLP transcoders can connect to sparsely.

Note: the loss is currently vanilla_MSE + coeff * pruned_MSE. Might also want MSE(vanilla_feature_acts - pruned_feature_acts) or something. Or just pruned_MSE. This choice seems important and subtle.

Usage:

I already trained a suite of TopK SAEs on gpt2-small, available at the HF repo jacobcd52/gpt2_suite_folded_ln. Here's how to finetune them:

0. pip install -r requirements.txt
1. gdown https://drive.google.com/uc?id=1IW3d-t0EuaLR4rSM7WQQqTPESp2nFGJH (this downloads a file that tells us which features are connected. I got these connections by taking, for each feature, the top 100 upstream features with highest average contribution over a large dataset)
2. python run_finetuning.py (or run as a notebook). You'll need to paste a wandb key. To tune hparams, you need to go into the file and mnaually change stuff; most important params are defined near the top.

Code Structure

Compared to Sam Marks' repo, the main new pieces of code are:

- AllActivationBuffer (in buffer.py)
- SCAESuite (in trainers/scae.py). 
- TrainerSCAESuite (in trainers/scae.py)
- train_scae_suite (in training.py)
- find_top_connections.py

(TODO: write descriptions of these classes)

TODOs

- Currently we just optimize the vanilla-active feature for good approx. It's too expensive to optimize *all* downstream features, but it might make sense to randomly select a small number of non-active features per input and optimize these.
- Sparse connections to embed/unembed?
- Decide on 3 losses
- Rename inputs/src to be consistent
- More informative errors (e.g. "if no repo_id_in provided, you must provide submodule_configs" etc)
- Implement patched CE and add logging step. Need to worry about layernorm, and remove_bos.
- Multi-GPU support
- Move SCAESuite evaluation methods to standalone functions in evaluate.py.