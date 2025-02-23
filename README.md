Overview

This is a fork of Sam Marks and Aaron Mueller's Dictionary Learning library, modified for training Sparsely Connected Autoencoders (SCAEs).

Given SAEs at various points in a model, we'd like each feature to depend on only a small, fixed set of upstream features. We operationalize this as follows:

1. Train attn_out SAEs and mlp_in -> mlp_out transcoders at each layer, using standard TopK SAE training.
2. Use some method to select, for each feature $F$, a small set of upstream features that will be "connected" to $F$. Here, we choose the top C upstream features that had the largest average direct-path contribution to the downstream one over a large dataset.
3. Define a *pruned forward pass* of the suite, where the input to a given MLP transcoder neuron is the sum of contributions from all upstream features it connects to (plus a contribution from the token embedding).
4. Finetune the suite to do good reconstruction (or get good downstream CE) using the pruned forward pass.

Usage:

I pretrained a suite of small (expansion=4) SAEs on TinyStories-33M. The HF repo is jacobcd52/TinyStories-33M_suite_4. 

1. gdown https://drive.google.com/drive/folders/1kxpZkpdL2Yhs3xnv2zhbDw-OnlBkzZcx?usp=sharing (this downloads a folder containing top_connections dictionaries for various values of the number C of connections per feature. I got these connections by taking, for each feature, the top C upstream features with highest average contribution over ~1M tokens.)
2. wandb login YOUR_API_KEY
2. python run_finetuning.py (or run as a notebook). You can change C and other hparams.

Code Structure

Compared to Sam Marks' repo, the main new pieces of code are:

TODO

TODOs

- Sparse connections to embed/unembed?
- BOS activations might be v large and screw up our attribution estimates.