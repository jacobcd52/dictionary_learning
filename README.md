**Overview**

This is a fork of Sam Marks' library, adapted for training Sparsely Connected Autoencoders (SCAEs).

Given SAEs at various points in a model, we'd like each feature to depend on only a small, fixed set of upstream features. We operationalize this as follows:

1. Train attn_out SAEs and mlp_in -> mlp_out transcoders at each layer, using standard TopK SAE training.
2. Use some method to select, for each feature $F$, a small set of upstream features that will be "connected" to $F$.
3. Define a *pruned forward pass* of the suite, where the input to a given SAE neuron is the sum of contributions from only the upstream features it connects to (plus a contribution from the token embedding and various biases).
4. Finetune the suite on local reconstruction and/or downstream CE using the pruned forward pass.


**Usage:**

For testing, I pretrained some small (expansion=4) TopK SAEs for Pythia-70M, available at the HF repo jacobcd52/TinyStories-33M_suite_4. You can finetune them as follows:

0. pip install -r requirements.txt
1. gdown --folder 1kxpZkpdL2Yhs3xnv2zhbDw-OnlBkzZcx (this downloads files specifying feature-connectivity graphs, for various values of the number C of connections per feature. I got these connections by taking, for each feature, the top C upstream features with highest average contribution over a large dataset).
2. python run_finetuning.py (or run as a notebook). You'll need to enter a wandb key.

**Code Structure**

The main differences to Sam Marks' code are:

- buffer.py now only contains SimpleBuffer. Calling next(buffer) runs the model on some tokens and outputs an transformer_lens ActivationCache object. The cache has a nice consistent naming system. E.g. cache['blocks.0.hook_resid_pre'] gives the residual stream activation just before the first attention block. This activation has shape [batch, ctx_len, d_model]. Note that the batch and sequence dimensions are never flattened, and activations are not shuffled, unlike usual buffers.
- trainers.scae.py contains the SCAESuite class, which implements the pruned forward pass.
- training.py contains train_scae_suite, which finetunes the pruned FP either on MSE or CE. I haven't yet added an option to use this function for pretraining the suite, but we don't really need this.

TODOs

- Sparse connections to embed/unembed?
- Multi-GPU support
