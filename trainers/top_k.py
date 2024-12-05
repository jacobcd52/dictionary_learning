"""
Implements the SAE training scheme from https://arxiv.org/abs/2406.04093.
Significant portions of this code have been copied from https://github.com/EleutherAI/sae/blob/main/sae
"""

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from typing import List

from config import DEBUG
from dictionary import Dictionary
from trainers.trainer import SAETrainer


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class AutoEncoderTopK(Dictionary, nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the AutoEncoderTopK class docstring in that branch.

    With some additional effort, you can train a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x: t.Tensor, return_topk: bool = False, use_sparse_connections=False):

        if use_sparse_connections:
            # split forward pass per feature
            # x has shape [batch, feature, d]
            assert x.shape[-2] == self.dict_size
            preact_BF = einops.einsum(
                x - self.b_dec,
                self.encoder.weight, # TODO need data here?
                "... f d, f d -> ... f",
            ) + self.encoder.bias
        else:
            # vanilla forward pass
            # x has shape [batch, d]
            preact_BF = self.encoder(x - self.b_dec)

        post_relu_feat_acts_BF = nn.functional.relu(preact_BF)
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.weight.dtype).eps
        norm = t.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )

    def from_pretrained(path, k: int, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class TrainerTopK(SAETrainer):
    """
    Top-K SAE training scheme.
    """

    def __init__(
        self,
        dict_class=AutoEncoderTopK,
        activation_dim=512,
        dict_size=64 * 512,
        k=100,
        auxk_alpha=1 / 32,  # see Appendix A.2
        decay_start=24000,  # when does the lr decay start
        steps=30000,  # when when does training end
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="AutoEncoderTopK",
        submodule_name=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        self.wandb_name = wandb_name
        self.steps = steps
        self.k = k
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialise autoencoder
        self.ae = dict_class(activation_dim, dict_size, k)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
        scale = dict_size / (2**14)
        self.lr = 2e-4 / scale**0.5
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000

        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))

        def lr_fn(step):
            if step < decay_start:
                return 1.0
            else:
                return (steps - step) / (steps - decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Training parameters
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)

        # Log the effective L0, i.e. number of features actually used, which should a constant value (K)
        # Note: The standard L0 is essentially a measure of dead features for Top-K SAEs)
        self.logging_parameters = ["effective_l0", "dead_features"]
        self.effective_l0 = -1
        self.dead_features = -1

    def loss(self, x, step=None, logging=False):
        # Run the SAE
        f, top_acts, top_indices = self.ae.encode(x, return_topk=True)
        x_hat = self.ae.decode(f)

        # Measure goodness of reconstruction
        e = x_hat - x
        total_variance = (x - x.mean(0)).pow(2).sum(0)

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        # Compute dead feature mask based on "number of tokens since fired"
        dead_mask = (
            self.num_tokens_since_fired > self.dead_feature_threshold
            if self.auxk_alpha > 0
            else None
        ).to(f.device)
        self.dead_features = int(dead_mask.sum())

        # If dead features: Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = t.where(dead_mask[None], f, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(f)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.ae.decode(auxk_acts_BF)
            auxk_loss = (e_hat - e).pow(2)  # .sum(0)
            auxk_loss = scale * t.mean(auxk_loss / total_variance)
        else:
            auxk_loss = x_hat.new_tensor(0.0)

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = auxk_loss.sum(dim=-1).mean()
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()},
            )

    def update(self, step, x):
        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            self.ae.b_dec.data = median

        # Make sure the decoder is still unit-norm
        self.ae.set_decoder_norm_to_unit_norm()

        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.ae.remove_gradient_parallel_to_decoder_directions()

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "TrainerTopK",
            "dict_class": "AutoEncoderTopK",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }


class TrainerSCAE(SAETrainer):
    def __init__(
            self,
            dict_class=AutoEncoderTopK,
            activation_dims={},
            dict_sizes={},
            ks={},
            submodules={},
            important_features={},
            pretrained_info=None,
            model_config=None,
            auxk_alpha=0,
            connection_sparsity_coeff=0,
            use_sparse_connections=True,
            decay_start=24000,
            steps=30000,
            seed=None,
            device=None,
            wandb_name="SCAE",
            dtype=t.bfloat16, 
        ):
        super().__init__(seed)
        
        # Store dtype
        self.dtype = dtype
        
        assert set(activation_dims.keys()) == set(dict_sizes.keys()) == set(ks.keys()) == set(submodules.keys())
        self.submodule_names = list(submodules.keys())
        self.submodules = submodules
        self.n_autoencoders = len(submodules)
        self.use_sparse_connections = use_sparse_connections

        if pretrained_info is not None:
            assert model_config is not None, "model_config must be provided when using pretrained SAEs"
            expected_modules = set()
            for layer in range(model_config.n_layer):
                expected_modules.add(f"mlp_{layer}")
                expected_modules.add(f"attn_{layer}")
            assert set(pretrained_info.keys()) == expected_modules

        self.wandb_name = wandb_name
        self.steps = steps
        self.ks = ks
        
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialize autoencoders
        if pretrained_info is not None:
            self.aes = t.nn.ModuleDict()
            for name in self.submodule_names:
                repo_info = pretrained_info[name]
                ae = self._load_pretrained_sae(
                    repo_info,
                    activation_dims[name],
                    dict_sizes[name],
                    ks[name]
                )
                ae = ae.to(dtype=self.dtype)
                self.aes[name] = ae
        else:
            self.aes = t.nn.ModuleDict({
                name: dict_class(activation_dims[name], dict_sizes[name], ks[name]).to(dtype=self.dtype)
                for name in self.submodule_names
            })
        
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.aes.to(self.device)
        
        # Move important_features to correct device and dtype
        move_dict_to_device = lambda d: {k: v.to(device=self.device, dtype=t.long) for k, v in d.items()}
        self.important_features = {
            down_name: move_dict_to_device(important_features[down_name])
            for down_name in important_features.keys()
        }
        

        self.lrs = {name: 2e-4 / (size / 2**14)**0.5 for name, size in dict_sizes.items()}
        self.auxk_alpha = auxk_alpha
        self.connection_sparsity_coeff = connection_sparsity_coeff
        self.dead_feature_threshold = 10_000_000

        self.optimizer = t.optim.Adam([
            {'params': ae.parameters(), 'lr': self.lrs[name]}
            for name, ae in self.aes.items()
        ], betas=(0.9, 0.999))

        def lr_fn(step):
            if step < decay_start:
                return 1.0
            else:
                return (steps - step) / (steps - decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Training parameters
        self.num_tokens_since_fired = {
            name: t.zeros(size, dtype=t.long, device=device)
            for name, size in dict_sizes.items()
        }

        self.logging_parameters = ["effective_l0s", "dead_features"]
        self.effective_l0s = {name: -1 for name in self.submodule_names}
        self.dead_features = {name: -1 for name in self.submodule_names}

    def get_approx_input(self, name, input_acts, vanilla_feature_acts):
        """Compute approximated input for a single autoencoder."""
        input_acts_rep = einops.repeat(input_acts, "... d -> ... f d", f=vanilla_feature_acts[name].shape[-1])
        if name.startswith('attn'):
            # Currently we don't approximate the attention input
            return input_acts_rep
        
        elif name.startswith('mlp'):
            current_layer = int(name.split('_')[1])
            approx_input = t.zeros_like(input_acts_rep) # shape [batch, f_down, d]
            
            for up_name, up_feature_ids in self.important_features[name].items():   
                # upstream_feature_ids has shape [f_down, C]               
                upstream_layer = int(up_name.split('_')[1])
                
                if upstream_layer >= current_layer: # TODO attn -> mlp same layer
                    continue

                print("up_name", up_name, "current_name", name)
                up_feature_acts = vanilla_feature_acts[up_name] # shape [batch, f_up]
                decoder = self.important_decoders[up_name] # shape [f_down, C, d]
                
                chunk_size = 4096  # TODO: make this a parameter
                num_groups = up_feature_ids.shape[0]
                
                # chunk the downstream feature IDs to reduce memory usage
                for chunk_start in range(0, num_groups, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_groups)
                    up_feature_ids_chunk = up_feature_ids[chunk_start:chunk_end] # shape [chunk_size, C]
                    expanded_indices = up_feature_ids_chunk.unsqueeze(0).expand(up_feature_acts.shape[0], -1, -1)
                    up_feature_acts_chunk = t.gather(up_feature_acts.unsqueeze(1).expand(-1, up_feature_ids_chunk.shape[0], -1), 
                                        dim=2, 
                                        index=expanded_indices) # shape [batch, chunk_size, C]
                    decoder_chunk = decoder[chunk_start:chunk_end] # shape [chunk_size, C, d]

                    print("1")
                    approx_input_chunk = einops.einsum(
                        up_feature_acts_chunk,
                        decoder_chunk,
                        "batch chunk C, chunk C d -> batch chunk d"
                        )
                    print("2")
                    approx_input = approx_input.clone()
                    print("3")
                    approx_input[:, chunk_start:chunk_end] = approx_input[:, chunk_start:chunk_end] + approx_input_chunk
                    print("4")
                    

        else:
            raise ValueError(f"Invalid submodule name: {name}")
            
        return approx_input

    def update(self, step, input_acts: dict, target_acts: dict):
        if step == 0:
            for name, x in input_acts.items():
                median = geometric_median(x)
                self.aes[name].b_dec.data = median

        for ae in self.aes.values():
            ae.set_decoder_norm_to_unit_norm()

        # Precompute important decoder weights
        self.important_decoders = {}
        for down_name, features in self.important_features.items():
            if down_name.startswith('mlp'):
                for up_name, up_features in self.important_features[down_name].items():
                    # up_features has shape [f_down, C]
                    f_down, C = up_features.shape
                    up_full_decoder = self.aes[up_name].decoder.weight # shape [d, f_up]
                    d, f_up = up_full_decoder.shape
                    up_features_expanded = up_features.unsqueeze(-1)
                    up_full_decoder_expanded = up_full_decoder.unsqueeze(0).unsqueeze(0)
                    result = t.gather(up_full_decoder_expanded.expand(f_down, C, -1, -1), 
                                        dim=-1,
                                        index=up_features_expanded.unsqueeze(-2).expand(-1, -1, d, -1))
                    self.important_decoders[up_name] = result.squeeze(-1) # shape [f_down, C, d]

        input_acts = {name: x.to(self.device) for name, x in input_acts.items()}
        target_acts = {name: x.to(self.device) for name, x in target_acts.items()}
        
        self.optimizer.zero_grad()
        total_loss = 0
        feature_acts = {}  # Store feature activations for use in approx inputs
        
        # First pass: vanilla forward and backward
        for name in self.submodule_names:
            x = input_acts[name]
            tgt = target_acts[name]
            ae = self.aes[name]
            
            # Vanilla forward pass
            f, top_acts, top_indices = ae.encode(x, return_topk=True)
            x_hat = ae.decode(f)
            feature_acts[name] = f.detach()  # Store for approx inputs

            # Update tracking stats
            self.effective_l0s[name] = top_acts.size(1)
            num_tokens_in_step = x.size(0)
            did_fire = t.zeros_like(self.num_tokens_since_fired[name], dtype=t.bool)
            did_fire[top_indices.flatten()] = True
            self.num_tokens_since_fired[name] += num_tokens_in_step
            self.num_tokens_since_fired[name][did_fire] = 0

            # Compute vanilla losses
            e = x_hat - tgt
            l2_loss = e.pow(2).sum(dim=-1).mean()
            
            # Handle dead features
            auxk_loss = x_hat.new_tensor(0.0)
            if self.auxk_alpha > 0:
                dead_mask = self.num_tokens_since_fired[name] > self.dead_feature_threshold
                dead_mask = dead_mask.to(f.device)
                self.dead_features[name] = int(dead_mask.sum())
                
                if (num_dead := int(dead_mask.sum())) > 0:
                    k_aux = x.shape[-1] // 2
                    scale = min(num_dead / k_aux, 1.0)
                    k_aux = min(k_aux, num_dead)
                    total_variance = (tgt - tgt.mean(0)).pow(2).sum(0)

                    auxk_latents = t.where(dead_mask[None], f, -t.inf)
                    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
                    auxk_buffer_BF = t.zeros_like(f)
                    auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
                    e_hat = ae.decode(auxk_acts_BF)
                    auxk_loss = scale * t.mean((e_hat - e).pow(2) / total_variance)

            # Vanilla backward pass
            total_loss_ae = l2_loss + self.auxk_alpha * auxk_loss
            print("calling backward on vanilla FP, name=", name)
            total_loss_ae.backward()
            
            # Process gradients before next forward pass
            t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae.remove_gradient_parallel_to_decoder_directions()
            
            total_loss += total_loss_ae.detach()

        # Second pass: approx forward and backward (if using sparse connections)
        if self.use_sparse_connections:
            for name in self.submodule_names:
                x = input_acts[name]
                tgt = target_acts[name]
                ae = self.aes[name]
                
                # Get approximated input
                approx_x = self.get_approx_input(name, x, feature_acts)
                
                # Approx forward pass
                f_approx, _, _ = ae.encode(approx_x, return_topk=True, use_sparse_connections=True)
                x_hat = ae.decode(f_approx)
                
                # Compute approx losses
                e = x_hat - tgt
                l2_loss = e.pow(2).sum(dim=-1).mean()
                connection_loss = (f_approx - feature_acts[name]).pow(2).sum(dim=-1).mean()
                
                # Approx backward pass
                total_loss_ae = l2_loss + self.connection_sparsity_coeff * connection_loss
                print("calling backward on approx FP, name=", name)
                total_loss_ae.backward()
                
                # Process gradients
                t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                ae.remove_gradient_parallel_to_decoder_directions()
                
                total_loss += total_loss_ae.detach()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return total_loss.item()
        
    def evaluate_varexp_batch(
        self,
        input_acts: dict,
        target_acts: dict,
        use_sparse_connections: bool = None,
        normalize_batch: bool = False,
        device: str = None,
    ) -> dict:
        """Evaluate reconstruction quality metrics like variance explained."""
        if device is None:
            device = self.device
        if use_sparse_connections is None:
            use_sparse_connections = self.use_sparse_connections
            
        input_acts = {name: x.to(device) for name, x in input_acts.items()}
        target_acts = {name: x.to(device) for name, x in target_acts.items()}
        
        with t.no_grad():
            metrics = {}

            # First get vanilla features
            feature_acts = {}
            for name in self.submodule_names:
                x = input_acts[name]
                f, _, _ = self.aes[name].encode(x, return_topk=True)
                feature_acts[name] = f

            # Get approx features if using sparse connections
            if use_sparse_connections:
                feature_acts_final = {}
                for name in self.submodule_names:
                    x = input_acts[name]
                    approx_x = self.get_approx_input(name, x, feature_acts, x.size(0))
                    f_approx, _, _ = self.aes[name].encode(approx_x, return_topk=True, use_sparse_connections=True)
                    feature_acts_final[name] = f_approx
            else:
                feature_acts_final = feature_acts

            # Compute reconstruction metrics
            for name in self.submodule_names:
                x = input_acts[name]
                tgt = target_acts[name]
                if normalize_batch:
                    scale = (self.aes[name].activation_dim ** 0.5) / x.norm(dim=-1).mean()
                    x = x * scale
                    tgt = tgt * scale
                
                f = feature_acts_final[name]
                x_hat = self.aes[name].decode(f)
                
                if normalize_batch:
                    x_hat = x_hat / scale
                
                l2_loss = t.linalg.norm(tgt - x_hat, dim=-1).mean()
                l1_loss = f.norm(p=1, dim=-1).mean()
                l0 = (f != 0).float().sum(dim=-1).mean()
                frac_alive = t.flatten(f, start_dim=0, end_dim=1).any(dim=0).sum() / self.aes[name].dict_size
                
                tgt_normed = tgt / t.linalg.norm(tgt, dim=-1, keepdim=True)
                x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
                cossim = (tgt_normed * x_hat_normed).sum(dim=-1).mean()
                
                l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(tgt, dim=-1)).mean()
                
                total_variance = t.var(tgt, dim=0).sum()
                residual_variance = t.var(tgt - x_hat, dim=0).sum()
                frac_variance_explained = (1 - residual_variance / total_variance)
                
                x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
                x_dot_x_hat = (tgt * x_hat).sum(dim=-1)
                relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()
                
                metrics[name] = {
                    'l2_loss': l2_loss.item(),
                    'l1_loss': l1_loss.item(),
                    'l0': l0.item(),
                    'frac_alive': frac_alive.item(),
                    'frac_variance_explained': frac_variance_explained.item(),
                    'cossim': cossim.item(),
                    'l2_ratio': l2_ratio.item(),
                    'relative_reconstruction_bias': relative_reconstruction_bias.item()
                }
            
            return metrics

    def evaluate_ce_batch(
            self,
            model,
            text,
            max_len=None,
            use_sparse_connections: bool = None,
            normalize_batch: bool = False,
            device: str = None,
            tracer_args={'use_cache': False, 'output_attentions': False}
        ) -> dict:
        """
        Evaluate cross entropy loss when patching in reconstructed activations for all submodules.
        Returns per-submodule statistics.
        """
        if device is None:
            device = self.device
        if use_sparse_connections is None:
            use_sparse_connections = self.use_sparse_connections

        if max_len is None:
            invoker_args = {}
        else:
            invoker_args = {"truncation": True, "max_length": max_len}

        # First get unmodified logits
        with model.trace(text, invoker_args=invoker_args):
            logits_original = model.output.save()
        logits_original = logits_original.value

        # Get all activations in one pass
        saved_activations = {}
        with model.trace(text, **tracer_args, invoker_args=invoker_args):
            for name, (submodule, io) in self.submodules.items():
                if io in ['in', 'in_and_out']:
                    x = submodule.input
                elif io == 'out':
                    x = submodule.output
                else:
                    raise ValueError(f"Invalid value for io: {io}")
                
                if normalize_batch:
                    scale = (self.aes[name].activation_dim ** 0.5) / x.norm(dim=-1).mean()
                    x = x * scale
                
                saved_activations[name] = {
                    'x': x.save(),
                    'io': io,
                    'scale': scale if normalize_batch else 1.0
                }

        # Handle tuples outside trace context
        for name, saved in saved_activations.items():
            if type(saved['x'].value) == tuple:
                saved['x'] = saved['x'].value[0]
            saved_activations[name] = saved

        # If using sparse connections, first get vanilla features
        vanilla_feature_acts = {}
        if use_sparse_connections:
            for name, saved in saved_activations.items():
                x = saved['x'].to(device)
                f, _, _ = self.aes[name].encode(x.view(-1, x.shape[-1]), return_topk=True)
                vanilla_feature_acts[name] = f.view(x.shape[:-1] + (-1,))

        # Get reconstructions
        reconstructions = {}
        for name, saved in saved_activations.items():
            x = saved['x'].to(device)
            
            if use_sparse_connections:
                # Get approximated input using vanilla features
                approx_x = self.get_approx_input(name, x, vanilla_feature_acts, x.size(0))
                f, _, _ = self.aes[name].encode(approx_x, return_topk=True, use_sparse_connections=True)
                x_hat = self.aes[name].decode(f)
            else:
                x_hat = self.aes[name](x.view(-1, x.shape[-1])).view(x.shape)
            
            if normalize_batch:
                x_hat = x_hat / saved['scale']
            
            reconstructions[name] = x_hat.to(model.dtype)

        # Format logits consistently
        try:
            logits_original = logits_original.logits
        except:
            pass

        # Get tokens for loss computation
        if isinstance(text, t.Tensor):
            tokens = text
        else:
            try:
                with model.trace(text, **tracer_args, invoker_args=invoker_args):
                    input = model.input.save()
                tokens = input.value[1]['input_ids']
            except:
                tokens = input.value[1]['input']

        # Set up loss function
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            loss_kwargs = {'ignore_index': model.tokenizer.pad_token_id}
        else:
            loss_kwargs = {}

        # Compute original loss once
        loss_original = t.nn.CrossEntropyLoss(**loss_kwargs)(
            logits_original[:, :-1, :].reshape(-1, logits_original.shape[-1]),
            tokens[:, 1:].reshape(-1)
        ).item()

        # Compute per-submodule metrics
        results = {}

        for name, (submodule, io) in self.submodules.items():
            # Run model with just this reconstruction patched in
            with model.trace(text, **tracer_args, invoker_args=invoker_args):
                x_hat = reconstructions[name]
                submodule.input = x_hat
                
                if io in ['out', 'in_and_out']:
                    if "attn" in name:
                        submodule.output = (x_hat,)
                    elif "mlp" in name:
                        submodule.output = x_hat
                    else:
                        raise ValueError(f"Invalid submodule name: {name}")
                elif io == 'in':
                    submodule.input = x_hat
                
                logits_reconstructed = model.output.save()
            
            # Run model with just this submodule zeroed
            with model.trace(text, **tracer_args, invoker_args=invoker_args):
                if io in ['in', 'in_and_out']:
                    x = submodule.input
                    submodule.input = t.zeros_like(x)
                if io in ['out', 'in_and_out']:
                    x = submodule.output
                    if "attn" in name:
                        submodule.output = (t.zeros_like(x[0]),)
                    elif "mlp" in name:
                        submodule.output = t.zeros_like(x)
                    else:
                        raise ValueError(f"Invalid submodule name: {name}")

                logits_zero = model.output.save()

            # Format logits
            try:
                logits_reconstructed = logits_reconstructed.value.logits
                logits_zero = logits_zero.value.logits
            except:
                logits_reconstructed = logits_reconstructed.value
                logits_zero = logits_zero.value

            # Compute losses for this submodule
            loss_reconstructed = t.nn.CrossEntropyLoss(**loss_kwargs)(
                logits_reconstructed[:, :-1, :].reshape(-1, logits_reconstructed.shape[-1]),
                tokens[:, 1:].reshape(-1)
            ).item()
            
            loss_zero = t.nn.CrossEntropyLoss(**loss_kwargs)(
                logits_zero[:, :-1, :].reshape(-1, logits_zero.shape[-1]),
                tokens[:, 1:].reshape(-1)
            ).item()

            if loss_original - loss_zero != 0:
                frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
            else:
                frac_recovered = 0.0
            
            results[name] = {
                'loss_original': loss_original,
                'loss_reconstructed': loss_reconstructed,
                'loss_zero': loss_zero,
                'frac_recovered': frac_recovered
            }

        return results
          
    def _load_pretrained_sae(self, repo_info, activation_dim, dict_size, k):
        """
        Load a pretrained Sparse Autoencoder from a Hugging Face repository.
        
        Args:
            repo_info (dict): Dictionary containing:
                - repo_id (str): Hugging Face repo ID (e.g. "organization/repo-name")
                - filename (str): Name of the .pt file containing the SAE weights
            activation_dim (int): Expected dimension of the activation space
            dict_size (int): Expected size of the learned dictionary
            k (int): Number of features to use for reconstruction
            
        Returns:
            AutoEncoderTopK: Loaded autoencoder with pretrained weights
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "The huggingface_hub package is required to load pretrained SAEs. "
                "Please install it with `pip install huggingface_hub`."
            )
        
        # Validate repo_info format
        if not isinstance(repo_info, dict) or 'repo_id' not in repo_info or 'filename' not in repo_info:
            raise ValueError(
                "repo_info must be a dictionary containing 'repo_id' and 'filename' keys. "
                f"Got: {repo_info}"
            )
        
        # Download the weights file from Hugging Face
        try:
            weights_path = hf_hub_download(
                repo_id=repo_info['repo_id'],
                filename=repo_info['filename'],
                repo_type="model"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download weights from Hugging Face repo {repo_info['repo_id']}: {str(e)}"
            )
        
        # Load the state dict
        try:
            state_dict = t.load(weights_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load weights file {weights_path}: {str(e)}")
        
        # Validate the loaded weights
        expected_shapes = {
            'encoder.weight': (dict_size, activation_dim),
            'decoder.weight': (activation_dim, dict_size),
            'b_dec': (activation_dim,)
        }
        
        for key, expected_shape in expected_shapes.items():
            if key not in state_dict:
                raise ValueError(f"Missing key {key} in loaded state dict")
            if state_dict[key].shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {key}. "
                    f"Expected {expected_shape}, got {state_dict[key].shape}"
                )
        
        # Initialize a new autoencoder with the correct dimensions
        ae = AutoEncoderTopK(activation_dim, dict_size, k)
        
        # Load the weights
        ae.load_state_dict(state_dict)
        
        return ae

    @property
    def config(self):
        return {
            "trainer_class": "MultiTrainerTopK",
            "dict_class": "AutoEncoderTopK",
            "lrs": self.lrs,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dims": {name: ae.activation_dim for name, ae in self.aes.items()},
            "dict_sizes": {name: ae.dict_size for name, ae in self.aes.items()},
            "ks": self.ks,
            "device": self.device,
            "submodule_names": self.submodule_names,
            "wandb_name": self.wandb_name,
        }