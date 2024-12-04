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
            print("x", x.shape)
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
        self.important_features = {
            name: features.to(device=self.device, dtype=self.dtype)
            for name, features in important_features.items()
        }
        
        # Precompute important decoder weights
        self.important_decoders = {}
        for name, features in self.important_features.items():
            if name.startswith('mlp'):
                decoder = self.aes[name].decoder.weight
                self.important_decoders[name] = decoder

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

    def get_approx_input(self, name, x, feature_acts, batch_size):
        """Compute approximated input for a single autoencoder."""
        if not name.startswith('mlp'):
            return x

        current_layer = int(name.split('_')[1])
        approx_input = t.zeros_like(x)
        
        for upstream_name, upstream_features in self.important_features.items():
            if not upstream_name.startswith('mlp'):
                continue
                    
            upstream_layer = int(upstream_name.split('_')[1])
            if upstream_layer >= current_layer:
                continue
            
            upstream_acts = feature_acts[upstream_name]
            decoder = self.important_decoders[upstream_name]
            
            chunk_size = 2048
            num_groups = upstream_features.shape[0]
            
            for chunk_start in range(0, num_groups, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_groups)
                chunk_features = upstream_features[chunk_start:chunk_end]
                chunk_acts = upstream_acts[:, chunk_features].sum(dim=-1)
                chunk_decoder = decoder[:, chunk_features[:, 0]]
                approx_input += t.matmul(chunk_acts, chunk_decoder.t())
        
        return approx_input

    def update(self, step, input_acts: dict, target_acts: dict):
        if step == 0:
            for name, x in input_acts.items():
                median = geometric_median(x)
                self.aes[name].b_dec.data = median

        for ae in self.aes.values():
            ae.set_decoder_norm_to_unit_norm()

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
                approx_x = self.get_approx_input(name, x, feature_acts, x.size(0))
                
                # Approx forward pass
                f_approx, _, _ = ae.encode(approx_x, return_topk=True, use_sparse_connections=True)
                x_hat = ae.decode(f_approx)
                
                # Compute approx losses
                e = x_hat - tgt
                l2_loss = e.pow(2).sum(dim=-1).mean()
                connection_loss = (f_approx - feature_acts[name]).pow(2).sum(dim=-1).mean()
                
                # Approx backward pass
                total_loss_ae = l2_loss + self.connection_sparsity_coeff * connection_loss
                total_loss_ae.backward()
                
                # Process gradients
                t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                ae.remove_gradient_parallel_to_decoder_directions()
                
                total_loss += total_loss_ae.detach()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return total_loss.item()
        
    def evaluate_varexp(
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
            metrics = {
                'submodule_metrics': {},
                'total_metrics': {
                    'l2_loss': 0.0,
                    'l1_loss': 0.0,
                    'l0': 0.0,
                    'frac_alive': 0.0,
                    'frac_variance_explained': 0.0,
                    'cossim': 0.0,
                    'l2_ratio': 0.0,
                    'relative_reconstruction_bias': 0.0
                }
            }

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
                
                metrics['submodule_metrics'][name] = {
                    'l2_loss': l2_loss.item(),
                    'l1_loss': l1_loss.item(),
                    'l0': l0.item(),
                    'frac_alive': frac_alive.item(),
                    'frac_variance_explained': frac_variance_explained.item(),
                    'cossim': cossim.item(),
                    'l2_ratio': l2_ratio.item(),
                    'relative_reconstruction_bias': relative_reconstruction_bias.item()
                }
                
                for metric_name in metrics['total_metrics'].keys():
                    metrics['total_metrics'][metric_name] += metrics['submodule_metrics'][name][metric_name]

            # Average metrics
            num_submodules = len(self.submodule_names)
            metrics['total_metrics'] = {
                k: v / num_submodules 
                for k, v in metrics['total_metrics'].items()
            }
            
            return metrics

    def evaluate_patched_ce(
            self,
            model,
            text,
            max_len=None,
            use_sparse_connections: bool = None,
            normalize_batch: bool = False,
            device: str = None,
            tracer_args={'use_cache': False, 'output_attentions': False}
        ) -> dict:
            if max_len is None:
                invoker_args = {}
            else:
                invoker_args = {"truncation": True, "max_length": max_len}

            # Initialize dictionaries
            feature_acts = {}
            saved_acts = {}
            reconstructions = {}  # For demonstration only in this debug version

            with t.no_grad():
                print("\n=== First Trace ===")
                with model.trace(text, **tracer_args, invoker_args=invoker_args):
                    print("Inside trace - before saving:")
                    print(f"model.output type: {type(model.output)}")
                    
                    logits_original = model.output.save()
                    print("\nSaved full output")
                    
                    for name, (submod, io) in self.submodules.items():
                        print(f"\nProcessing module {name} with io {io}")
                        if io == 'in':
                            saved_acts[name] = submod.input[0].save()
                        elif io == 'out':
                            saved_acts[name] = submod.output.save()
                        elif io == 'in_and_out':
                            saved_acts[name] = submod.input[0].save()
                
                print("\nCreating dummy reconstructions...")
                for name in saved_acts:
                    x = saved_acts[name].value
                    if isinstance(x, tuple):
                        x = x[0]
                    reconstructions[name] = t.zeros_like(x)
                
                print("\n=== Reconstruction Trace ===")
                # intervene with x_hat
                with model.trace(text, **tracer_args, invoker_args=invoker_args):
                    for name, (submod, io) in self.submodules.items():
                        x_hat = reconstructions[name]
                        print(f"\nProcessing {name} with io={io}")
                        
                        if io == 'in':
                            x = submod.input[0]
                            print(f"Input x type: {type(x)}")
                            x_saved = x.save()
                            x = x_saved.value
                            print(f"Input value type: {type(x)}")
                            
                            if normalize_batch:
                                scale = (self.aes[name].activation_dim ** 0.5) / x.norm(dim=-1).mean()
                                x_hat = x_hat / scale
                                
                            if type(x) == tuple:
                                print("Setting input with [:]")
                                submod.input[0][:] = x_hat
                            else:
                                print("Setting input directly")
                                submod.input = x_hat
                                
                        elif io == 'out':
                            x = submod.output
                            print(f"Output x type: {type(x)}")
                            x_saved = x.save()
                            x = x_saved.value
                            print(f"Output value type: {type(x)}")
                            
                            if normalize_batch:
                                scale = (self.aes[name].activation_dim ** 0.5) / x.norm(dim=-1).mean()
                                x_hat = x_hat / scale
                                
                            if type(x) == tuple:
                                print("Setting output as tuple")
                                submod.output = (x_hat,)
                            else:
                                print("Setting output directly")
                                submod.output = x_hat
                                
                        elif io == 'in_and_out':
                            x = submod.input[0]
                            print(f"Input x type: {type(x)}")
                            x_saved = x.save()
                            x = x_saved.value
                            print(f"Input value type: {type(x)}")
                            
                            if normalize_batch:
                                scale = (self.aes[name].activation_dim ** 0.5) / x.norm(dim=-1).mean()
                                x_hat = x_hat / scale
                                
                            print("Setting output directly")
                            submod.output = x_hat
                            
                        else:
                            raise ValueError(f"Invalid value for io: {io}")
                    
                    print("\nSaving reconstructed output")
                    logits_reconstructed = model.output.save()
                
                print("\nOutside reconstruction trace:")
                print(f"logits_reconstructed.value type: {type(logits_reconstructed.value)}")

                return {}
        
        
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