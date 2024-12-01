"""
Implements the SAE training scheme from https://arxiv.org/abs/2406.04093.
Significant portions of this code have been copied from https://github.com/EleutherAI/sae/blob/main/sae
"""

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from typing import List

from ..config import DEBUG
from ..dictionary import Dictionary
from ..trainers.trainer import SAETrainer


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

    def encode(self, x: t.Tensor, return_topk: bool = False):
        if len(x.shape) == 2:
            # vanilla forward pass
            # x has shape [batch, d]
            preact_BF = self.encoder(x - self.b_dec)

        elif len(x.shape) == 3:
            # split forward pass per feature
            # x has shape [batch, feature, d]
            print("x", x.shape)
            assert x.shape[1] == self.dict_size
            preact_BF = einops.einsum(
                x - self.b_dec,
                self.encoder.weight, # TODO need data here?
                "b f d, f d -> b f",
            ) + self.encoder.bias

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
        activation_dims={},  # Dict mapping submodule names to dimensions
        dict_sizes={},      # Dict mapping submodule names to dictionary sizes
        ks={},             # Dict mapping submodule names to k values
        submodules={},      # Dict mapping submodule names to (submodule, io) tuples
        important_features = {},
        pretrained_info=None,  # Dict mapping submodule names to HF repo/filename info
        model_config=None,     # Required if using pretrained_info
        auxk_alpha=0,
        connection_sparsity_coeff=0,
        decay_start=24000,
        steps=30000,
        seed=None,
        device=None,
        wandb_name="SCAE",
    ):
        super().__init__(seed)
        
        # Validate inputs
        assert set(activation_dims.keys()) == set(dict_sizes.keys()) == set(ks.keys()) == set(submodules.keys())
        self.submodule_names = list(submodules.keys())
        self.submodules = submodules
        self.n_autoencoders = len(submodules)
        self.timings = {}  # Store timing information

        # Validate pretrained info if provided
        if pretrained_info is not None:
            assert model_config is not None, "model_config must be provided when using pretrained SAEs"
            expected_modules = set()
            for layer in range(model_config.n_layer):
                expected_modules.add(f"mlp_{layer}")
                expected_modules.add(f"attn_{layer}")
            assert set(pretrained_info.keys()) == expected_modules, (
                "When using pretrained SAEs, must provide info for all MLPs and attention layers. "
                f"Expected modules: {expected_modules}, got: {set(pretrained_info.keys())}"
            )

        self.wandb_name = wandb_name
        self.steps = steps
        self.ks = ks
        
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialize autoencoders as a ModuleDict
        self.aes = t.nn.ModuleDict()
        
        if pretrained_info is not None:
            # Load pretrained SAEs
            for name in self.submodule_names:
                repo_info = pretrained_info[name]
                self.aes[name] = self._load_pretrained_sae(
                    repo_info,
                    activation_dims[name],
                    dict_sizes[name],
                    ks[name]
                )
        else:
            # Initialize new SAEs
            self.aes = t.nn.ModuleDict({
                name: dict_class(activation_dims[name], dict_sizes[name], ks[name])
                for name in self.submodule_names
            })
        
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.aes.to(self.device)
        
        # Move important_features to the correct device
        self.important_features = {
            name: features.to(self.device) 
            for name, features in important_features.items()
        }
        
        # Precompute important decoder weights
        self.important_decoders = {}
        for name, features in self.important_features.items():
            if name.startswith('mlp'):
                decoder = self.aes[name].decoder.weight  # [activation_dim, num_features]
                self.important_decoders[name] = decoder  # [activation_dim, num_features]

        # Auto-select LRs using 1/sqrt(d) scaling law
        self.lrs = {name: 2e-4 / (size / 2**14)**0.5 for name, size in dict_sizes.items()}
        self.auxk_alpha = auxk_alpha
        self.connection_sparsity_coeff = connection_sparsity_coeff
        self.dead_feature_threshold = 10_000_000

        # Optimizer and scheduler
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

        # Training parameters for each autoencoder
        self.num_tokens_since_fired = {
            name: t.zeros(size, dtype=t.long, device=device)
            for name, size in dict_sizes.items()
        }

        # Logging parameters
        self.logging_parameters = ["effective_l0s", "dead_features"]
        self.effective_l0s = {name: -1 for name in self.submodule_names}
        self.dead_features = {name: -1 for name in self.submodule_names}
    
    def reset_timings(self):
        """Reset all timing measurements"""
        self.timings = {}
    
    def _time_function(self, func_name):
        """Context manager to measure execution time of a function"""
        class TimerCtx:
            def __init__(self, trainer, name):
                self.trainer = trainer
                self.name = name
                
            def __enter__(self):
                self.start = t.cuda.Event(enable_timing=True)
                self.end = t.cuda.Event(enable_timing=True)
                self.start.record()
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end.record()
                t.cuda.synchronize()
                elapsed_time = self.start.elapsed_time(self.end)  # in milliseconds
                
                if self.name not in self.trainer.timings:
                    self.trainer.timings[self.name] = []
                self.trainer.timings[self.name].append(elapsed_time)
                
        return TimerCtx(self, func_name)

    def run_forward_vanilla(self, input_acts: dict, target_acts: dict):
        with self._time_function("vanilla_forward"):
            vanilla_l2_loss = 0
            vanilla_auxk_loss = 0
            vanilla_individual_losses = {}
            vanilla_reconstructions = {}
            vanilla_feature_acts = {}

            for name in self.submodule_names:
                with self._time_function(f"vanilla_forward_{name}"):
                    x = input_acts[name]
                    tgt = target_acts[name]
                    ae = self.aes[name]
                    
                    with self._time_function(f"vanilla_encode_{name}"):
                        f, top_acts, top_indices = ae.encode(x, return_topk=True)
                    
                    with self._time_function(f"vanilla_decode_{name}"):
                        x_hat = ae.decode(f)
                        
                    vanilla_feature_acts[name] = f
                    vanilla_reconstructions[name] = x_hat

                    e = x_hat - tgt
                    total_variance = (tgt - tgt.mean(0)).pow(2).sum(0)

                    self.effective_l0s[name] = top_acts.size(1)

                    num_tokens_in_step = x.size(0)
                    did_fire = t.zeros_like(self.num_tokens_since_fired[name], dtype=t.bool)
                    did_fire[top_indices.flatten()] = True
                    self.num_tokens_since_fired[name] += num_tokens_in_step
                    self.num_tokens_since_fired[name][did_fire] = 0

                    dead_mask = (
                        self.num_tokens_since_fired[name] > self.dead_feature_threshold
                        if self.auxk_alpha > 0
                        else None
                    )
                    if dead_mask is not None:
                        dead_mask = dead_mask.to(f.device)
                        self.dead_features[name] = int(dead_mask.sum())

                    if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
                        k_aux = x.shape[-1] // 2
                        scale = min(num_dead / k_aux, 1.0)
                        k_aux = min(k_aux, num_dead)

                        auxk_latents = t.where(dead_mask[None], f, -t.inf)
                        auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

                        auxk_buffer_BF = t.zeros_like(f)
                        auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

                        e_hat = ae.decode(auxk_acts_BF)
                        auxk_loss = (e_hat - e).pow(2)
                        auxk_loss = scale * t.mean(auxk_loss / total_variance)
                    else:
                        auxk_loss = x_hat.new_tensor(0.0)

                    l2_loss = e.pow(2).sum(dim=-1).mean()
                    auxk_loss = auxk_loss.sum(dim=-1).mean()

                    vanilla_l2_loss += l2_loss
                    vanilla_auxk_loss += auxk_loss

                    vanilla_individual_losses[f"{name}_l2_loss"] = l2_loss.item()
                    vanilla_individual_losses[f"{name}_auxk_loss"] = auxk_loss.item()

        return vanilla_feature_acts, vanilla_l2_loss, vanilla_auxk_loss, vanilla_individual_losses


    def get_approx_inputs(self, input_acts: dict, vanilla_feature_acts: dict):
        with self._time_function("get_approx_inputs"):
            approx_inputs = {}
            
            for name, x in input_acts.items():
                with self._time_function(f"get_approx_inputs_{name}"):
                    if name.startswith('attn'):
                        approx_inputs[name] = x
                    elif name.startswith('mlp'):
                        current_layer = int(name.split('_')[1])
                        batch_size = x.shape[0]
                        
                        approx_input = t.zeros_like(x)
                        
                        for upstream_name, upstream_features in self.important_features.items():
                            if not upstream_name.startswith('mlp'):
                                continue
                                
                            upstream_layer = int(upstream_name.split('_')[1])
                            if upstream_layer >= current_layer:
                                continue
                            
                            upstream_acts = vanilla_feature_acts[upstream_name]
                            
                            batch_indices = t.arange(batch_size, device=upstream_acts.device)[:, None, None]
                            feature_indices = upstream_features[None, :, :]
                            
                            batch_indices = batch_indices.expand(-1, upstream_features.shape[0], upstream_features.shape[1])
                            feature_indices = feature_indices.expand(batch_size, -1, -1)
                            
                            important_feature_acts = upstream_acts[batch_indices, feature_indices]
                            summed_acts = important_feature_acts.sum(dim=-1)
                            
                            approx_input += t.einsum('bn,dn->bd', summed_acts, self.important_decoders[upstream_name])
                        
                        approx_inputs[name] = approx_input
                    else:
                        raise ValueError(f"Unknown submodule type: {name}")
                    
        return approx_inputs

    def run_forward_approx(self, approx_inputs: dict, vanilla_feature_acts: dict, target_acts: dict):
        with self._time_function("approx_forward"):
            approx_l2_loss = 0
            connection_loss = 0
            approx_individual_losses = {}
            approx_reconstructions = {}
            approx_feature_acts = {}        
        
            for name in self.submodule_names:
                with self._time_function(f"approx_forward_{name}"):
                    x = approx_inputs[name]
                    tgt = target_acts[name]
                    f = vanilla_feature_acts[name]
                    ae = self.aes[name]
                    
                    with self._time_function(f"approx_encode_{name}"):
                        f_approx, top_acts, top_indices = ae.encode(x, return_topk=True)
                    
                    with self._time_function(f"approx_decode_{name}"):
                        x_hat = ae.decode(f_approx)
                        
                    approx_feature_acts[name] = f_approx
                    approx_reconstructions[name] = x_hat

                    e = x_hat - tgt
                    l2_loss = e.pow(2).sum(dim=-1).mean()
                    curr_connection_loss = (f_approx - f).pow(2).sum(dim=-1).mean()

                    approx_l2_loss += l2_loss
                    connection_loss += curr_connection_loss

                    approx_individual_losses[f"{name}_approx_l2_loss"] = l2_loss.item()
                    approx_individual_losses[f"{name}_connection_loss"] = curr_connection_loss.item()

        return approx_feature_acts, approx_l2_loss, connection_loss, approx_individual_losses

    def loss(self, input_acts: dict, target_acts: dict, step=None, logging=False):
        """
        Compute the total loss for all autoencoders.
        
        Args:
            xs: Dict mapping submodule names to input tensors
            step: Current training step (optional)
            logging: Whether to return logging information
            
        Returns:
            total_loss if not logging, otherwise returns logging dict
        """
        with self._time_function("total_loss"):
            vanilla_feature_acts, vanilla_l2_loss, vanilla_auxk_loss, vanilla_individual_losses = self.run_forward_vanilla(input_acts, target_acts)
            approx_inputs = self.get_approx_inputs(input_acts, vanilla_feature_acts)
            approx_feature_acts, approx_l2_loss, connection_loss, approx_individual_losses = self.run_forward_approx(approx_inputs, vanilla_feature_acts, target_acts)

            all_individual_losses = {**vanilla_individual_losses, **approx_individual_losses}

            total_loss = vanilla_l2_loss 
            total_loss += self.auxk_alpha * vanilla_auxk_loss 
            total_loss += approx_l2_loss 
            total_loss += self.connection_sparsity_coeff * connection_loss

        if not logging:
            return total_loss
        else:
            # Calculate average timings
            avg_timings = {
                name: sum(times) / len(times) 
                for name, times in self.timings.items()
            }
            
            return {
                'total_loss': total_loss.item(),
                'vanilla_l2_loss': vanilla_l2_loss.item(),
                'vanilla_auxk_loss': vanilla_auxk_loss.item(),
                'approx_l2_loss': approx_l2_loss.item(),
                'connection_loss': connection_loss.item(),
                **all_individual_losses,
                'effective_l0s': self.effective_l0s,
                'dead_features': self.dead_features,
                'timings': avg_timings  # Add timings to logging output
            }

    def update(self, step, input_acts: dict, target_acts: dict):
        # Initialize decoder biases
        if step == 0:
            for name, x in input_acts.items():
                median = geometric_median(x)
                self.aes[name].b_dec.data = median

        # Ensure decoders maintain unit norm
        for ae in self.aes.values():
            ae.set_decoder_norm_to_unit_norm()

        # Move inputs to device
        input_acts = {name: x.to(self.device) for name, x in input_acts.items()}
        target_acts = {name: x.to(self.device) for name, x in target_acts.items()}
        
        # Compute loss and backward pass
        loss = self.loss(input_acts, target_acts, step=step)
        loss.backward()

        # Clip gradients and remove parallel components
        for ae in self.aes.values():
            t.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae.remove_gradient_parallel_to_decoder_directions()

        # Optimization step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return loss.item()

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