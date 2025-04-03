import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableMask(nn.Module):
    """
    Learnable binary mask using the Hard Concrete distribution
    (the binary equivalent of Gumbel-Softmax). Allows annealing
    towards a target L0 norm via regularization. Includes option
    for temperature annealing.
    """

    def __init__(
        self,
        n_features_down,
        n_features_up,
        target_l0,
        init_mean=0.0,
        init_std=0.01,
    ):
        """
        Args:
            n_features_down: Dimension 1 size.
            n_features_up: Dimension 2 size.
            target_l0: Target L0 norm.
            init_mean: Mean for initializing log_alpha parameters.
            init_std: Standard deviation for initializing log_alpha parameters.
        """
        super().__init__()
        self.n_features_down = n_features_down
        self.n_features_up = n_features_up

        # Parameters for the Hard Concrete distribution (log_alpha)
        # Initialized near zero for roughly 0.5 probability initially.
        self.log_alpha = nn.Parameter(
            torch.empty(n_features_down, n_features_up).normal_(
                init_mean, init_std
            )
        )

        # Parameters for the stretched sigmoid
        # These are typically fixed constants.
        self.register_buffer("gamma", torch.tensor(-0.1))
        self.register_buffer("zeta", torch.tensor(1.1))
        self.target_l0 = target_l0

    def forward(self, temperature, hard=True):  # Default temp = 2/3
        """
        Sample from the Hard Concrete distribution.

        Args:
            temperature: Controls the discreteness. Lower values -> more discrete.
                         This now directly controls the beta parameter.
            hard: If True, use straight-through estimator for hard binary mask.
                  If False, return the continuous relaxation.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        beta = temperature  # Use the passed temperature directly as beta

        # Sample uniform noise
        eps = 1e-7  # Small constant for numerical stability
        u = torch.rand(self.log_alpha.shape, device=self.log_alpha.device)
        u = torch.clamp(u, eps, 1.0 - eps)  # Avoid log(0)

        # Compute stretched sigmoid input
        s_input = (
            torch.log(u) - torch.log(1.0 - u) + self.log_alpha
        ) / beta  # Use annealed beta

        # Compute continuous relaxation (stretched sigmoid)
        s = torch.sigmoid(s_input)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma

        # Clamp to [0, 1] (binary concrete sample)
        mask_relaxed = torch.clamp(s_stretched, 0.0, 1.0)

        if hard:
            # Binarize using Straight-Through Estimator (STE)
            mask_hard = (mask_relaxed > 0.5).float()
            # STE: Use hard values but pass gradients through the relaxed version
            mask = mask_hard - mask_relaxed.detach() + mask_relaxed
        else:
            mask = mask_relaxed

        return mask

    def l0_regularization(self, temperature):
        """
        Calculate the L0 regularization penalty based on expected sparsity.
        Forces the expected number of non-zero elements towards target_l0.

        Args:
            target_l0: The desired number of non-zero elements (sparsity target).
            temperature: The current temperature (beta) used in the forward pass.
                         Must match the one used if consistency is needed,
                         or can be fixed if regularization target is independent
                         of sampling temperature. It's safer to pass it.

        Returns:
            A scalar tensor representing the L2 penalty.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        beta = temperature  # Use the passed temperature

        # Calculate the probability of each element being non-zero (P(s > 0))
        # This uses the CDF of the Hard Concrete distribution.
        log_alpha = self.log_alpha
        # Use the same beta as in the forward calculation for consistency
        p_nonzero = torch.sigmoid(
            log_alpha - beta * math.log(-self.gamma / self.zeta)
        )

        # Calculate the expected total number of non-zero elements (expected L0 norm)
        expected_l0 = p_nonzero.sum()

        # Calculate the L2 penalty between expected L0 and target L0
        l0_penalty = (expected_l0 - self.target_l0) ** 2

        return l0_penalty
