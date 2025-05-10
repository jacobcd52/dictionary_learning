import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableMask(nn.Module):
    """
    Learnable binary mask using the Hard Concrete distribution
    (the binary equivalent of Gumbel-Softmax). Allows annealing
    towards a target C via regularization. Includes option
    for temperature annealing.
    """

    def __init__(
        self,
        n_features_down,
        n_features_up,
        target_C,
        init_mean=0.0,
        init_std=0.01,
    ):
        """
        Args:
            n_features_down: Dimension 1 size.
            n_features_up: Dimension 2 size.
            target_C: Target num connections per feature.
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
        self.target_C = target_C

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
            # print(f"mask_relaxed: {mask_relaxed.sum()}")
            # Binarize using Straight-Through Estimator (STE)
            mask_hard = (mask_relaxed > 0.5).float()
            # print(f"mask_hard: {mask_hard.sum()}")
            # STE: Use hard values but pass gradients through the relaxed version
            mask = mask_hard - mask_relaxed.detach() + mask_relaxed
        else:
            mask = mask_relaxed
        mask = mask.to(torch.bfloat16)
        return mask

    def mask_loss(self, temperature):
        """
        Calculate the regularization penalty based on expected sparsity.
        Forces the expected number of non-zero elements per row towards target_C.

        Args:
            target_C: The desired number of connections per feature.
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
        # print("log_alpha.mean()", log_alpha.mean())
        # Use the same beta as in the forward calculation for consistency
        p_nonzero = torch.sigmoid(
            log_alpha - beta * math.log(-self.gamma / self.zeta)
        )

        # Calculate the expected total number of non-zero elements per row (expected C)
        expected_p_nonzero = p_nonzero.mean()

        target_p_nonzero = self.target_C / p_nonzero.shape[0]  # this is C / num_features
        # print(f"expected_p_nonzero: {expected_p_nonzero}, target_p_nonzero: {target_p_nonzero}")
        # Calculate the L2 penalty between expected C and target C
        mask_loss = (expected_p_nonzero - target_p_nonzero) ** 2

        return mask_loss


class SimpleBinaryMask(nn.Module):
    """
    Simpler learnable binary mask using sigmoid and Straight-Through Estimator (STE).
    """

    def __init__(
        self,
        n_features_down,
        n_features_up, 
        target_C,
        init_mean=0.0, # this is weirdly important... might indicate that something is set up wrong
        init_std=0.01,
    ):
        """
        Args:
            n_features_down: Dimension 1 size (rows of the mask).
            n_features_up: Dimension 2 size (columns of the mask).
            target_C: Target num connections for each 'n_features_up' feature
                      (i.e., target average column sum of probabilities).
            init_mean: Mean for initializing logits.
            init_std: Standard deviation for initializing logits.
        """
        super().__init__()
        self.n_features_down = n_features_down
        self.n_features_up = n_features_up
        self.target_C = target_C

        # Learnable parameters (logits)
        self.logits = nn.Parameter(
            torch.empty(n_features_down, n_features_up).normal_(
                init_mean, init_std
            )
        )

    def forward(self, temperature, hard=True):
        """
        Sample from the mask.

        Args:
            temperature: Unused in this simple mask, but kept for API compatibility.
            hard: If True, use straight-through estimator for hard binary mask.
                  If False, return the continuous probabilities.
        """
        probs = torch.sigmoid(self.logits)

        if hard:
            # Binarize using Straight-Through Estimator (STE)
            mask_hard = (probs > 0.5).float()
            # STE: Use hard values but pass gradients through the probabilities
            mask = mask_hard - probs.detach() + probs
        else:
            mask = probs
        
        mask = mask.to(torch.bfloat16)
        return mask

    def mask_loss(self, temperature):
        """
        Calculate the regularization penalty based on expected sparsity.
        Forces the expected number of connections for each 'n_features_up'
        feature towards target_C.

        Args:
            temperature: Unused in this simple mask, but kept for API compatibility.
            target_C: The desired number of connections for each 'n_features_up' feature.

        Returns:
            A scalar tensor representing the L1 penalty, scaled.
        """
        probs = torch.sigmoid(self.logits) # (n_features_down, n_features_up)

        # Expected number of connections for each 'n_features_down' feature
        # (summing probabilities over the 'n_features_up' dimension - i.e., row sums)
        expected_connections_to_f_down = probs.sum(dim=1)
        
        # L1 penalty: sum |expected_connections_j - target_C|
        # This calculates the sum of absolute differences between the expected sum of connections
        # for each n_features_up column and the target_C.
        loss_val = (expected_connections_to_f_down - self.target_C).abs().sum()
        
        scaled_loss = loss_val / self.n_features_down 
        
        return scaled_loss



