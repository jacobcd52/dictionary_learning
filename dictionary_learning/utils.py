import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleBinaryMask(nn.Module):
    """
    Simpler learnable binary mask using sigmoid and Straight-Through Estimator (STE).
    """

    def __init__(
        self,
        n_features_down,
        n_features_up, 
        target_C,
        init_mean=0.1, # this is weirdly important... might indicate that something is set up wrong
        init_std=0.001,
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



