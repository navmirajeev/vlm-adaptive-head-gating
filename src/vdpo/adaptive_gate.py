import torch
import torch.nn as nn


class AdaptiveHeadGate(nn.Module):
    """
    Lightweight adaptive gate.

    Input:
        36 features:
        9 selected heads ×
        [prior_norm, visual_norm, difference_norm, cosine]

    Output:
        9 weights, one for each selected head.
    """

    def __init__(
        self,
        input_dim=36,
        num_heads=9,
        hidden_dim=64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.network(features)