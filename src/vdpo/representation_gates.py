import torch
import torch.nn as nn


class HeadOnlyGate(nn.Module):
    """
    Adaptive gate based only on the representation of one attention head.

    Input:
        h_i ∈ R^128

    Output:
        g_i ∈ [0, 1]

    Architecture:
        128 -> 8 -> 1 -> sigmoid
    """

    def __init__(
        self,
        head_dim=128,
        hidden_dim=8,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, head_output):
        """
        head_output:
            (..., 128)

        returns:
            (..., 1)
        """
        return self.network(head_output)


class ContextOnlyGate(nn.Module):
    """
    Adaptive gate based only on the same-token hidden state.

    Input:
        x_t ∈ R^2048

    Output:
        g_i ∈ [0, 1]
    """

    def __init__(
        self,
        hidden_dim_model=2048,
        gate_hidden_dim=8,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_dim_model, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state):
        return self.network(hidden_state)


class HeadContextGate(nn.Module):
    """
    Option 2 adaptive gate.

    Input:
        Head representation: 128
        Context representation: 32

    Combined:
        128 + 32 = 160

    Architecture:
        160 -> 8 -> 1 -> sigmoid
    """

    def __init__(
        self,
        head_dim=128,
        context_dim=32,
        hidden_dim=8,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                head_dim + context_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        head_output,
        context,
    ):
        context = context.to(
            dtype=head_output.dtype,
            device=head_output.device,
        )

        combined = torch.cat(
            [
                head_output,
                context,
            ],
            dim=-1,
        )

        return self.network(combined)