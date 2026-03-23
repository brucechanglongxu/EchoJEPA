import torch
import torch.nn as nn
import torch.nn.functional as F


class StateHead(nn.Module):
    """Compress encoder token features into a single physiological state vector.

    Pipeline: mean_pool(tokens) → LayerNorm → Linear → L2-normalize
    """

    def __init__(self, embed_dim: int, state_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, state_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) encoder output tokens
        Returns:
            z: (B, d_state) L2-normalized state vector on unit sphere
        """
        z = tokens.mean(dim=1)
        z = self.norm(z)
        z = self.proj(z)
        z = F.normalize(z, dim=-1)
        return z


class Transition(nn.Module):
    """Residual MLP that predicts the next physiological state.

    z_{t+1} = normalize(z_t + scale * MLP(z_t))

    The last layer is zero-initialized so the model starts as exact persistence
    (f(z) = z). The learnable scale factor starts small (0.01) to ensure the
    model makes tiny corrections initially and grows them only if beneficial.
    """

    def __init__(self, state_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.log_scale = nn.Parameter(torch.tensor(-4.6))  # exp(-4.6) ≈ 0.01

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_state) current state (assumed L2-normalized)
        Returns:
            z_next: (B, d_state) predicted next state (L2-normalized)
        """
        scale = self.log_scale.exp()
        z_next = z + scale * self.net(z)
        return F.normalize(z_next, dim=-1)

    def rollout(self, z: torch.Tensor, horizon: int) -> list[torch.Tensor]:
        """Multi-step rollout for evaluation.

        Returns list of predicted states [z_{t+1}, z_{t+2}, ..., z_{t+H}].
        """
        states = []
        for _ in range(horizon):
            z = self.forward(z)
            states.append(z)
        return states
