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
    """MLP that predicts the next physiological state.

    Two architectural toggles, controlled by config flags:

    * ``residual=True`` (default): ``z_{t+1} = normalize(z_t + s * MLP(z_t))``
      where ``s = exp(init_scale)``. Combined with ``zero_init=True`` this
      gives an *identity-initialized* transition: f(z) == z at step 0,
      which is the right inductive bias for a probe (Sec. 3 of the MI4MedFM
      paper) and the right warm-start for a world model (Sec. 3 of MWM).

    * ``residual=False``: ``z_{t+1} = normalize(MLP(z_t))``. Used as an
      ablation cell (S5_init / noresidual) to test whether the residual
      structure matters at all.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 512,
        zero_init: bool = True,
        residual: bool = True,
        init_scale: float = -4.6,
    ):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Learnable log-scale; only applied in the residual branch.
        # exp(-4.6) ~= 0.01 puts the model arbitrarily close to identity.
        # Register only when residual=True so DDP doesn't see an unused param.
        if residual:
            self.log_scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_parameter("log_scale", None)

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_state) current state (assumed L2-normalized)
        Returns:
            z_next: (B, d_state) predicted next state (L2-normalized)
        """
        if self.residual:
            scale = self.log_scale.exp()
            z_next = z + scale * self.net(z)
        else:
            z_next = self.net(z)
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
