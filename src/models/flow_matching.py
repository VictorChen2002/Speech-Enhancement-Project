"""
flow_matching.py — Rectified Flow (Flow-Matching) utilities.

Implements:
  * ``RectifiedFlow``  – training loss (MSE on vector field) 
  * ``ode_solve``      – Euler ODE integration from X_0 → X_1
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, List


class RectifiedFlow:
    """
    Rectified-flow helper that computes the straight-path interpolation
    and the MSE loss between the predicted and target vector fields.

    Notation
    --------
    X_0  : noisy DAC latent (source)
    X_1  : clean DAC latent (target)
    t    : scalar in [0, 1]
    X_t  = t * X_1 + (1 - t) * X_0        (straight-line interpolation)
    V    = X_1 - X_0                       (target vector field, constant)
    """

    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample uniform random timesteps in (0, 1)."""
        return torch.rand(batch_size, device=device)

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the straight-path interpolation X_t.

        Parameters
        ----------
        x_0 : (B, T, D) noisy latent
        x_1 : (B, T, D) clean latent
        t   : (B,) timestep

        Returns
        -------
        x_t : (B, T, D)
        """
        t = t[:, None, None]  # (B, 1, 1)
        return t * x_1 + (1 - t) * x_0

    def target_vector_field(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Target vector field V = X_1 - X_0  (constant along the path).

        Returns
        -------
        v : (B, T, D)
        """
        return x_1 - x_0

    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_layers: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        One training step: sample t, interpolate, predict V, return MSE loss.

        Parameters
        ----------
        model       : DiffusionTransformer
        x_0         : (B, T, D) noisy latent
        x_1         : (B, T, D) clean latent
        cond        : (B, T_c, D_c) or None — MOSS last-layer embedding
        cond_layers : list of (B, T_c, D_c) or None — MOSS multi-layer embeddings

        Returns
        -------
        loss : scalar tensor
        """
        B = x_0.shape[0]
        device = x_0.device

        t = self.sample_t(B, device)
        x_t = self.interpolate(x_0, x_1, t)
        v_target = self.target_vector_field(x_0, x_1)

        v_pred = model(x_t, t, cond=cond, cond_layers=cond_layers)
        return self.loss_fn(v_pred, v_target)


# --------------------------------------------------------------------------- #
#  ODE Solver (Euler method)                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ode_solve(
    model: nn.Module,
    x_0: torch.Tensor,
    num_steps: int = 50,
    cond: Optional[torch.Tensor] = None,
    cond_layers: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Solve the ODE  dx/dt = v_θ(x_t, t)  from t=0 (noisy) to t=1 (clean)
    using the Euler method.

    Parameters
    ----------
    model     : DiffusionTransformer
    x_0       : (B, T, D) noisy DAC latent
    num_steps : number of Euler steps
    cond      : optional MOSS last-layer condition
    cond_layers : optional MOSS multi-layer conditions

    Returns
    -------
    x_1 : (B, T, D) predicted clean DAC latent
    """
    dt = 1.0 / num_steps
    x_t = x_0.clone()

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((x_t.shape[0],), t_val, device=x_t.device)
        v = model(x_t, t, cond=cond, cond_layers=cond_layers)
        x_t = x_t + v * dt

    return x_t
