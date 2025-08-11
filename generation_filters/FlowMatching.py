from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint


class CFM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        autoencoder: nn.Module,
        sigma_max : float = 1.0,
        sigma_min : float = 1e-5,
        odeint_kwargs: dict = dict(method="euler"),
        num_channels=None,
        sampling_timesteps: int = 10,
        default_use_ode: bool = False,
        loss_type: str = "mean"
    ):
        super().__init__()
        self.num_channels = num_channels

        self.base_model = base_model
        self.autoencoder = autoencoder
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.odeint_kwargs = odeint_kwargs
        self.sampling_timesteps = sampling_timesteps
        self.default_use_ode = default_use_ode
        self.loss_type = loss_type

    @property
    def device(self):
        return next(self.parameters()).device
    
    # Get the psi as Backbone's input
    def get_psi(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        assert (
            t.shape[0] == x0.shape[0]
        ), f"Batch size of t and x0 does not agree {t.shape[0]} vs. {x0.shape[0]}"
        assert (
            t.shape[0] == x1.shape[0]
        ), f"Batch size of t and x1 does not agree {t.shape[0]} vs. {x1.shape[0]}"
        return (t * (self.sigma_min / self.sigma_max - 1) + 1) * x0 + t * x1
    
    # Dt / dx function
    def get_flow(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        assert x0.shape[0] == x1.shape[0]
        return (self.sigma_min / self.sigma_max - 1) * x0 + x1


    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        self_cond=None,
        *,
        steps = None,
        use_ode: bool | None = None
    ):
        use_ode = self.default_use_ode if use_ode is None else use_ode
        steps = self.sampling_timesteps if steps is None else steps

        condition = condition.to(next(self.parameters()).dtype)
        z_cond = self.autoencoder.encode(condition)

        def fn(t, x):
            nonlocal z_cond, self_cond
            return self.base_model(x=x, t=t.expand(x.shape[0]), z_cond=z_cond, x_self_cond=self_cond)
        
        # y0 = torch.randn_like(condition)
        y0 = condition

        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=condition.dtype)
        
        if use_ode:
            trajectory = odeint(fn, y0, t,** self.odeint_kwargs)
        else:
            trajectory = self._iterative_sample(fn, y0, t, steps, condition)

        out = trajectory[-1]

        return out, trajectory

    def _iterative_sample(self, fn, y0, t, steps, cond):
        dt = t[1] - t[0]

        trajectory = [cond]
        x = y0
        method = self.odeint_kwargs.get("method", "euler")
        
        for i in range(steps):
            if method == "euler":
                k1 = fn(t[i], x)
                x = x + dt * k1

            elif method == "midpoint":
                k1 = fn(t[i], x)
                k2 = fn(t[i] + dt/2, x + dt/2 * k1)
                x = x + dt * k2
            trajectory.append(x)
            
        return torch.stack(trajectory)

    def forward(self, input, clean, self_cond=None):
        """Flow Matching forward pass
        Args:
            input (torch.Tensor): Noisy ECG signal
            clean (torch.Tensor): Clean ECG signal
            self_cond (torch.Tensor, optional): Noisy ECG signal, Defaults to None.

        Returns:
            loss (torch.Tensor): Loss value
            pred (torch.Tensor): Predicted flow
        """

        batch, _, dtype = *input.shape[:2], input.dtype

        x1 = clean
        # x0 = torch.randn_like(x1)
        x0 = input
        
        with torch.no_grad():
            z_cond = self.autoencoder.encode(input)
            if random.random() < 0.1:
                self_cond += 0.15 * torch.randn_like(self_cond)

        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = self.get_psi(t, x0, x1)
        flow = self.get_flow(t, x0, x1)

        pred = self.base_model(
            x=φ, t=time, z_cond=z_cond, x_self_cond=self_cond
        )

        loss = F.mse_loss(pred, flow, reduction="none")

        return loss.mean(), pred
    