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
        for p in self.autoencoder.parameters():
            p.requires_grad = False
            
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
        y0 = condition.clone()

        z_cond = self.autoencoder.inference(condition)

        def fn(t, x):
            nonlocal z_cond, self_cond
            velocity = self.base_model(x=x, t=t.expand(x.shape[0]), z_cond=z_cond, x_self_cond=self_cond)
            return velocity
        
        y0 = y0 - z_cond

        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=condition.dtype)
        
        if use_ode:
            trajectory = odeint(fn, y0, t,** self.odeint_kwargs)
        else:
            trajectory = self._iterative_sample(fn, y0, t, steps, condition)

        out = trajectory[-1]
        out = out + z_cond

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
    
    def add_noise(self, x0):
        B, C, L = x0.shape
        a = x0.abs()
        pad_left = torch.zeros(B, C, 8, device=x0.device, dtype=x0.dtype)
        pad_right = torch.zeros(B, C, 8, device=x0.device, dtype=x0.dtype)
        padded = torch.cat([pad_left, a, pad_right], dim=-1) 
        windows = padded.unfold(dimension=-1, size=17, step=1)
        w = windows.sum(dim=-1).abs()

        w_min = w.amin(dim=-1, keepdim=True)
        w_max = w.amax(dim=-1, keepdim=True)
        w_norm = (w - w_min) / (w_max - w_min + 1e-8)
        norm = 2.0 * w_norm - 1.0
        scale = 0.1 + 0.05 * norm
        noise = torch.randn_like(x0)
        x0 = x0 + scale * noise
        return x0

    def forward(self, input, clean, self_cond=None, snr=None ,step=0):
        """Flow Matching forward pass
        Args:
            input (torch.Tensor): Noisy ECG signal
            clean (torch.Tensor): Clean ECG signal
            self_cond (torch.Tensor, optional): Noisy ECG signal, Defaults to None.

        Returns:
            loss (torch.Tensor): Loss value
            pred (torch.Tensor): Predicted flow
        """

        batch, _, dtype, device = *input.shape[:2], input.dtype, input.device

        x1 = clean
        x0 = input.clone()
        
        with torch.no_grad():
            z_cond = self.autoencoder.inference(input)
            
            x1 = x1 - z_cond
            x0 = x0 - z_cond
            
            x0 = self.add_noise(x0)

        time = torch.rand((batch,), dtype=dtype, device=device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = self.get_psi(t, x0, x1)
        flow = self.get_flow(t, x0, x1)

        pred, _ = self.base_model(
            x=φ, t=time, z_cond=z_cond, x_self_cond=self_cond
        )

        if snr is not None:
            snr_min, snr_max = -7.0, 18.0
            span = 24.0
            snr = snr.to(dtype=dtype, device=device)
            loss_weight = ((snr - snr_min) / span).view(batch, 1)
        else:
            loss_weight = torch.ones((batch, 1), dtype=dtype, device=device)

        pred_loss = F.mse_loss(pred, flow, reduction="none").mean(dim=-1)
        loss = (pred_loss * loss_weight).mean()

        return loss, pred
