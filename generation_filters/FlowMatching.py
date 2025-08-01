from __future__ import annotations

import os
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint
from .utils import default, exists

from .AttnUnet import AutoEncoder
class CFM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        sigma=1.75,
        odeint_kwargs: dict = dict(method="euler"),
        num_channels=None,
        sampling_timesteps: int = 10,
        default_use_ode: bool = True,
        loss_type: str = "mean"
    ):
        super().__init__()
        self.num_channels = num_channels

        self.base_model = base_model
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.sampling_timesteps = sampling_timesteps
        self.default_use_ode = default_use_ode
        self.loss_type = loss_type
        self.autoencoder = AutoEncoder()

    @property
    def device(self):
        return next(self.parameters()).device

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss weight based on the cosine function.
        The weight is positive and decreases as t approaches 1.
        """
        if self.loss_type == "adaptive":
            return torch.pow(torch.cos(torch.pi / 2 * (t - 3)), self.sigma).to(self.device)
        elif self.loss_type == "mean":
            return torch.ones_like(t).to(self.device)
    
    def vae_loss(self, recon_x, x, mean, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -1e-4 * (1 + logvar - mean.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss.mean()
    
        return vae_loss

    @torch.no_grad()
    def sample(
        self,
        cond,
        *,
        steps = None,
        no_ref_audio=False,
        use_ode: bool | None = None
    ):
        use_ode = self.default_use_ode if use_ode is None else use_ode
        steps = default(steps, self.sampling_timesteps)

        cond = cond.to(next(self.parameters()).dtype)
        cond = self.autoencoder.encode(cond)[0] # Add encoder
     
        step_cond = cond

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        def fn(t, x):
            nonlocal cond
            x = torch.cat((x, cond), dim=1) if exists(cond) else x
            return self.base_model(x=x, time=t.expand(x.shape[0]), x_self_cond=cond)
        
        # y0 = torch.randn_like(cond)
        y0 = cond
        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        
        if use_ode:
            trajectory = odeint(fn, y0, t,** self.odeint_kwargs)
        else:
            trajectory = self._iterative_sample(fn, y0, t, steps, cond)

        sampled = trajectory[-1]
        out = sampled
        out = self.autoencoder.decode(out) # Add decoder

        return out, trajectory

    def _iterative_sample(self, fn, y0, t, steps, cond):
        dt = t[1] - t[0]
        # trajectory = [y0]
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

    def forward(self, input, clean):

        batch, _, dtype, device, _ = *input.shape[:2], input.dtype, self.device, self.sigma

        # x1 = clean
        # x0 = torch.randn_like(x1)
        x1, _, _ = self.autoencoder.encode(clean)
        x0, mean, logvar = self.autoencoder.encode(input)
        recon_x0 = self.autoencoder.decode(x0)
        # x0 = input
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0
        cond = x0

        φ = torch.cat((φ, cond), dim=1) if exists(cond) else φ
        pred = self.base_model(
            x=φ, time=time, x_self_cond=cond
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        vae_loss = self.vae_loss(recon_x0, input, mean, logvar)
        # loss = loss.mean(dim=2).squeeze(1) * self.loss_weight(t=time)
        loss = loss.mean(dim=(1,2)) * self.loss_weight(t=time)
        return loss.mean() + vae_loss, cond, pred
    