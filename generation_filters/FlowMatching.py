from __future__ import annotations

import os
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint
from .utils import default, exists


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

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss weight based on the cosine function.
        The weight is positive and decreases as t approaches 1.
        """
        if self.loss_type == "adaptive":
            return torch.pow(torch.cos(torch.pi / 2 * (t - 3)), self.sigma).to(self.device)
        elif self.loss_type == "mean":
            return torch.ones_like(t).to(self.device)

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
     
        step_cond = cond

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        def fn(t, x):
            nonlocal cond
            x = torch.cat((x, cond), dim=1) if exists(cond) else x
            return self.base_model(x=x, time=t.expand(x.shape[0]))
        
        y0 = torch.randn_like(cond)
        # y0 = cond
        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        
        if use_ode:
            trajectory = odeint(fn, y0, t,** self.odeint_kwargs)
        else:
            trajectory = self._iterative_sample(fn, y0, t, steps, cond)

        sampled = trajectory[-1]
        out = sampled

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

        x1 = clean
        x0 = torch.randn_like(x1)
        # x0 = input
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0
        cond = input

        φ = torch.cat((φ, cond), dim=1) if exists(cond) else φ
        pred = self.base_model(
            x=φ, time=time
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss.mean(dim=2).squeeze(1) * self.loss_weight(time)
        return loss.mean(), cond, pred
    

class AdaCFM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        adapt_scheduler: nn.Module,
        sigma=0.1,
        odeint_kwargs: dict = dict(method="euler"),
        num_channels=None,
        sampling_timesteps: int = 10,
        default_use_ode: bool = True
    ):
        super().__init__()
        self.num_channels = num_channels

        self.base_model = base_model
        self.adapt_scheduler = adapt_scheduler
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.sampling_timesteps = sampling_timesteps
        self.default_use_ode = default_use_ode

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self, cond, *, steps=None, use_ode: bool | None = None):
        use_ode = self.default_use_ode if use_ode is None else use_ode
        steps = default(steps, self.sampling_timesteps)

        cond = cond.to(next(self.parameters()).dtype)
        step_cond = cond.clone()
        
        device = self.device
        dtype = cond.dtype

        def fn(t, x):
            nonlocal current_cond
            x = torch.cat((x, current_cond), dim=1) if exists(current_cond) else x
            return self.base_model(x=x, time=t.expand(x.shape[0]))
        
        y0 = torch.randn_like(cond)

        with torch.no_grad():
            pred_time = self.adapt_scheduler(y0).squeeze(-1)
            sample_steps = torch.ceil(pred_time * steps).long()
            unique_steps = torch.unique(sample_steps)

        out = torch.zeros_like(y0)
        trajectories = torch.zeros((steps+1, *y0.shape), device=device, dtype=dtype) if not use_ode else None

        for step in unique_steps:
            mask = (sample_steps == step)
            sub_batch = y0[mask]
            current_cond = step_cond[mask]
            
            current_steps = step.item()
            t_start = 1 - current_steps / steps
            t = torch.linspace(t_start, 1, current_steps + 1, device=device, dtype=dtype)

            if use_ode:
                sub_trajectory = odeint(fn, sub_batch, t, **self.odeint_kwargs)
                out[mask] = sub_trajectory[-1]
            else:
                sub_trajectory = self._iterative_sample(fn, sub_batch, t, current_steps)
                out[mask] = sub_trajectory[-1]
        
        return out, trajectories if not use_ode else out

    def _iterative_sample(self, fn, y0, t, steps):
        dt = t[1] - t[0]
        trajectory = [y0]
        x = y0
        method = self.odeint_kwargs.get("method", "euler")
        
        for i in range(steps):
            if method == "euler":
                k1 = fn(t[i], x)
                x = x + dt * k1
            elif method == "midpoint":
                k1 = fn(t[i], x)
                k2 = fn(t[i] + dt / 2, x + dt / 2 * k1)
                x = x + dt * k2
            trajectory.append(x)
            
        return torch.stack(trajectory)

    def forward(self, input, clean):

        batch, _, dtype, device, _ = *input.shape[:2], input.dtype, self.device, self.sigma

        x1 = clean
        # x0 = torch.randn_like(x1)
        x0 = input
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        pred_time = self.adapt_scheduler(φ)
        flow = x1 - x0
        cond = input
        φ = torch.cat((φ, cond), dim=1) if exists(cond) else φ
        pred = self.base_model(
            x=φ, time=time
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss + self.sigma * F.mse_loss(pred_time, time.unsqueeze(-1), reduction="none")
        return loss.mean(), cond, pred