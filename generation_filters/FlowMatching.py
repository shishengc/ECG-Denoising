from __future__ import annotations

import os
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint
from .utils import default, exists

import pywt
import numpy as np
class SimpleWaveletExtractor:

    def __init__(self, wavelet='db6', levels=8, fs=360):
        self.wavelet = wavelet
        self.levels = levels
        self.fs = fs
    
    def extract_components(self, ecg_signal):

        coeffs = pywt.wavedec(ecg_signal, self.wavelet, level=self.levels)

        d4_d8_coeffs = [np.zeros_like(coeffs[0])]
        d4_d8_coeffs.extend([coeffs[i] for i in range(1, 6)])
        d4_d8_coeffs.extend([np.zeros_like(coeffs[i]) for i in range(6, 9)])
        d4_d8_signal = pywt.waverec(d4_d8_coeffs, self.wavelet)

        return d4_d8_signal



class CFM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        sigma=1.75,
        odeint_kwargs: dict = dict(method="euler"),
        num_channels=None,
        sampling_timesteps: int = 10,
        default_use_ode: bool = True,
        loss_type: str = "mean",
        wavelet_cond: bool = False
    ):
        super().__init__()
        self.num_channels = num_channels

        self.base_model = base_model
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.sampling_timesteps = sampling_timesteps
        self.default_use_ode = default_use_ode
        self.loss_type = loss_type
        
        if wavelet_cond:
            self.wavelet_extractor = SimpleWaveletExtractor()
        else:
            self.wavelet_extractor = None

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

    @torch.no_grad()
    def sample(
        self,
        cond,
        *,
        steps = None,
        use_ode: bool | None = None
    ):
        use_ode = self.default_use_ode if use_ode is None else use_ode
        steps = default(steps, self.sampling_timesteps)

        cond = cond.to(next(self.parameters()).dtype)
        self_cond = cond
        
        if self.wavelet_extractor is not None:
            cond = cond.cpu().numpy()
            cond = self.wavelet_extractor.extract_components(cond)
            cond = torch.tensor(cond, dtype=next(self.parameters()).dtype, device=self.device)

        def fn(t, x):
            nonlocal cond, self_cond
            x = torch.cat((x, cond), dim=1) if exists(cond) else x
            return self.base_model(x=x, time=t.expand(x.shape[0]), x_self_cond=self_cond)
        
        y0 = torch.randn_like(cond)
        # y0 = cond
        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=cond.dtype)
        
        if use_ode:
            trajectory = odeint(fn, y0, t,** self.odeint_kwargs)
        else:
            trajectory = self._iterative_sample(fn, y0, t, steps, cond)

        out = trajectory[-1]

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
        self_cond = input
        
        if self.wavelet_extractor is not None:
            cond = input.cpu().numpy()
            cond = self.wavelet_extractor.extract_components(cond)
            cond = torch.tensor(cond, dtype=next(self.parameters()).dtype, device=self.device)
        else:
            cond = input

        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        φ = torch.cat((φ, cond), dim=1) if exists(cond) else φ
        pred = self.base_model(
            x=φ, time=time, x_self_cond=self_cond
        )

        loss = F.mse_loss(pred, flow, reduction="none")

        loss = loss.mean(dim=(1,2)) * self.loss_weight(t=time)
        return loss.mean(), cond, pred
    