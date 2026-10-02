# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Flow-matching Euler sampler with Wan's shifted sigma schedule (shift=5 for TI2V-5B)."""
from __future__ import annotations

import torch


class FlowMatchEuler:
    def __init__(self, num_steps: int = 50, shift: float = 5.0, num_train_timesteps: int = 1000):
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

    def step(self, velocity: torch.Tensor, index: int, sample: torch.Tensor) -> torch.Tensor:
        """x_{i+1} = x_i + v * (sigma_{i+1} - sigma_i); ``sample``/``velocity``: [B, C, F, H, W]."""
        b, _, f = sample.shape[:3]
        sigma = self.sigmas[index].repeat(b, f)
        nxt = self.sigmas[index + 1].repeat(b, f) if index + 1 < len(self.sigmas) else torch.zeros(b, f)
        diff = (nxt - sigma).to(velocity.device).view(b, 1, f, 1, 1)
        return (sample + velocity * diff).to(sample)
