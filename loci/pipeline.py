# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Chunk-wise causal generation with read-then-commit memory updates.

Timeline of latent frames: frame 0 is the condition frame, then chunks of 5 frames.
For every chunk the model (i) denoises the chunk from noise while *reading* the committed
history (softmax K/V of the planned history frames, KDA recurrent state), then (ii) runs one
extra *commit* forward of the finished chunk that writes its K/V and advances the KDA state.
The commit forward may see the chunk lightly re-noised (``commit_timestep``), which matches
the noise level the model saw for history during training. Frame 0 is always committed clean.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .memory import HistoryPlanner, HistoryState, TimeRotator
from .scheduler import FlowMatchEuler


@dataclass
class HistoryConfig:
    mode: str = "sparse"          # "sparse" (bounded: sink + view bank + recent) or "dense" (all frames)
    bank_size: int = 20
    recent: int = 8
    commit_timestep: float = 100.0
    renumber: bool = True         # compact temporal positions for the selected history frames
    pin_anchors: int = 10         # sparse: early anchor frames pinned in the view bank (0 = off, paper setting)
    pin_until: int = 80           # anchors are latent frames 1, 1 + pin_stride, ... <= pin_until (80 = 20 s)
    pin_stride: int = 8           # one anchor every 8 latent frames (2 s)

    @classmethod
    def preset(cls, name: str) -> "HistoryConfig":
        if name == "sparse":
            return cls("sparse", 20, 8, 100.0, True)
        if name == "dense":
            return cls("dense", 20, 8, 0.0, False, 0)
        raise ValueError(name)


def chunk_ranges(num_frames: int, chunk: int):
    if (num_frames - 1) % chunk:
        raise ValueError(f"latent frame count must be 1 + {chunk}k, got {num_frames}")
    return [(0, 1)] + [(s, s + chunk) for s in range(1, num_frames, chunk)]


def fov_intrinsics_norm(x_fov_deg: float, height: int, width: int):
    """Normalised pinhole intrinsics [fx/W, fy/H, cx/W, cy/H] for a horizontal FOV."""
    import math
    fx = .5 / math.tan(math.radians(float(x_fov_deg)) / 2)
    return [fx, fx * width / height, .5, .5]


class LociSampler:
    def __init__(self, transformer, *, steps: int = 50, shift: float = 5.0):
        self.model = transformer
        self.scheduler = FlowMatchEuler(steps, shift)

    def _forward(self, x, t, text, cam, s, e, history, kda_states, commit):
        f = e - s
        text_chunk = text[None, None].expand(1, f, *text.shape).contiguous()
        pose, x_fov, xi = cam
        return self.model(x, t, text_chunk, pose[:, s:e], x_fov[:, s:e], xi[:, s:e],
                          history=history, kda_states=kda_states, commit=commit)

    def _commit(self, x, s, e, text, cam, history, kda_states, commit_t: float):
        device = x.device
        t = torch.zeros((), device=device, dtype=torch.float32)
        if commit_t > 0 and s != 0:
            g = torch.Generator(device=device).manual_seed(1000003 * e + 17)
            eps = torch.randn(x.shape, generator=g, device=device, dtype=torch.float32)
            sig = commit_t / 1000.
            x = ((1 - sig) * x.float() + sig * eps).to(x.dtype)
            t = torch.full((), commit_t, device=device, dtype=torch.float32)
        history.begin_chunk(s, e)
        self._forward(x, t, text, cam, s, e, history, kda_states, commit=True)

    @torch.inference_mode()
    def generate(self, *, text: torch.Tensor, pose: torch.Tensor, x_fov, xi=0.0, prefix: torch.Tensor,
                 num_frames: int, history_cfg: HistoryConfig, planner_K_norm=None, seed: int = 42,
                 progress=None) -> torch.Tensor:
        """Returns normalised latents ``[1, C, num_frames, h, w]`` (prefix included).

        text:   [512, D] UMT5 embedding (padded, masked) in the model dtype
        pose:   [num_frames, 3|4, 4] camera-to-world per latent frame (frame 0 at the origin)
        prefix: [1, C, P, h, w] clean normalised latents; P = 1 (image) or 1 + 5k (video prefix)
        """
        model = self.model
        device, dtype = prefix.device, model.dtype
        chunk = model.config.chunk_size
        _, channels, P, h, w = prefix.shape
        ranges = chunk_ranges(num_frames, chunk)
        if P not in [e for _, e in ranges]:
            raise ValueError("prefix must end on a chunk boundary (1 + 5k latent frames)")
        pose = torch.as_tensor(pose, dtype=torch.float32)[:num_frames]
        if len(pose) != num_frames:
            raise ValueError(f"trajectory has {len(pose)} poses, need {num_frames}")
        x_fov_t = torch.as_tensor(x_fov, dtype=torch.float32).reshape(-1).expand(num_frames)[None].to(device)
        xi_t = torch.as_tensor(xi, dtype=torch.float32).reshape(-1).expand(num_frames)[None].to(device)
        cam = (pose[None].to(device), x_fov_t, xi_t)
        text = text.to(device=device, dtype=dtype).reshape(text.shape[-2:])

        patch = model.config.patch_size
        tpf = (h // patch[1]) * (w // patch[2])
        if planner_K_norm is None:
            planner_K_norm = fov_intrinsics_norm(float(x_fov_t[0, 0]), h, w)
        planner = HistoryPlanner(pose, planner_K_norm, mode=history_cfg.mode, bank_size=history_cfg.bank_size,
                                 recent=history_cfg.recent, chunk=chunk, pin_max=history_cfg.pin_anchors,
                                 pin_until=history_cfg.pin_until, pin_stride=history_cfg.pin_stride)
        history = HistoryState(planner, tpf, history_cfg.renumber, TimeRotator.for_model(model))
        kda_states = [dict() for _ in model.blocks]
        out = prefix.to(device=device, dtype=dtype)

        for s, e in ranges:
            if e > P:
                break
            self._commit(out[:, :, s:e], s, e, text, cam, history, kda_states, history_cfg.commit_timestep)

        generator = torch.Generator(device=device).manual_seed(seed)
        timesteps = self.scheduler.timesteps.to(device)
        todo = [(s, e) for s, e in ranges if e > P]
        for n, (s, e) in enumerate(todo):
            x = torch.randn(1, channels, e - s, h, w, device=device, dtype=dtype, generator=generator)
            history.begin_chunk(s, e)
            for i, t in enumerate(timesteps):
                v = self._forward(x, t, text, cam, s, e, history, kda_states, commit=False)
                x = self.scheduler.step(v, i, x)
            if e < num_frames:
                self._commit(x, s, e, text, cam, history, kda_states, history_cfg.commit_timestep)
            out = torch.cat([out, x], dim=2)
            if progress is not None:
                progress(n + 1, len(todo))
        return out
