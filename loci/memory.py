# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""History management for chunk-wise generation.

Two pieces:

* :class:`HistoryPlanner` decides, for every chunk, which past latent frames the softmax
  layers (and the camera branch) may attend to.
    - ``dense``: all past frames.
    - ``sparse``: a bounded set = frame 0 (sink) + a view bank + the ``recent`` most recent
      frames. Frames leaving the recent window enter the bank only if they add enough new
      field-of-view coverage (earliest-view-per-direction policy); when the bank is full the
      frame most redundant with the others is evicted. Memory is constant in time.
      Anchor pinning (default on): the first ``pin_max`` frames ``1, 1 + pin_stride, ...`` up to
      latent frame ``pin_until`` (default 10 frames, one every 2 s within the first 20 s) are
      admitted to the bank when they leave the recent window, regardless of coverage, and are
      never evicted. This keeps early, clean views in memory on long rollouts and reduces
      long-horizon drift. ``pin_max=0`` gives the paper configuration.
  The KDA recurrent state is independent of this choice: it always accumulates every
  committed chunk in order.

* :class:`FrameKVCache` is a fixed-capacity K/V buffer per (layer, tensor) that keeps exactly
  the planned history frames followed by the current chunk. With ``renumber=True`` the
  selected history frames are given compact temporal positions (their order preserved) and
  cached main-attention keys are re-rotated to those positions.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MAX_ROPE_START = 1015  # queries never use temporal RoPE positions beyond this (table size 1024)


def _np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)


class HistoryPlanner:
    def __init__(self, pose, K_norm, mode: str = "sparse", bank_size: int = 20, recent: int = 8,
                 min_new_coverage: float = 0.3, radius: float = 6.0, probes: int = 4096, chunk: int = 5,
                 pin_max: int = 10, pin_until: int = 80, pin_stride: int = 8):
        if mode not in ("dense", "sparse"):
            raise ValueError("mode must be 'dense' or 'sparse'")
        if pin_max < 0 or pin_stride < 1 or (mode == "sparse" and pin_max >= bank_size > 0):
            raise ValueError("need 0 <= pin_max < bank_size and pin_stride >= 1")
        self.mode = mode
        self.bank_size = bank_size
        self.pin_max = pin_max
        self.pin_until = pin_until
        self.pin_stride = pin_stride
        self.pinned: list[int] = []
        self.recent = recent
        self.min_new_coverage = min_new_coverage
        self.radius = radius
        self.chunk = chunk
        self.pose = _np(pose)[:, :3, :4]
        self.frames = len(self.pose)
        self.bank: list[int] = []
        self.plans: dict[int, list[int]] = {}
        if mode == "dense":
            return
        k = np.broadcast_to(_np(K_norm), (self.frames, 4))
        n = np.arange(probes, dtype=np.float64)
        z = 1 - 2 * (n + .5) / probes
        phi = np.pi * (3 - np.sqrt(5)) * n
        radial = np.sqrt(1 - z * z)
        dirs = np.stack((radial * np.cos(phi), radial * np.sin(phi), z), axis=-1)  # Fibonacci sphere
        self.cover = []
        for rotation, ki in zip(self.pose[:, :3, :3], k):
            local = dirs @ rotation
            zz = local[:, 2]
            safe = np.where(zz > 0, zz, 1)
            u = local[:, 0] / safe * ki[0] + ki[2]
            v = local[:, 1] / safe * ki[1] + ki[3]
            self.cover.append((zz > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))

    def _covered_fraction(self, i: int, others) -> float:
        t = self.pose[:, :3, 3]
        union = np.zeros_like(self.cover[i])
        for j in others:
            if j != i and np.linalg.norm(t[i] - t[j]) < self.radius:
                union |= self.cover[j]
        return np.count_nonzero(self.cover[i] & union) / max(1, np.count_nonzero(self.cover[i]))

    def _update_bank(self, leaving):
        bank = list(self.bank)
        for f in sorted(leaving):
            pin = (f <= self.pin_until and (f - 1) % self.pin_stride == 0 and len(self.pinned) < self.pin_max)
            if not pin and 1 - self._covered_fraction(f, [0] + bank) < self.min_new_coverage:
                continue
            bank.append(f)
            if pin:
                self.pinned.append(f)
            if len(bank) > self.bank_size:
                evictable = [b for b in bank if b not in self.pinned]
                redundancy = {b: round(self._covered_fraction(b, [0] + bank), 12) for b in evictable}
                bank.remove(max(evictable, key=lambda b: (redundancy[b], b)))
        self.bank = sorted(bank)

    def history(self, end: int) -> list[int]:
        """Past frames visible to the chunk that ends (exclusively) at latent frame ``end``."""
        if end in self.plans:
            return self.plans[end]
        n = 0 if end == 1 else (end - 1) // self.chunk
        start = 0 if n == 0 else end - self.chunk
        if self.mode == "dense":
            hist = list(range(start))
        else:
            R = self.recent
            leaving = [f for f in range(start - R - self.chunk, start - R) if f >= 1]
            self._update_bank(leaving)
            hist = ([0] if start else []) + self.bank + list(range(max(1, start - R), start))
        hist = sorted(set(hist))
        self.plans[end] = hist
        return hist


class FrameKVCache:
    """Fixed-capacity per-tensor history buffer (frames = latent frames of ``tokens_per_frame`` tokens)."""

    def __init__(self, name: str, tokens_per_frame: int, capacity_frames: int | None, rotate):
        self.name = name
        self.tpf = tokens_per_frame
        self.capacity = capacity_frames
        self.rotate = rotate  # callable(k, delta_frames) or None (values / camera keys)
        self.buf = None
        self.rbuf = None
        self.hist: list[int] = []
        self.cur: list[int] = []
        self.end = None

    def update(self, x: torch.Tensor, end: int, keep: list[int], delta: dict | None) -> torch.Tensor:
        tpf = self.tpf
        n = x.shape[1] // tpf
        if end != self.end:
            frames = self.hist + self.cur
            pos = {f: i for i, f in enumerate(frames)}
            missing = [f for f in keep if f not in pos]
            if missing:
                raise RuntimeError(f"{self.name}: history frames {missing} were already evicted")
            need = (len(keep) + n) * tpf
            if self.buf is None or self.buf.shape[1] < need:
                size = self.capacity * tpf if self.capacity else need + 40 * tpf
                new = torch.empty((x.shape[0], size) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
                if keep:
                    idx = torch.tensor([pos[f] * tpf + t for f in keep for t in range(tpf)], device=x.device)
                    new[:, :len(keep) * tpf] = self.buf.index_select(1, idx)
                self.buf = new
            elif keep != frames[:len(keep)]:
                idx = torch.tensor([pos[f] * tpf + t for f in keep for t in range(tpf)], device=x.device)
                self.buf[:, :len(keep) * tpf] = self.buf.index_select(1, idx)
            self.hist = list(keep)
            self.cur = list(range(end - n, end))
            self.end = end
            if self.rotate is not None and delta is not None and any(v != 0 for v in delta.values()):
                if self.rbuf is None or self.rbuf.shape[1] < self.buf.shape[1]:
                    self.rbuf = torch.empty_like(self.buf)
                h = len(keep) * tpf
                if h:
                    d = torch.tensor([delta[f] for f in keep], device=x.device).repeat_interleave(tpf)
                    self.rbuf[:, :h] = self.rotate(self.buf[:, :h], d)
            else:
                self.rbuf = None
        h = len(self.hist) * tpf
        self.buf[:, h:h + x.shape[1]] = x
        if self.rbuf is not None:
            self.rbuf[:, h:h + x.shape[1]] = x
            return self.rbuf[:, :h + x.shape[1]]
        return self.buf[:, :h + x.shape[1]]


class TimeRotator:
    """Re-rotates cached keys along the Wan temporal RoPE sub-space by an integer frame offset."""

    def __init__(self, rope_cos: torch.Tensor, rope_sin: torch.Tensor, time_dims: int = 44):
        """``rope_cos``/``rope_sin``: the float32 Wan RoPE tables (before any cast to the model dtype)."""
        fc, fs = rope_cos.double(), rope_sin.double()
        self.time_dims = time_dims
        self.omega = torch.atan2(fs[1, 0:time_dims:2], fc[1, 0:time_dims:2])

    @classmethod
    def for_model(cls, model):
        from .transformer import ChunkRotaryPosEmbed
        cfg = model.config
        table = ChunkRotaryPosEmbed(cfg.attention_head_dim, tuple(cfg.patch_size), cfg.rope_max_seq_len)
        t_dim = cfg.attention_head_dim - 2 * (cfg.attention_head_dim // 3)
        return cls(table.freqs_cos.float(), table.freqs_sin.float(), time_dims=t_dim)

    def __call__(self, k: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        td = self.time_dims
        ang = delta.double()[:, None] * self.omega.to(k.device)[None, :]
        cos = ang.cos().float()[None, :, None, :]
        sin = ang.sin().float()[None, :, None, :]
        out = k.clone()
        x1 = k[..., 0:td:2].float()
        x2 = k[..., 1:td:2].float()
        out[..., 0:td:2] = (x1 * cos - x2 * sin).to(k.dtype)
        out[..., 1:td:2] = (x1 * sin + x2 * cos).to(k.dtype)
        return out


class HistoryState:
    """Per-generation state shared by all layers: plan, positions, and per-layer caches."""

    def __init__(self, planner: HistoryPlanner, tokens_per_frame: int, renumber: bool, rotator: TimeRotator):
        self.planner = planner
        self.tpf = tokens_per_frame
        self.capacity = None if planner.mode == "dense" else 1 + planner.bank_size + planner.recent + planner.chunk
        self.renumber = renumber
        self.rotator = rotator
        self.caches: dict[tuple, FrameKVCache] = {}
        self.end = None
        self.keep: list[int] = []
        self.delta: dict[int, int] = {}
        self.written: dict[int, int] = {}
        self.rope_start = 0

    def begin_chunk(self, start: int, end: int):
        """Called once per forward; (re)plans when a new chunk starts."""
        if end == self.end:
            return
        q0 = min(start, MAX_ROPE_START)
        self.rope_start = q0
        self.keep = self.planner.history(end)
        for t in range(start, end):
            self.written[t] = q0 + (t - start)
        h = len(self.keep)
        self.delta = {}
        for r, t in enumerate(self.keep):
            want = q0 + ((r - h) if self.renumber else (t - start))
            self.delta[t] = want - self.written[t]
        self.end = end

    def cache(self, layer: int, name: str, x: torch.Tensor) -> torch.Tensor:
        key = (layer, name)
        c = self.caches.get(key)
        if c is None:
            rotate = self.rotator if name == "main_k" else None
            c = self.caches[key] = FrameKVCache(f"{layer}.{name}", self.tpf, self.capacity, rotate)
        return c.update(x, self.end, self.keep, self.delta)
