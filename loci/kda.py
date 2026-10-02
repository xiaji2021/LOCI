# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Hybrid chunk attention: intra-chunk softmax + chunk-level KDA recurrent memory.

A hybrid block replaces the full-history softmax of a Wan block by

    out = softmax_attention(current chunk only) + sigmoid(W_g x) * read(S)

where ``S`` is a Kimi-Delta-Attention (KDA) recurrent state that summarises all committed
history chunks. Reading happens before the current chunk is written ("read-then-commit"):
the state is only advanced by an explicit commit forward of the finished chunk.

Recurrent addresses (q/k/v of the state) are per-head 1x1 projections of the block's
own q/k/v. Their channel layout per 128-d head is
  [0:12]   Wan temporal RoPE slot -> identity (no time encoding in the state),
  [12:44]  camera slot            -> PRoPE with per-ray camera frames and pinhole K,
  [44:128] Wan spatial (h, w) RoPE, unchanged.
Decay is chunk-level (one data-dependent retention per chunk, pair-tied across channels,
floored at ``min_retention``) so memory fades per generated chunk rather than per token.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .attention import PropeTransforms, apply_rope, chunk_local_attention

CAMERA_SLOT = (12, 44)
TIME_DIMS = 12


class KDAMemory(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, min_retention: float = 0.2,
                 prope_image_scale: int = 32, translation_scale: float = 6.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_scale = self.head_dim ** -0.5
        self.min_retention = min_retention
        self.prope_image_scale = prope_image_scale
        self.translation_scale = translation_scale
        # Decay is tied across channel pairs: 64 gates per head for a 128-d head.
        self.group_sizes = (2,) * (self.head_dim // 2)
        gate_dim = num_heads * len(self.group_sizes)
        self.a_proj = nn.Linear(dim, gate_dim, bias=False)   # decay (log-retention) projection
        self.b_proj = nn.Linear(dim, num_heads, bias=False)  # write strength (beta)
        self.gate_proj = nn.Linear(dim, num_heads, bias=False)  # read-out gate
        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(gate_dim, dtype=torch.float32))
        self.phi_q = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads, bias=False)
        self.phi_k = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads, bias=False)
        self.phi_v = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads, bias=False)
        self._repeats = None

    # ------------------------------------------------------------------ helpers
    def _project(self, phi: nn.Conv1d, t: torch.Tensor) -> torch.Tensor:
        b, l, h, d = t.shape
        out = phi(t.flatten(2).transpose(1, 2)).transpose(1, 2)
        return out.contiguous().view(b, l, h, d)

    @staticmethod
    def _camera_slot(t: torch.Tensor, fn) -> torch.Tensor:
        lo, hi = CAMERA_SLOT
        with torch.autocast(device_type=t.device.type, enabled=False):
            value = fn(t[..., lo:hi].float().transpose(1, 2)).transpose(1, 2)
        return torch.cat((t[..., :lo], value.to(t.dtype), t[..., hi:]), -1)

    def _decay(self, x: torch.Tensor) -> torch.Tensor:
        """Log-retention ``g`` [B, L, H, D]: one chunk-level gate placed on the first token."""
        projected = self.a_proj(x.mean(dim=1, keepdim=True)).float()
        b, l, _ = projected.shape
        n_gates = len(self.group_sizes)
        raw = projected.view(b, l, self.num_heads, n_gates) + self.dt_bias.view(self.num_heads, n_gates)
        rate = self.A_log.float().exp().view(1, 1, self.num_heads, 1)
        gate = -rate * torch.nn.functional.softplus(raw)
        if self._repeats is None or self._repeats.device != gate.device:
            self._repeats = torch.tensor(self.group_sizes, device=gate.device, dtype=torch.long)
        gate = torch.repeat_interleave(gate, self._repeats, dim=-1, output_size=sum(self.group_sizes))
        gate = gate.clamp_min(math.log(self.min_retention))
        g = x.new_zeros((x.shape[0], x.shape[1], *gate.shape[2:]), dtype=gate.dtype)
        g[:, 0:1] = gate
        return g

    # ------------------------------------------------------------------ forward
    def forward(self, x, q, k, v, rope, camera, state: dict, commit: bool) -> torch.Tensor:
        """One chunk. ``x``: normalised block input [B, L, C]; q/k/v: [B, L, H, D] (after QK-norm).

        ``rope``: Wan (cos, sin) for the chunk. ``camera``: dict from ``build_camera_inputs``.
        ``state``: per-stream dict holding the committed fp32 state under ``"S"``.
        """
        from fla.modules.l2norm import l2_norm
        from fla.ops.kda import chunk_kda

        cos, sin = rope
        roped_q = apply_rope(q, cos, sin).to(v.dtype)
        roped_k = apply_rope(k, cos, sin).to(v.dtype)
        intra = chunk_local_attention(roped_q, roped_k, v)

        # Recurrent addressing: identity on the time slot, PRoPE on the camera slot, Wan
        # spatial RoPE on the rest (computed in fp32, as trained).
        batch = q.shape[0]
        rec_cos = torch.cat((torch.ones_like(cos[..., :CAMERA_SLOT[1]], dtype=torch.float32),
                             cos[..., CAMERA_SLOT[1]:].float()), -1).expand(batch, -1, -1, -1)
        rec_sin = torch.cat((torch.zeros_like(sin[..., :CAMERA_SLOT[1]], dtype=torch.float32),
                             sin[..., CAMERA_SLOT[1]:].float()), -1).expand(batch, -1, -1, -1)
        grid = camera["grid"]
        prope = PropeTransforms(camera["rec_mats"], camera["rec_K"], head_dim=CAMERA_SLOT[1] - CAMERA_SLOT[0],
                                grid_hw=grid[1:], image_scale=self.prope_image_scale)
        q_s = self._camera_slot(self._project(self.phi_q, q), prope.q)
        k_s = self._camera_slot(self._project(self.phi_k, k), prope.kv)
        v_s = self._camera_slot(self._project(self.phi_v, v), prope.kv)
        q_s = l2_norm(apply_rope(q_s, rec_cos, rec_sin))
        k_s = l2_norm(apply_rope(k_s, rec_cos, rec_sin))

        S0 = state.get("S")
        if S0 is None:
            S0 = torch.zeros((batch, self.num_heads, self.head_dim, self.head_dim), device=q.device,
                             dtype=torch.float32)
        else:
            S0 = S0.clone()
        # Read the committed history (before this chunk is written).
        inter = torch.einsum("bthd,bhdv->bthv", q_s, S0.type_as(q_s)) * self.state_scale
        if commit:
            g = self._decay(x)
            beta = self.b_proj(x).sigmoid().float() * 2
            _, S1 = chunk_kda(q=q_s, k=k_s, v=v_s, g=g, beta=beta, scale=self.state_scale,
                              initial_state=S0, output_final_state=True)
            state["S"] = S1.float().detach().clone()
            state["chunks"] = state.get("chunks", 0) + 1
        inter = self._camera_slot(inter, prope.o)
        gate = self.gate_proj(x).sigmoid().view(batch, x.shape[1], self.num_heads, 1)
        return intra + gate * inter.to(intra.dtype)
