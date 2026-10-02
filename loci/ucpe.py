# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""UCPE-style camera attention branch (one per transformer block).

A narrow self-attention whose q/k/v are transformed by PRoPE with per-ray camera frames
("relative ray" encoding) and whose input is augmented with an absolute up/latitude map.
Its (zero-initialised) output projection is added to the block's self-attention output.
Design follows UCPE (https://github.com/chengzhag/UCPE, MIT license); this is an
independent implementation integrated with LOCI's chunk-wise history cache.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PropeTransforms


class CameraAttention(nn.Module):
    def __init__(self, dim: int, attn_dim: int, num_heads: int, emb_dim: int = 3, image_scale: int = 32):
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.image_scale = image_scale
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        self.cam_encoder = nn.Linear(emb_dim, dim)

    def forward(self, x: torch.Tensor, camera: dict, history) -> torch.Tensor:
        """``x``: normalised block input [B, L, C]; ``history``: callable(key_name, tensor[B,L,H,D]) ->
        tensor[B, L_hist + L, H, D] that stores the current chunk and returns the visible K/V."""
        b, n, _ = x.shape
        input_dtype = x.dtype
        x = x + self.cam_encoder(camera["abs_map"].to(device=x.device, dtype=x.dtype))
        prope = PropeTransforms(camera["ray_mats"].to(device=x.device, dtype=torch.float32), None,
                                head_dim=self.head_dim, grid_hw=camera["grid"][1:],
                                image_scale=self.image_scale, cast_dtype=x.dtype)
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q = prope.q(q)
        k = prope.kv(k)
        v = prope.kv(v)
        k = history("camera_k", k.transpose(1, 2)).transpose(1, 2)
        v = history("camera_v", v.transpose(1, 2)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = prope.o(out)
        out = out.transpose(1, 2).reshape(b, n, self.attn_dim).to(dtype=input_dtype)
        return self.out_proj(out)
