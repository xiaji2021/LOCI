# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Attention primitives shared by the LOCI blocks."""
from __future__ import annotations

import functools
import os
import warnings

import torch
import torch.nn.functional as F
from einops import rearrange

from .third_party.prope import PropeDotProductAttention, _prepare_apply_fns

try:  # FlashAttention-3 (Hopper). Used for the intra-chunk attention of the hybrid blocks.
    from flash_attn_interface import flash_attn_func as _fa3_func
except ImportError:  # pragma: no cover - depends on the local install
    _fa3_func = None

_WARNED = False


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary embedding in the Wan layout, as in diffusers' ``WanAttnProcessor`` (Apache-2.0).

    ``x``: [B, L, H, D]; cos/sin: [*, L, 1, D]. Arithmetic follows the dtype of ``cos``/``sin``.
    """
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = cos[..., 0::2]
    sin = sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


def sdpa_bshd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full (non-causal) attention on [B, S, H, D] tensors via torch SDPA."""
    q = rearrange(q, "b s h d -> b h s d")
    k = rearrange(k, "b s h d -> b h s d")
    v = rearrange(v, "b s h d -> b h s d")
    out = F.scaled_dot_product_attention(q, k, v)
    return rearrange(out, "b h s d -> b s h d")


def chunk_local_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bidirectional attention inside the current chunk, [B, L, H, D] in and out.

    The reference implementation uses FlashAttention-3 when available (bit-identical to the
    paper runs); otherwise torch SDPA is used, which matches up to bf16 rounding.
    """
    global _WARNED
    backend = os.environ.get("LOCI_LOCAL_ATTN", "fa3" if _fa3_func is not None else "sdpa")
    if backend == "fa3" and _fa3_func is not None and q.is_cuda:
        out = _fa3_func(q, k, v, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out.contiguous()
    if not _WARNED:
        warnings.warn("FlashAttention-3 unavailable: intra-chunk attention falls back to torch SDPA "
                      "(numerically close, not bit-identical to the reference outputs).")
        _WARNED = True
    return sdpa_bshd(q, k, v).contiguous()


@functools.lru_cache(maxsize=None)
def _prope_coeff_module(head_dim: int, patches_x: int, patches_y: int, image_scale: int):
    # The RoPE coefficients are computed once on CPU in float32 (as in the reference).
    return PropeDotProductAttention(head_dim=head_dim, patches_x=patches_x, patches_y=patches_y,
                                    image_width=patches_x * image_scale, image_height=patches_y * image_scale,
                                    precompute_coeffs=True)


class PropeTransforms:
    """PRoPE block-diagonal transforms (projective + 2-D RoPE) for per-ray cameras.

    ``cast_dtype``: if given, the precomputed matrices / coefficients are cast to it and the
    transforms run in that dtype; otherwise they run in float32 on float32 inputs.
    """

    def __init__(self, mats: torch.Tensor, K: torch.Tensor | None, *, head_dim: int, grid_hw, image_scale: int,
                 cast_dtype: torch.dtype | None = None):
        patches_y, patches_x = grid_hw
        ref = _prope_coeff_module(head_dim, patches_x, patches_y, image_scale)
        device = mats.device
        coeffs_x = (ref.coeffs_x_0.to(device), ref.coeffs_x_1.to(device))
        coeffs_y = (ref.coeffs_y_0.to(device), ref.coeffs_y_1.to(device))
        with torch.autocast(device_type=device.type, enabled=False):
            self.q, self.kv, self.o = _prepare_apply_fns(
                head_dim=head_dim, viewmats=mats, Ks=K, patches_x=patches_x, patches_y=patches_y,
                image_width=patches_x * image_scale, image_height=patches_y * image_scale,
                coeffs_x=coeffs_x, coeffs_y=coeffs_y,
            )
        if cast_dtype is not None:
            self.q, self.kv, self.o = (_cast_transform(f, cast_dtype) for f in (self.q, self.kv, self.o))


def _cast_transform(apply_fn, dtype):
    pairs = []
    for transform, size in apply_fn.keywords["func_size_pairs"]:
        kwargs = dict(transform.keywords)
        if "matrix" in kwargs:
            kwargs["matrix"] = kwargs["matrix"].to(dtype=dtype)
        if "coeffs" in kwargs:
            kwargs["coeffs"] = tuple(c.to(dtype=dtype) for c in kwargs["coeffs"])
        pairs.append((functools.partial(transform.func, *transform.args, **kwargs), size))
    return functools.partial(apply_fn.func, func_size_pairs=pairs)
