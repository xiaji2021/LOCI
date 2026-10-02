# Copyright 2026 The LOCI Authors.
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LOCI transformer: Wan2.2-TI2V-5B with chunk-causal memory and camera conditioning.

Built from the official Wan2.2 components as packaged in diffusers
(``diffusers.models.transformers.transformer_wan``: time/text embedding, RoPE tables,
attention projections, feed-forward, output head). LOCI additions:

* chunk-wise causal generation with a history K/V cache (:mod:`loci.memory`);
* hybrid blocks (``hybrid_layers``) whose self-attention is intra-chunk softmax plus a
  KDA recurrent memory (:mod:`loci.kda`); the other blocks attend to the planned history;
* a camera attention branch in every block (:mod:`loci.ucpe`).

Every forward processes exactly one chunk (the 1-frame condition chunk or a 5-frame chunk).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from einops import rearrange

from .attention import apply_rope, sdpa_bshd
from .camera import build_camera_inputs
from .kda import KDAMemory
from .ucpe import CameraAttention

PAPER_HYBRID_LAYERS = (2, 4, 6, 7, 8, 9, 11, 13, 14, 16, 23, 24, 25, 27, 28)


class ChunkRotaryPosEmbed(WanRotaryPosEmbed):
    """Wan 3-D RoPE tables evaluated for a chunk that starts at temporal position ``start``."""

    def forward(self, shape, start: int = 0):
        _, _, num_frames, height, width = shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        split = [self.attention_head_dim - 2 * (self.attention_head_dim // 3),
                 self.attention_head_dim // 3, self.attention_head_dim // 3]
        cos = self.freqs_cos.split(split, dim=1)
        sin = self.freqs_sin.split(split, dim=1)
        frames = slice(start, start + ppf)

        def grid(parts):
            f = parts[0][frames].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
            h = parts[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
            w = parts[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
            return torch.cat([f, h, w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return grid(cos), grid(sin)


def _modulate(x, scale, shift):
    b, f, _, c = scale.shape
    return (x.view(b, f, -1, c) * (1 + scale) + shift).view(b, -1, c)


def _gated_residual(x, y, gate):
    b, f, _, c = gate.shape
    return (x.view(b, f, -1, c) + y.view(b, f, -1, c) * gate).view(b, -1, c)


class LociBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, *, eps, qk_norm, cross_attn_norm, hybrid: bool,
                 camera_compress: int, camera_image_scale: int, kda_kwargs: dict):
        super().__init__()
        attn_kwargs = dict(query_dim=dim, heads=num_heads, kv_heads=num_heads, dim_head=dim // num_heads,
                           qk_norm=qk_norm, eps=eps, bias=True, cross_attention_dim=None, out_bias=True)
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(**attn_kwargs)
        self.attn2 = Attention(**attn_kwargs, added_kv_proj_dim=None, added_proj_bias=True)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.camera_attn = CameraAttention(dim, dim // camera_compress, num_heads // camera_compress,
                                           emb_dim=3, image_scale=camera_image_scale)
        self.kda = KDAMemory(dim, num_heads, **kda_kwargs) if hybrid else None

    def _qkv(self, x):
        attn = self.attn1
        q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        q, k = attn.norm_q(q), attn.norm_k(k)
        e = attn.out_dim // attn.heads
        split = lambda t: rearrange(t, "b fhw (n e) -> b fhw n e", e=e)
        return split(q), split(k), split(v)

    def _cross(self, x, text, frames):
        attn = self.attn2
        q, k, v = attn.to_q(x), attn.to_k(text), attn.to_v(text)
        q, k = attn.norm_q(q), attn.norm_k(k)
        e = attn.out_dim // attn.heads
        b = q.size(0)
        q = rearrange(q, "b (f hw) (h e) -> (b f) h hw e", f=frames, e=e)
        k = rearrange(k, "b f s (h e) -> (b f) h s e", e=e)
        v = rearrange(v, "b f s (h e) -> (b f) h s e", e=e)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b f) h hw e -> b (f hw) (h e)", b=b).type_as(q)
        return attn.to_out[1](attn.to_out[0](out))

    def forward(self, x, text, temb, rope, camera, *, index: int, history, kda_state: dict, commit: bool):
        frames = camera["grid"][0]
        shift, scale, gate, c_shift, c_scale, c_gate = (self.scale_shift_table.unsqueeze(1) + temb.float()).chunk(6, dim=2)
        h = _modulate(self.norm1(x.float()), scale, shift).type_as(x)

        q, k, v = self._qkv(h)
        if self.kda is not None:
            out = self.kda(h, q, k, v, rope, camera, kda_state, commit)
        else:
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)
            k = history.cache(index, "main_k", k)
            v = history.cache(index, "main_v", v)
            out = sdpa_bshd(q, k, v)
        out = rearrange(out, "b fhw n e -> b fhw (n e)")
        out = self.attn1.to_out[1](self.attn1.to_out[0](out))
        out = out + self.camera_attn(h, camera, lambda name, t: history.cache(index, name, t))
        x = _gated_residual(x.float(), out, gate).type_as(x)

        x = x + self._cross(self.norm2(x.float()).type_as(x), text, frames)

        h = _modulate(self.norm3(x.float()), c_scale, c_shift).type_as(x)
        x = _gated_residual(x.float(), self.ffn(h).float(), c_gate).type_as(x)
        return x


class LociTransformer3DModel(ModelMixin, ConfigMixin):
    _no_split_modules = ["LociBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        chunk_size: int = 5,
        hybrid_layers=PAPER_HYBRID_LAYERS,
        camera_attn_compress: int = 4,
        camera_image_scale: int = 32,
        kda_min_retention: float = 0.2,
        kda_image_scale: int = 32,
        kda_translation_scale: float = 6.0,
    ):
        super().__init__()
        inner = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.patch_embedding = nn.Conv3d(in_channels, inner, kernel_size=patch_size, stride=patch_size)
        self.rope = ChunkRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.condition_embedder = WanTimeTextImageEmbedding(dim=inner, time_freq_dim=freq_dim,
                                                            time_proj_dim=inner * 6, text_embed_dim=text_dim)
        kda_kwargs = dict(min_retention=kda_min_retention, prope_image_scale=kda_image_scale,
                          translation_scale=kda_translation_scale)
        self.blocks = nn.ModuleList([
            LociBlock(inner, ffn_dim, num_attention_heads, eps=eps, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm,
                      hybrid=i in set(hybrid_layers), camera_compress=camera_attn_compress,
                      camera_image_scale=camera_image_scale, kda_kwargs=kda_kwargs)
            for i in range(num_layers)
        ])
        self.norm_out = FP32LayerNorm(inner, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner, out_channels * patch_size[0] * patch_size[1] * patch_size[2])
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner) / inner ** 0.5)

    # The reference runs used torch.compile for the patch embedding; kept for bit-identical outputs.
    def _embed(self, x):
        y = self.patch_embedding(x)
        return y.flatten(2).transpose(1, 2).contiguous()

    def forward(self, latents, timestep, text, pose, x_fov, xi, *, history, kda_states, commit: bool):
        """Denoise (or commit) one chunk.

        latents  [B, C, F, H, W]  current chunk (noisy, or clean/lightly noised when committing)
        timestep []               flow-matching timestep of the chunk (0..1000)
        text     [B, F, 512, D]   per-frame UMT5 embeddings
        pose     [B, F, 3|4, 4]   camera-to-world, x_fov/xi: [B, F] (degrees / UCM xi)
        history  HistoryState (``begin_chunk`` already called for this chunk)
        """
        b, _, f, hh, ww = latents.shape
        p_t, p_h, p_w = self.config.patch_size
        grid = (f // p_t, hh // p_h, ww // p_w)
        rope = self.rope(latents.shape, history.rope_start)
        x = _EMBED(self, latents) if _COMPILE else self._embed(latents)
        tokens = x.shape[1]
        t = timestep.reshape(1, 1).expand(b, f)
        t = t.unsqueeze(-1).expand(b, f, tokens // f).reshape(b, tokens)

        camera = build_camera_inputs(pose, x_fov, xi, grid,
                                     recurrent_translation_scale=self.config.kda_translation_scale,
                                     recurrent_image_scale=self.config.kda_image_scale, model_dtype=x.dtype)
        camera["grid"] = grid
        temb, tproj, text, _ = self.condition_embedder(t.flatten(), text, None)
        temb = temb.view(b, tokens, 1, -1)
        tproj = tproj.view(b, tokens, 6, -1)

        for i, block in enumerate(self.blocks):
            x = block(x, text, tproj, rope, camera, index=i, history=history,
                      kda_state=kda_states[i], commit=commit)

        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        x = _modulate(self.norm_out(x.float()), scale, shift).type_as(x)
        x = self.proj_out(x)
        x = rearrange(x, "b (t hw) (ph pw c) -> b c t hw ph pw", b=b, t=grid[0], ph=p_h, pw=p_w)
        return rearrange(x, "b c t (h w) ph pw -> b c t (h ph) (w pw)", h=grid[1], w=grid[2], ph=p_h, pw=p_w)


def _embed_impl(model, x):
    return model._embed(x)


_COMPILE = os.environ.get("LOCI_COMPILE_PATCH_EMBED", "1") == "1"
_EMBED = torch.compile(_embed_impl) if _COMPILE else _embed_impl
