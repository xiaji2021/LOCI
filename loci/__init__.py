# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""LOCI: long-horizon camera-controlled video world model on Wan2.2-TI2V-5B (inference code)."""
from .pipeline import HistoryConfig, LociSampler
from .transformer import LociTransformer3DModel, PAPER_HYBRID_LAYERS


def load_transformer(path, device="cuda", dtype=None):
    """Load converted LOCI weights (see scripts/convert_checkpoint.py) and cast to ``dtype`` (bf16)."""
    import torch

    from .transformer import ChunkRotaryPosEmbed

    dtype = dtype or torch.bfloat16
    model = LociTransformer3DModel.from_pretrained(path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    cfg = model.config
    # Rebuild the (non-persistent) RoPE tables in float32, then round them once to the model dtype.
    model.rope = ChunkRotaryPosEmbed(cfg.attention_head_dim, tuple(cfg.patch_size), cfg.rope_max_seq_len)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


__all__ = ["HistoryConfig", "LociSampler", "LociTransformer3DModel", "PAPER_HYBRID_LAYERS", "load_transformer"]
