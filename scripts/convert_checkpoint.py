#!/usr/bin/env python
# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Convert a LOCI training checkpoint into the release layout.

Input : a training checkpoint directory (``model.safetensors.index.json`` + shards) whose
        transformer weights are stored under the ``model.`` prefix.
Output: a diffusers-style directory (``config.json`` + sharded safetensors) loadable with
        ``loci.load_transformer``.

Key mapping (everything else is unchanged; base weights keep the diffusers Wan names):
    model.<name>                          -> <name>
    blocks.N.cam_self_attn.*              -> blocks.N.camera_attn.*
    blocks.N.attn1.gdn.*                  -> blocks.N.kda.*
    negative_embeddings                   -> dropped (unused at guidance 1; recompute with UMT5 if needed)
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RENAMES = [
    (re.compile(r"^model\."), ""),
    (re.compile(r"\.cam_self_attn\."), ".camera_attn."),
    (re.compile(r"\.attn1\.gdn\."), ".kda."),
]
DROP = {"negative_embeddings"}


def convert_key(key: str):
    if key in DROP:
        return None
    for pattern, repl in RENAMES:
        key = pattern.sub(repl, key)
    return key


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="training checkpoint directory")
    p.add_argument("--dst", type=Path, required=True, help="output directory")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="storage dtype (inference runs in bf16 either way; bf16 halves the size)")
    p.add_argument("--shard-gb", type=float, default=5.0)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loci.transformer import LociTransformer3DModel

    src_cfg = json.loads((args.src / "config.json").read_text())
    kw = src_cfg.get("model_kwargs", {})
    hybrid = list(kw.get("hybrid_layers", []))
    config = dict(hybrid_layers=hybrid, chunk_size=int(src_cfg.get("denoising_chunk_size", 5)),
                  camera_attn_compress=int(kw.get("ucpe_attn_compress", 4)),
                  camera_image_scale=int(kw.get("ucpe_prope_image_scale", 32)),
                  kda_min_retention=float(kw.get("gdn_chunk_decay_min_retention", 0.2)))
    with torch.device("meta"):
        ref = LociTransformer3DModel(**config)
    expected = dict(ref.state_dict())

    index = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    out, seen = {}, set()
    for shard in sorted(set(index.values())):
        with safe_open(str(args.src / shard), framework="pt") as f:
            for key in f.keys():
                new = convert_key(key)
                if new is None:
                    continue
                if new not in expected:
                    raise KeyError(f"unexpected tensor {key} -> {new}")
                t = f.get_tensor(key)
                if tuple(t.shape) != tuple(expected[new].shape):
                    raise ValueError(f"shape mismatch {new}: {tuple(t.shape)} vs {tuple(expected[new].shape)}")
                out[new] = t.to(dtype).contiguous()
                seen.add(new)
    missing = sorted(set(expected) - seen)
    if missing:
        raise KeyError(f"{len(missing)} parameters missing, e.g. {missing[:5]}")

    args.dst.mkdir(parents=True, exist_ok=True)
    limit = int(args.shard_gb * 2**30)
    shards, cur, size = [], {}, 0
    for key in sorted(out):
        n = out[key].numel() * out[key].element_size()
        if cur and size + n > limit:
            shards.append(cur)
            cur, size = {}, 0
        cur[key] = out[key]
        size += n
    shards.append(cur)
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        name = f"diffusion_pytorch_model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(args.dst / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in shard})
    total = sum(t.numel() * t.element_size() for t in out.values())
    (args.dst / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    ref.register_to_config(**config)
    ref.save_config(str(args.dst))
    print(f"wrote {len(out)} tensors ({total / 2**30:.2f} GiB, {args.dtype}) to {args.dst}")


if __name__ == "__main__":
    main()
