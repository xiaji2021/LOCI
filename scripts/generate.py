#!/usr/bin/env python
# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Generate a camera-controlled video with LOCI.

Inputs: a first frame (image) or a clean prefix (video / latents), a text prompt, and a camera
trajectory file (see examples/ and README). Output: an mp4 (16 fps) and optionally latents.

Example:
    python scripts/generate.py --weights /path/to/loci-weights --wan /path/to/Wan2.2-TI2V-5B-Diffusers \
        --image first_frame.png --prompt "a stone temple courtyard" \
        --trajectory examples/look_around.json --history sparse --output out.mp4
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loci import HistoryConfig, LociSampler, load_transformer  # noqa: E402


def _mediapy():
    """mediapy with a usable ffmpeg: the one on PATH, else the binary shipped with imageio-ffmpeg."""
    import mediapy
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    return mediapy


def load_trajectory(path):
    spec = json.loads(Path(path).read_text())
    pose = torch.tensor(spec["c2w"], dtype=torch.float32)
    if pose.shape[-2:] == (4, 4):
        pose = pose[:, :3, :4]
    if pose.ndim != 3 or pose.shape[-2:] != (3, 4):
        raise ValueError("c2w must be a list of 3x4 (or 4x4) camera-to-world matrices")
    first = pose[0]
    if (first[:, :3] - torch.eye(3)).abs().max() > 1e-4 or first[:, 3].abs().max() > 1e-4:
        raise ValueError("the first pose must be the identity (trajectory relative to the first frame)")
    return dict(pose=pose, x_fov=spec.get("x_fov_deg", 90.0), xi=spec.get("xi", 0.0),
                planner_K=spec.get("intrinsics_norm"))


def load_tensor(path):
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        data = load_file(path)
        return data[sorted(data)[0]] if len(data) == 1 else data["latents"]
    return torch.load(path, map_location="cpu", weights_only=True)


def encode_text(prompt, wan_dir, device, dtype, max_length=512, padding="max_length"):
    """UMT5 embedding [512, 4096], zero beyond the prompt tokens.

    ``padding="max_length"`` encodes the 512-token padded sequence (Wan/diffusers convention);
    ``"longest"`` encodes only the prompt tokens (numerically slightly different in bf16).
    """
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(wan_dir, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(wan_dir, subfolder="text_encoder").to(device).to(dtype).eval()
    inputs = tok([prompt], padding=padding, max_length=max_length, truncation=True,
                 add_special_tokens=True, return_attention_mask=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        emb = enc(**inputs).last_hidden_state
        emb = emb * inputs.attention_mask.unsqueeze(-1).to(emb.dtype)
    del enc
    torch.cuda.empty_cache()
    out = torch.zeros(max_length, emb.shape[-1], dtype=dtype, device=emb.device)
    out[:emb.shape[1]] = emb[0].to(dtype)
    return out


def load_frames(path, height, width, max_frames=None):
    """Image or video -> float tensor [1, 3, T, H, W] in [-1, 1] (resize + centre crop)."""
    from PIL import Image
    if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        frames = [np.asarray(Image.open(path).convert("RGB"))]
    else:
        frames = list(_mediapy().read_video(str(path)))[:max_frames]
    out = []
    for f in frames:
        img = Image.fromarray(f)
        scale = max(width / img.width, height / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
        left, top = (img.width - width) // 2, (img.height - height) // 2
        img = img.crop((left, top, left + width, top + height))
        out.append(torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1))
    return torch.stack(out, dim=1)[None]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="converted LOCI transformer directory")
    p.add_argument("--wan", required=True, help="Wan2.2-TI2V-5B-Diffusers directory or hub id (VAE + UMT5)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="first frame (image file)")
    src.add_argument("--prefix-video", help="clean video prefix to continue from (1 + 20k RGB frames used)")
    src.add_argument("--prefix-latents", help="clean normalised latents [48, P, h, w] (.pt / .safetensors)")
    p.add_argument("--prefix-frames", type=int, default=None,
                   help="with --prefix-video: use only the first N video frames (rounded down to 1 + 20k); default: all")
    txt = p.add_mutually_exclusive_group(required=True)
    txt.add_argument("--prompt")
    txt.add_argument("--prompt-embedding", help="precomputed UMT5 embedding [512, 4096] (.pt)")
    p.add_argument("--text-padding", choices=("max_length", "longest"), default="max_length")
    p.add_argument("--trajectory", required=True, help="camera trajectory JSON (one pose per latent frame)")
    p.add_argument("--num-frames", type=int, default=None,
                   help="LATENT frames to produce, 1 + 5k (one latent frame = 4 video frames = 0.25 s); "
                        "default: all poses of the trajectory")
    p.add_argument("--height", type=int, default=512, help="output height, multiple of 32 (ignored with --prefix-latents)")
    p.add_argument("--width", type=int, default=768, help="output width, multiple of 32 (ignored with --prefix-latents)")
    p.add_argument("--history", choices=("sparse", "dense"), default="sparse",
                   help="sparse: bounded sink + view bank + recent window (paper default); dense: full history")
    p.add_argument("--bank-size", type=int, default=None)
    p.add_argument("--recent", type=int, default=None)
    p.add_argument("--commit-timestep", type=float, default=None)
    p.add_argument("--pin-anchors", type=int, default=None, metavar="N",
                   help="sparse only: pin the first N early frames (one every 2 s within the first 20 s) in the view "
                        "bank so they are never evicted; reduces long-horizon drift. Default 10; 0 = off (paper setting)")
    p.add_argument("--no-pin", action="store_true", help="same as --pin-anchors 0")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decode", choices=("all", "generated"), default="all",
                   help="decode the full timeline or only the newly generated latents")
    p.add_argument("--output", required=True, help="output .mp4")
    p.add_argument("--save-latents", action="store_true")
    args = p.parse_args()

    # Validate the cheap things before any model is loaded.
    if not torch.cuda.is_available():
        p.error("a CUDA GPU is required")
    for name in ("image", "prefix_video", "prefix_latents", "prompt_embedding", "trajectory"):
        value = getattr(args, name)
        if value and not Path(value).is_file():
            p.error(f"--{name.replace('_', '-')}: file not found: {value}")
    if not (Path(args.weights) / "config.json").is_file():
        p.error(f"--weights: {args.weights} is not a LOCI weights directory (no config.json)")
    if args.pin_anchors is not None and not 0 <= args.pin_anchors < (args.bank_size or 20):
        p.error("--pin-anchors must be >= 0 and smaller than the bank size")
    if args.height % 32 or args.width % 32:
        p.error("--height and --width must be multiples of 32")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    traj = load_trajectory(args.trajectory)
    num_frames = args.num_frames or len(traj["pose"])
    if num_frames < 6 or (num_frames - 1) % 5:
        p.error(f"the number of latent frames must be 1 + 5k (k >= 1 chunks), got {num_frames} "
                f"({'--num-frames' if args.num_frames else 'poses in --trajectory'})")
    if num_frames > len(traj["pose"]):
        p.error(f"--num-frames {num_frames} exceeds the {len(traj['pose'])} poses in --trajectory")

    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    vae = AutoencoderKLWan.from_pretrained(args.wan, subfolder="vae").to(device).to(dtype).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)

    text = (encode_text(args.prompt, args.wan, device, dtype, padding=args.text_padding) if args.prompt
            else load_tensor(args.prompt_embedding).reshape(-1, 512, 4096)[0])

    if args.prefix_latents:
        prefix = load_tensor(args.prefix_latents)
        prefix = prefix[None] if prefix.ndim == 4 else prefix
    else:
        path = args.image or args.prefix_video
        pixels = load_frames(path, args.height, args.width, args.prefix_frames)
        n = pixels.shape[2]
        n = 1 + ((n - 1) // 20) * 20  # 1 + 4*5k RGB frames -> 1 + 5k latent frames
        with torch.inference_mode():
            z = vae.encode(pixels[:, :, :n].to(device=device, dtype=dtype)).latent_dist.mode()
        prefix = (z - mean) / std
    prefix = prefix.to(device=device, dtype=dtype)
    if prefix.shape[2] >= num_frames or (prefix.shape[2] - 1) % 5:
        p.error(f"the clean prefix has {prefix.shape[2]} latent frames; it must be 1 + 5k and shorter than "
                f"the {num_frames} latent frames to produce")

    model = load_transformer(args.weights, device=device, dtype=dtype)
    cfg = HistoryConfig.preset(args.history)
    if args.bank_size is not None:
        cfg.bank_size = args.bank_size
    if args.recent is not None:
        cfg.recent = args.recent
    if args.commit_timestep is not None:
        cfg.commit_timestep = args.commit_timestep
    if args.pin_anchors is not None:
        cfg.pin_anchors = args.pin_anchors
    if args.no_pin:
        cfg.pin_anchors = 0
    sampler = LociSampler(model, steps=args.steps)
    chunks = (num_frames - prefix.shape[2]) // 5
    print(f"generating {chunks} chunks ({chunks * 20} frames, {chunks * 1.25:.1f} s of video) at "
          f"{prefix.shape[-1] * 16}x{prefix.shape[-2] * 16}, history={cfg.mode}"
          f"{f' (pinned anchors {cfg.pin_anchors})' if cfg.mode == 'sparse' else ''}, steps={args.steps}, seed={args.seed}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    latents = sampler.generate(text=text, pose=traj["pose"], x_fov=traj["x_fov"], xi=traj["xi"], prefix=prefix,
                               num_frames=num_frames, history_cfg=cfg, planner_K_norm=traj["planner_K"],
                               seed=args.seed,
                               progress=lambda i, n: print(f"chunk {i}/{n}  {time.time() - t0:.0f}s", flush=True))
    print(f"generation: {time.time() - t0:.1f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
          f"allocated / {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_latents:
        from safetensors.torch import save_file
        save_file({"latents": latents.cpu().contiguous()}, str(out.with_suffix(".latents.safetensors")))
    decode = latents if args.decode == "all" else latents[:, :, prefix.shape[2]:]
    del model
    torch.cuda.empty_cache()
    with torch.inference_mode():
        video = vae.decode(decode * std + mean, return_dict=False)[0]
    frames = VideoProcessor(vae_scale_factor=16).postprocess_video(video)
    _mediapy().write_video(str(out), np.concatenate(frames, axis=0), fps=16)
    print(f"saved {out} (decode included: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated)")


if __name__ == "__main__":
    main()
