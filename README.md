<div align="center">

# LOCI

### Spatial Linear Memory for Streaming World Models

**Ji Xia**<sup>1</sup> · **Tingting Liao**<sup>1</sup> · **Xuezhi Liang**<sup>1</sup> · **Hao Li**<sup>2,3</sup> · **Guangyi Liu**<sup>1,†</sup>

<sup>1</sup>Institute of Foundation Models, MBZUAI &nbsp;&nbsp; <sup>2</sup>MBZUAI &nbsp;&nbsp; <sup>3</sup>Pinscreen &nbsp;&nbsp; <sup>†</sup>Corresponding author

<a href="https://xiaji2021.github.io/LOCI/"><img src="https://img.shields.io/badge/Project-Page-f2b35b?style=for-the-badge" alt="Project page"></a>
<a href="https://arxiv.org/abs/2609.40222"><img src="https://img.shields.io/badge/arXiv-2609.40222-b31b1b?style=for-the-badge" alt="arXiv"></a>
<a href="https://huggingface.co/papers/2609.40222"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-Daily%20Papers-ffd21e?style=for-the-badge" alt="HF paper"></a>
<a href="https://huggingface.co/datasets/sum0214/LOCI-revisit-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-LOCI--revisit--data-ffd21e?style=for-the-badge" alt="Dataset"></a>
<img src="https://img.shields.io/badge/Checkpoints-coming%20soon-lightgrey?style=for-the-badge" alt="Checkpoints coming soon">

<img src="assets/teaser.jpg" width="92%" alt="Returning to a previously seen place: LOCI restores what was there, a same-recipe full-softmax model does not.">

<sub><i>Look away. Come back. It's still there.</i> — the camera returns to a place it saw earlier; LOCI restores it, a full-softmax model trained with the same recipe does not.</sub>

</div>

---

**LOCI** is a long-horizon, camera-controlled video world model built on
[Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2). Given a first frame (or a clean video prefix), a text prompt
and a camera trajectory, it generates the video chunk by chunk while remembering what it has already seen, so a
revisit shows the same world.

This repository contains the **inference code**. Model weights are coming soon; training code is not part of this release.

## ✨ How it works

<table>
<tr><td width="34%"><b>Hybrid memory blocks</b><br><sub>15 of 30 blocks</sub></td><td>Intra-chunk softmax attention plus a Kimi-Delta-Attention (KDA) recurrent state that summarises <i>all</i> committed history with chunk-level retention. Reads and writes carry per-ray PRoPE camera geometry. <code>loci/kda.py</code></td></tr>
<tr><td><b>History softmax blocks</b><br><sub>the other 15</sub></td><td>Attend to a history K/V cache: the full history (<code>dense</code>), or a bounded set (<code>sparse</code>: first frame + a view bank + the most recent frames) at constant memory. <code>loci/memory.py</code></td></tr>
<tr><td><b>Camera branch</b><br><sub>all 30 blocks</sub></td><td>UCPE-style relative-ray PRoPE attention with an absolute up/latitude map. <code>loci/ucpe.py</code>, <code>loci/camera.py</code></td></tr>
<tr><td><b>Read-then-commit</b></td><td>Each chunk is denoised while reading the committed history; a separate commit forward then writes it into the caches and the recurrent state. <code>loci/pipeline.py</code></td></tr>
</table>

## 📦 Dataset

[**LOCI-revisit-data**](https://huggingface.co/datasets/sum0214/LOCI-revisit-data) — game-engine video with exact
camera poses, depth and revisit pairs: 199 sequences, 70 maps, ~24 h. Each subset is released under the license of
its source scenes (see the dataset card).

## 🚀 Quick start

```bash
# install
conda create -n loci python=3.10 -y && conda activate loci
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# a camera trajectory: look around, 16 chunks = 20 s
python examples/make_trajectory.py --preset look_around --chunks 16 --output traj.json

# generate from one image + a prompt
python scripts/generate.py --weights /path/to/loci-weights --wan Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --image first_frame.png --prompt "A sunlit stone temple courtyard with red lanterns." \
    --trajectory traj.json --history sparse --height 512 --width 768 --output out.mp4
```

Details on requirements, FlashAttention-3, weights and all options follow below.

## 🖥️ Requirements

* Linux, one NVIDIA GPU with >= 24 GB memory for 512x768 / 480x864 (bf16). Measured on one H200
  (40 s of video = 32 chunks at 480x864, 50 steps): `sparse` ~330 s and 17 GiB peak (constant in
  video length), `dense` ~490 s and 31 GiB peak (grows with video length). Decoding the 40 s clip with
  the VAE peaks at ~27 GiB; plan for ~64 GB of CPU RAM while loading the text encoder.
* CUDA 12.x driver (the PyTorch wheels below are built for CUDA 12.8), Python 3.10.
* For FlashAttention-3 (optional, Hopper only): CUDA toolkit >= 12.3 with `nvcc`, a C++17 host compiler
  (g++ >= 9), `git` and network access to GitHub while building.

## 🔧 Install

```bash
conda create -n loci python=3.10 -y && conda activate loci
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`flash-linear-attention==0.5.2` provides the KDA kernels (Triton; any recent NVIDIA GPU). No system
`ffmpeg` is needed (the binary shipped with `imageio-ffmpeg` is used when none is on `PATH`).

<details>
<summary><b>FlashAttention-3</b> (recommended on H100/H200)</summary>

It is used for the intra-chunk attention and is
needed to reproduce the reference outputs bit-for-bit. Build it from source (about 30 min with 48 cores;
the build fetches the CUTLASS submodule itself):

```bash
pip install ninja packaging
export CUDA_HOME=/usr/local/cuda            # toolkit with nvcc >= 12.3
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper                   # tested: commit fb97d25 (git checkout fb97d25 to pin)
MAX_JOBS=16 python setup.py install         # each compile job needs a few GB of RAM; lower MAX_JOBS if it is killed
python -c "import flash_attn_interface"     # check (run outside the hopper/ directory)
```

</details>

Without FlashAttention-3 the intra-chunk attention falls back to PyTorch SDPA (a warning is printed;
`LOCI_LOCAL_ATTN=sdpa` forces it). Speed and memory are about the same and the outputs are equally
valid samples, but because generation is autoregressive they drift away from the reference samples
over long rollouts (different details, same scene).

The patch embedding is compiled with `torch.compile` (needed for bit-identical outputs). If
compilation fails on your system (e.g. no C compiler), set `LOCI_COMPILE_PATCH_EMBED=0`.

## ⚖️ Weights

* Base assets from `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (Hugging Face): only the VAE, the UMT5 text
  encoder and the tokenizer are used (~14 GB; the Wan transformer is not needed). `--wan` takes the
  hub id (downloaded on first use) or a local directory. On machines without internet access,
  download once and pass the directory:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*" "text_encoder/*" "tokenizer/*" \
    --local-dir ./Wan2.2-TI2V-5B-Diffusers
export HF_HUB_OFFLINE=1   # on the offline machine
```

* LOCI transformer weights (~11 GB, bf16): **coming soon** (will be released on Hugging Face). Pass the
  directory with `--weights`; it must contain

```
loci-weights/
  config.json
  diffusion_pytorch_model.safetensors.index.json
  diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors
```

  If you have a training checkpoint instead, convert it:

```bash
python scripts/convert_checkpoint.py --src /path/to/training/checkpoint --dst /path/to/loci-weights --dtype bf16
```

## 🎬 Usage

```bash
# 1) a camera trajectory (one pose per latent frame; see "Trajectory format")
#    presets: look_around, forward_back, orbit_return, walk_turn; 16 chunks = 20 s, 32 chunks = 40 s
python examples/make_trajectory.py --preset look_around --chunks 16 --output traj.json

# 2) generate from a first frame + prompt (paper default: bounded sparse history)
python scripts/generate.py --weights /path/to/loci-weights --wan Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --image first_frame.png --prompt "A sunlit stone temple courtyard with red lanterns." \
    --trajectory traj.json --history sparse --height 512 --width 768 --output out.mp4

# full (dense) history instead of the bounded memory
python scripts/generate.py ... --history dense

# continue a real video: the first 1 + 20k frames are used as clean history (here 81 frames = 5 s),
# the trajectory covers the prefix too (21 latent frames for 81 video frames)
python examples/make_trajectory.py --preset walk_turn --chunks 16 --output traj_including_prefix.json
python scripts/generate.py ... --prefix-video clip.mp4 --prefix-frames 81 --trajectory traj_including_prefix.json
```

`first_frame.png` is any RGB image (it is resized and centre-cropped to `--height` x `--width`);
set `--fov` of `make_trajectory.py` to the horizontal field of view of the cropped image (default
90 deg). The prompt should describe the scene. Progress, generation time and peak GPU memory are
printed; inputs are validated before the models are loaded.

Options: `--steps` (50), `--seed` (42), `--num-frames` (latent frames, `1 + 5k`; default: all poses),
`--bank-size` (20), `--recent` (8), `--commit-timestep` (100 for `sparse`, 0 for `dense`),
`--pin-anchors` (10; `0` or `--no-pin` turns anchor pinning off, see below),
`--decode all|generated`, `--save-latents`, `--prompt-embedding` (precomputed UMT5 embedding
`[512, 4096]`), `--text-padding max_length|longest` (UMT5 encoding convention, default `max_length`
as in Wan), `--prefix-latents` (normalised VAE latents `[48, P, h, w]`), `--prefix-frames`.
Height and width must be multiples of 32; the model was trained at 512x768 and also used at 480x864.
Harmless messages you may see: `Current Python version 3.10 is below the recommended 3.11` (fla),
`config attributes {'clip_output': False} ... will be ignored` (diffusers VAE config).

History presets (`--history`), matching the paper configurations:

| preset | softmax history | KDA state | commit forward | temporal positions |
|---|---|---|---|---|
| `sparse` (default) | frame 0 + view bank (20) + 8 most recent frames (<= 34 frames, constant memory) | all chunks | chunk re-noised to t=100 | selected frames renumbered compactly |
| `dense` | all past frames | all chunks | clean (t=0) | true positions |

**Anchor pinning (`sparse`, on by default).** Frames 1, 9, ..., 73 (one every 2 s within the first
20 s, 10 frames) are admitted to the view bank when they leave the recent window, regardless of their
new coverage, and are never evicted; the other 10 bank slots follow the coverage rule. On long
rollouts (minutes) this keeps early, clean views in memory and reduces late drift into flat or blocky
frames; it changes nothing before frame 1 leaves the recent window (~4 s). It is an inference-only
setting: `--pin-anchors 0` (or `--no-pin`) gives the paper configuration.

Guidance: the model is run without classifier-free guidance (guidance scale 1), as in the paper.

### Trajectory format

```json
{"x_fov_deg": 90.0, "xi": 0.0, "intrinsics_norm": [0.5, 0.889, 0.5, 0.5],
 "c2w": [[[1,0,0,0],[0,1,0,0],[0,0,1,0]], ...]}
```

* `c2w`: camera-to-world 3x4 (or 4x4) per **latent** frame (`1 + 5k` entries; one latent frame =
  4 output frames at 16 fps). OpenCV axes (x right, y down, z forward), translation in metres,
  the first pose must be the identity. The poses of a clean prefix are included.
* `x_fov_deg`, `xi`: camera model used by the conditioning (horizontal FOV, UCM xi; 0 = pinhole).
* `intrinsics_norm` (optional): `[fx/W, fy/H, cx/W, cy/H]`, only used to pick history views in
  `sparse` mode (defaults to the FOV-derived pinhole).

## 🔁 Reproducibility

Inference runs in bfloat16. With FlashAttention-3 and the pinned versions on an H200 the release
reproduces the research implementation used for the paper bit-for-bit (identical latents), in both
history presets (for `sparse` with `--pin-anchors 0`, the paper setting; with the default pinning it
matches the research implementation run with the same pinning). Timing and memory: see "Requirements".

## 📄 License

Code: Apache-2.0 (see `LICENSE`). Third-party components and their licenses are listed in
`THIRD_PARTY_LICENSES.md`. The Wan2.2 base weights are subject to their own license.

## 📚 Citation

```bibtex
@article{xia2026loci,
  title={LOCI: Spatial Linear Memory for Streaming World Models},
  author={Xia, Ji and Liao, Tingting and Liang, Xuezhi and Li, Hao and Liu, Guangyi},
  journal={arXiv preprint arXiv:2609.40222},
  year={2026}
}
```

## 🙏 Acknowledgements

LOCI builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (KDA kernels), [PRoPE](https://arxiv.org/abs/2507.10496), [UCPE](https://github.com/chengzhag/UCPE) and [FlashAttention](https://github.com/Dao-AILab/flash-attention). See `THIRD_PARTY_LICENSES.md`.
