# Third-party components

| Component | Used for | How it is used | License |
|---|---|---|---|
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) (TI2V-5B) | base model architecture and weights (transformer init, VAE, UMT5 text encoder) | weights downloaded separately (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`); architecture used through diffusers | Apache-2.0 |
| [diffusers](https://github.com/huggingface/diffusers) 0.36.0 | Wan transformer components (`WanTimeTextImageEmbedding`, `WanRotaryPosEmbed`, `Attention`, `FeedForward`, `FP32LayerNorm`), `AutoencoderKLWan`, `VideoProcessor` | imported (pip dependency); `loci/transformer.py` builds on these modules and keeps the upstream Apache-2.0 notice | Apache-2.0 |
| [transformers](https://github.com/huggingface/transformers) | UMT5 text encoder / tokenizer | imported | Apache-2.0 |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 0.5.2 (`fla`) | KDA chunk kernel (`fla.ops.kda.chunk_kda`) and `fla.modules.l2norm.l2_norm` | imported (pip dependency, unmodified) | MIT (c) Songlin Yang, Yu Zhang, Zhiyuan Li |
| [UCPE](https://github.com/chengzhag/UCPE) | camera-conditioning design (relative-ray PRoPE + absolute up/latitude map, UCM camera model) | `loci/third_party/prope.py` vendored unmodified (MIT header kept); `loci/camera.py` / `loci/ucpe.py` re-implement the formulation | MIT (c) 2026 Cheng Zhang |
| PRoPE ("Cameras as Relative Positional Encoding", arXiv:2507.10496) | projective positional encoding | reference implementation, included in UCPE's `thirdparty/prope/torch.py` (vendored above) | MIT |
| [FlashAttention-3](https://github.com/Dao-AILab/flash-attention) | optional kernel for intra-chunk attention (bit-exact reproduction on Hopper) | imported if installed | BSD-3-Clause |
| PyTorch, Triton, einops, safetensors, NumPy, Pillow, mediapy | runtime | imported | BSD-3 / MIT / MIT / Apache-2.0 / BSD-3 / MIT-CMU (HPND) / Apache-2.0 |

No code from other UCPE third-party directories (e.g. UniK3D, Q-Align, VIPE) is included or needed.
