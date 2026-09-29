# LOCI: Spatial Linear Memory for Streaming World Models

**Ji Xia · Tingting Liao · Xuezhi Liang · Hao Li · Guangyi Liu**

Institute of Foundation Models, Mohamed bin Zayed University of Artificial Intelligence · Mohamed bin Zayed University of Artificial Intelligence · Pinscreen

[Project page](https://xiaji2021.github.io/LOCI/) · [Paper PDF](https://xiaji2021.github.io/LOCI/assets/paper.pdf)

LOCI is a camera-controlled streaming video world model built on Wan2.2-TI2V-5B. It combines spatial recurrent linear-attention memory with a cache of past visual observations to recover previously observed content when the camera returns to a scene.

## Method

- **Spatial recurrent memory:** projective camera geometry conditions memory reads and writes in recurrent linear-attention blocks.
- **Visual history:** cache-backed attention blocks retain past observations and retrieve them using accumulated scene context.
- **Streaming generation:** chunk-by-chunk generation supports both full visual history and a bounded bank of retained observations.

## Release status

This repository currently contains the project README. Inference code, model weights, installation instructions and examples are being prepared for release.

The planned inference interface takes a first frame or a video prefix, a text prompt and a camera trajectory. Training code is not included in the planned initial release.

## Citation

```bibtex
@article{xia2026loci,
  title={LOCI: Spatial Linear Memory for Streaming World Models},
  author={Xia, Ji and Liao, Tingting and Liang, Xuezhi and Li, Hao and Liu, Guangyi},
  year={2026}
}
```
