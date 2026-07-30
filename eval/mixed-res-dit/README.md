# Phase-Aligned RoPE for Mixed-Resolution Diffusion Transformer

### [Project page](https://hao-yu-wu.github.io/mixed_res/) | [Paper](https://arxiv.org/abs/2511.19778)

This is the official implementation of **Phase-Aligned RoPE for Mixed-Resolution Diffusion Transformer**.

European Conference on Computer Vision (ECCV), 2026

## Installation

1. Environment

```bash
micromamba create -n mixed_res -c conda-forge python=3.11 pip
micromamba activate mixed_res
pip install -r requirements.txt
```

2. Checkpoints

Download [`ckpt.zip`](https://drive.google.com/file/d/1LO3IucXFFWdMfB_ez0GN5W9dPYi5I7Zp/view?usp=sharing) and unzip it into `ckpts/`.

## Demo

- Video generation (Wan 2.1)

```bash
python -m wan.demo
```

- Image generation (FLUX.1-dev)

```bash
python -m flux.demo       # 3-stage
python -m flux.demo_2stg  # 2-stage
```

- Reproduce Paper Results

```bash
# Video generation
python -m wan.demo --seed 0

# Image generation
python -m flux.demo --seed 0 --speed_up '4x'
python -m flux.demo --seed 0 --speed_up '6x'
python -m flux.demo_2stg --seed 0 --mask_type 'base'
```

## Limitations and Future Work

See [assets/limitations-and-future-work.md](assets/limitations-and-future-work.md).

## Acknowledgements

- We are grateful to [Meher Gitika Karumuri](https://openreview.net/profile?id=~Meher_Gitika_Karumuri1), [Brandon Smith](https://pages.cs.wisc.edu/~bmsmith/), [Amogh Gupta](https://www.linkedin.com/in/amogh112), and [Vidya Narayanan](https://vid8687.github.io/) for their insightful comments and valuable discussions. This work was supported in part by NSF grants IIS-2123920 and IIS-2212046.

- We borrowed code from [diffusers](https://github.com/huggingface/diffusers), [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev), and [RALU](https://github.com/ignoww/RALU). We thank all the authors for their great work and repos.

## Citation

If you find our code useful for your research, please cite
```bibtex
@article{wu2025one,
  title={One Attention, One Scale: Phase-Aligned Rotary Positional Embeddings for Mixed-Resolution Diffusion Transformer},
  author={Wu, Haoyu and Xu, Jingyi and Miao, Qiaomu and Samaras, Dimitris and Le, Hieu},
  journal={arXiv preprint arXiv:2511.19778},
  year={2025}
}
```
