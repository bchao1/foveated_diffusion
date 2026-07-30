# Importance-Based Token Merging for Efficient Image and Video Generation

### [Project page](https://hao-yu-wu.github.io/token_merging/) | [Paper](https://arxiv.org/abs/2411.16720)

This is the official implementation of **Importance-Based Token Merging for Efficient Image and Video Generation**.

International Conference on Computer Vision (ICCV), 2025

## Installation

```bash
conda create -n imp_tome python=3.10
conda activate imp_tome
pip install -r requirements.txt
```

## Demo
- For Stable Diffusion

    ```bash
    python demo_sd.py
    ```

- For PixArt-alpha
    ```bash
    python demo_pixart.py
    ```

## Acknowledgements

- This work was supported in part by the NASA Biodiversity Program (Award 80NSSC21K1027), and NSF Grant IIS-2212046.

- We borrowed code from [tomesd](https://github.com/dbolya/tomesd), [diffusers](https://github.com/huggingface/diffusers), [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha), [zero123plus](https://github.com/SUDO-AI-3D/zero123plus), and [AnimateDiff](https://github.com/guoyww/AnimateDiff). We thank all the authors for their great work and repos.

## Citation

If you find our code useful for your research, please cite
```
@article{wu2024importance,
  title={Importance-Based Token Merging for Efficient Image and Video Generation},
  author={Wu, Haoyu and Xu, Jingyi and Le, Hieu and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2411.16720},
  year={2024}
}
```