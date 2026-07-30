"""Cross-check: OUR mask + OUR speedup-matched schedule, inside THEIR FLUX.1 pipeline.

Their code on FLUX.1-dev is known-good (control_flux1_ralu.py). This runs the same code
with only two substitutions:

  1. the Canny edge map is replaced by our foveation mask, so their existing top-k
     selection picks exactly our foveal blocks (their selection code is untouched);
  2. N is our speedup-matched schedule; e / shift / Z come from NT-DM as usual, with
     h_ori = the FLUX.1 scheduler shift (3.0), which is what their own code passes.

If this is correctly exposed, our mask and schedule logic are sound and the FLUX.2
darkness is specific to that port or to the FLUX.2 transfer. If it is dark here too, the
problem is in our mask/schedule choice and is model-independent.

2-stage needs a mixed-resolution decode (their decode assumes a full-res raster order),
so that path reorders tokens by position before handing them to their unpack.
"""
import argparse
import os

import numpy as np
import torch

import pipeline_flux_RALU_importable as RP          # their code (whitespace fix only)
from pipeline_flux_RALU_importable import FluxPipeline_RALU
from pipeline_flux2_diffsynth_RALU_fov import ralu_schedule
from RALU_NTDM_general import NT_DM
from RALU_fov_inference_diffsynth import coarse_foveation_mask

MODEL = ("/home/brianchc/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev"
         "/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
LION = ("A lion lounges calmly, displaying its powerful physique, with a soft focus "
        "highlighting its majestic mane and intense gaze.")


class MaskAsEdges:
    """Stand-in for cv2 inside their module: Canny returns the foveation mask.

    Their stage 2 pools the returned map over 16x16 blocks and takes the top-k. Feeding a
    map that is saturated exactly on the foveal blocks makes that top-k select precisely
    those blocks, with their selection code unchanged.
    """
    def __init__(self, mask_coarse):
        self.mask = mask_coarse.numpy().astype(np.uint8)

    def Canny(self, gray, threshold1=None, threshold2=None):
        h, w = gray.shape
        bh, bw = self.mask.shape
        assert h == bh * 16 and w == bw * 16, f"edge-map shape {gray.shape} vs mask {self.mask.shape}"
        return np.kron(self.mask, np.ones((16, 16), dtype=np.uint8)) * 255


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=3, choices=[2, 3])
    ap.add_argument("--mask_radius", type=float, default=0.3)
    ap.add_argument("--mask_shape", default="circular")
    ap.add_argument("--lr_downsample_factor", type=int, default=2)
    ap.add_argument("--ours_steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--prompt", default=LION)
    ap.add_argument("--output_dir", default="../../results/ralu_compare/flux1_fov")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    f0 = args.lr_downsample_factor
    mask = coarse_foveation_mask(args.height, args.width, r=args.mask_radius,
                                 shape=args.mask_shape, lr_factor=f0)
    m = int(mask.sum())
    M_down = mask.numel()

    # speedup-matched N, same solver as the FLUX.2 runs
    N, e, info = ralu_schedule(num_stages=args.stages, match_compute=True,
                               height=args.height, width=args.width, f0=f0,
                               n_hr_blocks=m, ours_steps=args.ours_steps)
    # h_ori = 3.0 for FLUX.1-dev, which is what their set_params passes (scheduler.shift)
    shift, Z = NT_DM(N, e, h_ori=3.0)
    print(f"stages={args.stages}  N={N} (total {sum(N)})  e={e}")
    print(f"  cost ratio vs ours@{args.ours_steps}: {info['cost_ratio']}  tokens/stage {info['tokens_per_stage']}")
    print(f"  NT-DM (h_ori=3.0): shift={[round(float(x),3) for x in shift]}  Z={Z:.3f}")
    print(f"  mask: {m}/{M_down} blocks ({100*m/M_down:.2f}%)")

    pipe = FluxPipeline_RALU.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # up_ratio must reproduce m exactly, since their code sizes buffers with int(M_down*up_ratio)
    up_ratio = m / M_down
    assert int(M_down * up_ratio) == m, "up_ratio does not round-trip to the mask block count"
    # their set_params calls their NT_DM internally, which hardcodes 3 stages (E[1], S[2]).
    # Swap in the K-stage generalization; test_ntdm_regression.py shows it reproduces
    # their solver to 0.00e+00 at K=3, so this is a no-op for the 3-stage path.
    RP.NT_DM = NT_DM
    pipe.set_params(use_RALU_default=False, level=None, N=N, e=e, up_ratio=up_ratio)
    pipe.params["shift"], pipe.params["Z"] = list(shift), float(Z)
    # recompute the transition coefficients for the injected Z
    s, alphas, betas = [0.0], [], []
    for i in range(len(N) - 1):
        s.append(e[i] / (Z * (1 - e[i]) + e[i]))
        alphas.append(1 - s[i + 1]); betas.append(1 / (Z * (1 - e[i]) + e[i]))
    pipe.params.update(s=s, alphas=alphas, betas=betas)
    pipe.generate_noise("cuda", args.height, args.width)

    RP.cv2 = MaskAsEdges(mask)            # the one substitution

    out = pipe(args.prompt, guidance_scale=args.guidance_scale, max_sequence_length=256,
               generator=torch.Generator("cpu").manual_seed(args.seed),
               height=args.height, width=args.width,
               output_type="pil" if args.stages == 3 else "latent")

    if args.stages == 3:
        img = out.images[0]
    else:
        # mixed-resolution decode: place tokens by position, nearest-fill the periphery
        lat = out.images if not hasattr(out, "images") else out.images
        lat = lat if torch.is_tensor(lat) else lat[0]
        hf, wf = args.height // 16, args.width // 16
        hd, wd = hf // f0, wf // f0
        flat = mask.bool().flatten()
        idx = torch.arange(M_down)
        i1x, i2x = idx[~flat], idx[flat]
        full = torch.zeros((lat.shape[0], hf, wf, lat.shape[-1]), device=lat.device, dtype=lat.dtype)
        r1, c1 = (i1x // wd) * f0, (i1x % wd) * f0
        for dy in range(f0):
            for dx in range(f0):
                full[:, r1 + dy, c1 + dx, :] = lat[:, :len(i1x), :]
        r2, c2 = (i2x // wd) * f0, (i2x % wd) * f0
        k = len(i1x)
        for j, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            full[:, r2 + dy, c2 + dx, :] = lat[:, k + j::4, :]
        packed = full.reshape(lat.shape[0], hf * wf, lat.shape[-1])
        z = pipe._unpack_latents(packed, args.height, args.width, pipe.vae_scale_factor)
        z = (z / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        img = pipe.image_processor.postprocess(pipe.vae.decode(z, return_dict=False)[0],
                                               output_type="pil")[0]

    tag = f"{args.stages}stage_r{args.mask_radius}"
    img.save(f"{args.output_dir}/lion_{tag}.png")
    v = np.asarray(img.convert("RGB")).astype(float)
    print(f"\nFLUX.1-dev + our mask + our schedule, {args.stages}-stage: mean {v.mean():.1f} std {v.std():.1f}")
    print(f"  saved {args.output_dir}/lion_{tag}.png")
    print(f"  refs: their level-4 on FLUX.1 = 72.3 mean | our FLUX.2 port 3-stage = 38.8")


if __name__ == "__main__":
    main()
