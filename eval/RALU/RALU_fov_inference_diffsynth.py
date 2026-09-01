"""RALU-foveated baseline on DiffSynth's FLUX.2 pipeline.

CLI mirrors eval/inference.py (mask shape/radius, lr_downsample_factor, full_eval,
subset sharding, output layout) so large-scale runs and the metrics scripts work the same
way for RALU as for our method.

Schedule: stage boundaries `e` are RALU's published values and are held fixed; the
per-stage step counts are their proportions rescaled to --num_inference_steps.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Read-only reuse so the mask, the fixation points and the prompt order are identical to ours.
from inference import (  # noqa: E402
    create_foveation_mask,
    foveation_center_for_index,
    load_prompt_dataset,
    random_mask_center_range,
)
from shard_manifest import ShardManifest, TIMING_COLUMNS, pipe_timings  # noqa: E402

from diffsynth.pipelines.flux2_image import ModelConfig
from pipeline_flux2_diffsynth_RALU_fov import (
    Flux2DiffSynthRALUFovPipeline,
    RALU_PRESETS,
    ralu_schedule,
)


def coarse_foveation_mask(height, width, center=(0.0, 0.0), r=0.3, shape="circular", lr_factor=2):
    """RALU's coarse block mask, derived from eval/inference.py:create_foveation_mask.

    RALU selects whole `lr_factor x lr_factor` blocks, so it needs the low-res grid that
    create_foveation_mask builds internally before its nearest upsample to the token grid.
    Rather than re-deriving that geometry (which lets the two drift apart), call the shared
    function and invert the nearest upsample: a stride-`lr_factor` subsample recovers the
    low-res grid exactly, because nearest upsampling by an integer factor replicates each
    cell into a `lr_factor x lr_factor` block. The round-trip assert below makes that exact.
    """
    if shape not in ("circular", "square"):
        raise ValueError(f"unsupported mask shape for the RALU baseline: {shape}")

    token_mask = create_foveation_mask(
        height, width, center=center, r=r, shape=shape,
        device=torch.device("cpu"), lr_factor=lr_factor,
    )
    coarse = token_mask[::lr_factor, ::lr_factor].contiguous().float()

    round_trip = F.interpolate(
        coarse[None, None], size=tuple(token_mask.shape), mode="nearest"
    )[0, 0]
    if not torch.equal(round_trip, token_mask):
        raise RuntimeError(
            "coarse mask does not round-trip to eval/inference.py's token mask "
            f"(shape={shape}, r={r}, lr_factor={lr_factor}); the two masks would differ."
        )
    return coarse


def build_args():
    ap = argparse.ArgumentParser(description="RALU-foveated inference on FLUX.2 (DiffSynth)")
    # Model
    ap.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    ap.add_argument("--lora_checkpoint", type=str, default=None)
    # Generation
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    # True CFG (extra negative-prompt forward pass). Must match eval/inference.py's
    # --guidance_scale, which defaults to 4.0 and is passed straight to cfg_scale.
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    # Distillation guidance embedding. klein-base has guidance_embeds=False, so this is
    # ignored; kept at 4.0 only to mirror the pipeline default our runs also leave alone.
    ap.add_argument("--embedded_guidance", type=float, default=4.0)
    # Foveation mask (same flags/semantics as inference.py)
    ap.add_argument("--full_eval_mask", type=str, default="circular",
                    choices=["circular", "square"], help="mask shape")
    ap.add_argument("--mask_radius", type=float, default=0.3)
    ap.add_argument("--lr_downsample_factor", type=int, default=2)
    ap.add_argument("--random_mask_center", type=lambda s: str(s).strip().lower()
                    in ("1", "true", "t", "yes", "y"), default=False,
                    help="Randomize the mask center per prompt instead of centering it. Uses "
                         "inference.py:foveation_center_for_index, so ours/naive/ToMe produce the "
                         "identical mask for the same prompt index and settings. m moves with the "
                         "mask, so the compute match is re-solved per distinct m.")
    ap.add_argument("--mask_center_seed", type=int, default=42,
                    help="Seed for --random_mask_center; must match the other baselines.")
    ap.add_argument("--mask_center_range", type=float, default=0.3,
                    help="Centers drawn uniformly from [-range, range]^2 in normalized coords, "
                         "capped per axis by --clamp_mask_to_frame.")
    ap.add_argument("--clamp_mask_to_frame", type=lambda s: str(s).strip().lower()
                    in ("1", "true", "t", "yes", "y"), default=True,
                    help="Cap the center range so a randomized mask never cuts the frame edge; the cap "
                         "is a function of --mask_radius and --full_eval_mask. false allows clipping.")
    # RALU schedule
    ap.add_argument("--num_inference_steps", type=int, default=50,
                    help="total steps; split across stages in RALU's published proportions")
    ap.add_argument("--ralu_level", type=int, default=4, choices=sorted(RALU_PRESETS),
                    help="which published RALU operating point to take N/e proportions from")
    ap.add_argument("--num_stages", type=int, default=3, choices=[2, 3],
                    help="3 = their full design (uniform full-res output); "
                         "2 = drop stage 3 (mixed-res output, see RALU_FOVEATED.md caveat)")
    ap.add_argument("--match_compute", action="store_true",
                    help="solve total steps so RALU's token-steps equal ours at "
                         "--ours_steps for the same mask, instead of matching step count")
    ap.add_argument("--ours_steps", type=int, default=50,
                    help="step count our method uses, for --match_compute")
    ap.add_argument("--N", type=int, nargs="+", default=None,
                    help="override per-stage step counts (bypasses proportional scaling)")
    ap.add_argument("--e", type=float, nargs="+", default=None,
                    help="override stage end timesteps; last must be 1.0")
    ap.add_argument("--z_max", type=float, default=100.0,
                    help="upper bound on NT-DM's Z. On FLUX.2 the JSD objective is monotone "
                         "in Z, so the free solve lands at Z~28 and beta~0.05, i.e. the stage "
                         "transitions discard the previous stage. ~6.5 keeps RALU near the "
                         "re-noising strength the authors published (beta~0.21/0.25).")
    ap.add_argument("--use_published", action="store_true",
                    help="run their published config verbatim: their N, e, shift and Z "
                         "for --ralu_level, with no NT-DM re-solve. Ignores --match_compute.")
    ap.add_argument("--shift_mode", type=str, default="ntdm", choices=["ntdm", "natural"],
                    help="natural = per-stage shift exp(mu(tokens_i, N_i)); see set_params")
    ap.add_argument("--id_frame", type=str, default="coarse", choices=["coarse", "full"],
                    help="2-stage only; ignored for the 3-stage schedule")
    # Full evaluation (same flags as inference.py)
    ap.add_argument("--full_eval", action="store_true", default=False)
    ap.add_argument("--prompt_dataset_path", type=str, default=None)
    ap.add_argument("--num_prompts", type=int, default=None)
    ap.add_argument("--subset_idx", type=int, default=0)
    ap.add_argument("--num_subsets", type=int, default=1)
    # Debug
    ap.add_argument("--full_res_check", action="store_true",
                    help="force an all-ones mask (port sanity check)")
    return ap.parse_args()


def load_pipeline(args):
    pipe = Flux2DiffSynthRALUFovPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/"),
    )
    if args.lora_checkpoint:
        pipe.load_lora(pipe.dit, args.lora_checkpoint)      # native DiffSynth, no key renaming
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    return pipe


def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-axis sampling range, capped so the mask never crosses the frame edge. Same helper
    # ours/naive/ToMe call, so the cap is identical for a given (shape, r, lr_factor).
    center_range = random_mask_center_range(
        args.mask_center_range, args.mask_radius, args.full_eval_mask,
        args.height, args.width, args.lr_downsample_factor,
        clamp_to_frame=args.clamp_mask_to_frame,
    ) if args.random_mask_center else None

    def center_for(prompt_idx):
        """Centered, or this prompt's randomized fixation point — same helper ours/naive use."""
        if not args.random_mask_center:
            return (0, 0)
        return foveation_center_for_index(
            prompt_idx, seed=args.mask_center_seed, max_offset=center_range
        )

    def mask_for(center):
        m = coarse_foveation_mask(
            args.height, args.width, center=center, r=args.mask_radius,
            shape=args.full_eval_mask, lr_factor=args.lr_downsample_factor,
        )
        return torch.ones_like(m) if args.full_res_check else m

    # The schedule below is solved for prompt 0's mask. Under --random_mask_center the mask
    # moves per prompt, m changes with it, and the compute match is re-solved in the loop.
    mask = mask_for(center_for(0))

    seq_len = (args.height // 16) * (args.width // 16)
    from pipeline_flux2_diffsynth_RALU_fov import stage_token_counts

    def solve_schedule(n_hr):
        """RALU's schedule for a mask with n_hr foveal blocks. Depends on nothing else."""
        if args.use_published:
            pre = RALU_PRESETS[args.ralu_level]
            return (list(pre["N"]), list(pre["e"]), {"published": True},
                    list(pre["shift"]), float(pre["Z"]))
        N, e, info = ralu_schedule(
            total_steps=args.num_inference_steps, level=args.ralu_level,
            num_stages=args.num_stages, N=args.N, e=args.e,
            match_compute=args.match_compute, height=args.height, width=args.width,
            f0=args.lr_downsample_factor, n_hr_blocks=n_hr, ours_steps=args.ours_steps,
        )
        return N, e, info, None, None

    pipe = load_pipeline(args)
    pipe.id_frame = args.id_frame

    # The schedule, NT-DM params and correlated noise depend on the mask only through its
    # foveal-block count, so they are cached on n_hr: with a moving mask, m takes few
    # distinct values over 10k prompts and the NT-DM solve is not repeated per prompt.
    sched_cache = {}
    applied_n_hr = [None]

    def apply_mask(mask, verbose=False):
        n_hr = int(mask.sum())
        pipe.set_foveation_mask(mask)
        if n_hr in sched_cache:
            entry = sched_cache[n_hr]
            if applied_n_hr[0] != n_hr:
                pipe.params, pipe.n_primes = entry["params"], entry["n_primes"]
                applied_n_hr[0] = n_hr
            return entry
        N, e, info, pub_shift, pub_Z = solve_schedule(n_hr)
        params = pipe.set_params(
            N=N, e=e, seq_len=seq_len, f0=args.lr_downsample_factor,
            shift=pub_shift, Z=pub_Z, z_max=args.z_max, shift_mode=args.shift_mode,
            stage_tokens=stage_token_counts(args.height, args.width,
                                            args.lr_downsample_factor, n_hr, len(N)),
        )
        n_primes = pipe.generate_ralu_noise(args.height, args.width, seed=args.seed)
        entry = {"N": N, "e": e, "info": info, "params": params,
                 "n_primes": n_primes, "n_hr": n_hr}
        sched_cache[n_hr] = entry
        applied_n_hr[0] = n_hr
        print(f"RALU schedule for m={n_hr}: level={args.ralu_level} "
              f"stages={args.num_stages} N={N} (total {sum(N)}) e={e}"
              + (f"  compute-matched: {info}" if info.get("match_compute") else ""))
        if verbose:
            print(json.dumps(params, indent=2, default=str))
        return entry

    first = apply_mask(mask, verbose=True)
    N, e, sched_info, params = first["N"], first["e"], first["info"], first["params"]

    if args.full_eval:
        assert args.prompt_dataset_path, "--prompt_dataset_path is required for --full_eval"
        prompts = load_prompt_dataset(args.prompt_dataset_path)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_dataset_path}")
    else:
        assert args.prompt, "--prompt is required without --full_eval"
        prompts = [args.prompt]

    num_subsets = max(1, args.num_subsets)
    prompt_list, image_names, times, centers, hr_blocks = [], [], [], [], []
    # Append-only manifest with resume, for preemptible Slurm arrays. See shard_manifest.py.
    columns = (["image", "prompt"] + (["center_x", "center_y"] if args.random_mask_center else [])
               + TIMING_COLUMNS)
    manifest = ShardManifest(args.output_dir, args.subset_idx, columns=columns)
    if len(manifest):
        print(f"Resuming: {len(manifest)} sample(s) already complete in this shard")
    if args.random_mask_center:
        print(f"Randomized mask centers: seed={args.mask_center_seed}, "
              f"requested range=+/-{args.mask_center_range}, effective range=+/-"
              f"({center_range[0]:.4f}, {center_range[1]:.4f}) [capped to keep the "
              f"{args.full_eval_mask} r={args.mask_radius} mask inside the frame]")

    for global_idx, prompt in enumerate(prompts):
        if args.num_prompts is not None and global_idx >= args.num_prompts:
            break
        if (global_idx % num_subsets) != args.subset_idx:
            continue
        name = f"img_{global_idx:010d}.png"
        if manifest.is_done(name):
            continue          # row and image both present: nothing to redo

        center = center_for(global_idx)
        if args.random_mask_center:
            # m moves with the fixation point, so the compute match is re-solved (cached on m).
            entry = apply_mask(mask_for(center))
            hr_blocks.append(entry["n_hr"])
        centers.append(center)

        image = pipe(
            prompt=prompt,
            cfg_scale=args.cfg_scale,
            embedded_guidance=args.embedded_guidance,
            height=args.height, width=args.width,
            seed=args.seed,
        )
        image.save(os.path.join(args.output_dir, name))
        row = {"image": name, "prompt": prompt}
        if args.random_mask_center:
            row["center_x"], row["center_y"] = center[0], center[1]
        row.update(pipe_timings(pipe))
        manifest.append(row)      # image is on disk, so the row may be flushed
        prompt_list.append(prompt)
        image_names.append(name)
        times.append(pipe.denoise_time)
        print(f"[{global_idx}] {pipe.denoise_time:.2f}s  {prompt[:70]}")

    manifest.mark_complete()
    manifest.close()
    print(f"Shard {args.subset_idx}/{num_subsets}: {len(image_names)} generated this run, "
          f"{len(manifest)} complete total -> {args.output_dir}")

    n_hr, m_down = int(mask.sum()), mask.numel()
    stats = {
        "framework": "diffsynth",
        "method": "RALU-foveated",
        "ralu_level": args.ralu_level,
        "num_stages": args.num_stages,
        "schedule_info": sched_info,
        "params": params,
        "mask_shape": args.full_eval_mask,
        "mask_radius": args.mask_radius,
        "lr_downsample_factor": args.lr_downsample_factor,
        "hr_blocks": n_hr,
        "coarse_blocks": m_down,
        "fovea_area_frac": n_hr / m_down,
        "tokens_per_stage": [m_down, 4 * n_hr + (m_down - n_hr)] + ([seq_len] if len(N) == 3 else []),
        "tokens_full_res": seq_len,
        "num_images_this_run": len(image_names),
        "num_images_complete": len(manifest),
        "mean_denoise_time_s": float(np.mean(times)) if times else None,
    }
    if args.random_mask_center:
        # With a moving mask the fields above describe prompt 0 only; these describe the run.
        stats["random_mask_center"] = {
            "mask_center_seed": args.mask_center_seed,
            "mask_center_range_requested": args.mask_center_range,
            "mask_center_range_effective": list(center_range),
            "clamp_mask_to_frame": args.clamp_mask_to_frame,
            "distinct_hr_block_counts": len(sched_cache),
            "hr_blocks_min": int(min(hr_blocks)) if hr_blocks else None,
            "hr_blocks_max": int(max(hr_blocks)) if hr_blocks else None,
            "hr_blocks_mean": float(np.mean(hr_blocks)) if hr_blocks else None,
            "schedules_by_hr_blocks": {str(k): {"N": v["N"], "total_steps": sum(v["N"]),
                                                "cost_ratio": v["info"].get("cost_ratio")}
                                       for k, v in sorted(sched_cache.items())},
        }
    with open(os.path.join(args.output_dir, f"stats_{args.subset_idx:05d}.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))

    if args.subset_idx == 0:
        # Under --random_mask_center this is prompt 0's mask; the rest are reproducible from
        # the center_x/center_y columns in metadata_*.csv.
        mask_name = "mask_0000000000.png" if args.random_mask_center else "mask.png"
        Image.fromarray(
            (F.interpolate(mask[None, None], size=(args.height, args.width), mode="nearest")[0, 0]
             .numpy() * 255).astype(np.uint8)
        ).save(os.path.join(args.output_dir, mask_name))


if __name__ == "__main__":
    main()
