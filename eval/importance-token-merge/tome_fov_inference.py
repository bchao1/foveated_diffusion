"""
ToMe-2x2 baseline driver for the foveated diffusion comparison.

Runs FLUX.2 with importance-based token merging (Wu et al., ICCV 2025) adapted so the
merge grouping is supplied by *our* foveation mask instead of their error heat map, at a
token budget matched to our foveated pipeline.

Uses the stock `Flux2ImagePipeline`; the foveated pipeline, the foveated DiT, and the
TokenAE are never loaded. Output layout matches `eval/inference.py` exactly
(`img_%010d.png` + `metadata_%05d.csv`) so the existing metric scripts run unchanged.

Example:
    PYTHONPATH=<repo>/DiffSynth-Studio python tome_fov_inference.py \\
        --full_eval --prompt_dataset_path ../prompts/claude_prompts.csv \\
        --num_prompts 100 --lora_checkpoint <no_fov>/step-15000.safetensors \\
        --full_eval_mask circular --mask_radius 0.3 --merge_mlp false \\
        --output_dir ../../results/tome_eval
"""

import argparse
import glob
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Read-only reuse so the mask is bit-identical to the one our runs use.
from inference import (  # noqa: E402
    create_foveation_mask,
    foveation_center_for_index,
    load_prompt_dataset,
    random_mask_center_range,
)
from shard_manifest import ShardManifest  # noqa: E402

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig  # noqa: E402

from flux2_tome import apply_patch  # noqa: E402
from flux2_tome.merge_2x2 import BlockMerger  # noqa: E402


def _str_to_bool(s):
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y")


def _model_configs(args):
    """Build model configs from a local directory (--model_path) or a hub id (--model_id)."""
    if args.model_path is None:
        return (
            [
                ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/*.safetensors"),
                ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/*.safetensors"),
                ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/"),
        )

    root = os.path.abspath(os.path.expanduser(args.model_path))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--model_path is not a directory: {root}")

    def shards(subdir):
        found = sorted(glob.glob(os.path.join(root, subdir, "*.safetensors")))
        if not found:
            raise FileNotFoundError(f"no .safetensors under {os.path.join(root, subdir)}")
        return found

    print(f"Loading FLUX.2 from local path: {root}")
    return (
        [
            ModelConfig(path=shards("transformer")),
            ModelConfig(path=shards("text_encoder")),
            ModelConfig(path=shards("vae")),
        ],
        ModelConfig(path=os.path.join(root, "tokenizer")),
    )


def load_pipeline(args):
    print("Loading stock FLUX.2 pipeline (no foveated pipeline, no TokenAE)...")
    model_configs, tokenizer_config = _model_configs(args)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    if args.lora_checkpoint is not None:
        pipe.load_lora(pipe.dit, args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    return pipe


def generate(pipe, args, output_dir, prompt, prompt_idx):
    t0 = time.time()
    image = pipe(
        prompt=prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.guidance_scale,
    )
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))
    print(f"  prompt {prompt_idx:010d} done in {time.time() - t0:.2f}s")


def main(args):
    torch.manual_seed(args.seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline(args)
    latent_height, latent_width = args.height // 16, args.width // 16

    def mask_for(center):
        """Same mask our runs use. create_foveation_mask returns the (H/16, W/16) token grid."""
        return create_foveation_mask(
            args.height, args.width,
            center=center, r=args.mask_radius, shape=args.full_eval_mask,
            device=pipe.device, lr_factor=args.lr_downsample_factor,
        )

    # Per-axis sampling range, capped so the mask never crosses the frame edge. Same helper
    # ours/naive/RALU call, so the cap is identical for a given (shape, r, lr_factor).
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

    # The mask passed to apply_patch is the centered one; with --random_mask_center the
    # per-prompt mask replaces state["merger"] below before each generate() call.
    foveation_mask = mask_for(center_for(0) if args.random_mask_center else (0, 0))

    # Patch AFTER LoRA loading; this rebinds instance classes only.
    state = apply_patch(
        pipe.dit, foveation_mask, latent_height, latent_width,
        lr_factor=args.lr_downsample_factor,
        merge_mlp=args.merge_mlp,
        enabled=args.enable_tome,
    )
    merger = state["merger"]

    # Shard 0 only: every shard of a sharded eval would otherwise truncate and rewrite the
    # same path concurrently. All shards derive identical content, so shard 0's copy is it.
    if args.subset_idx == 0:
        with open(os.path.join(output_dir, "tome_config.txt"), "w") as f:
            f.write(
                f"variant=tome_2x2_mask\n"
                f"mask={args.full_eval_mask} r={args.mask_radius} lr_factor={args.lr_downsample_factor}\n"
                f"mask_center={'random per prompt (seed %d, effective range %.4f/%.4f)' % (args.mask_center_seed, center_range[0], center_range[1]) if args.random_mask_center else 'centered'}\n"
                f"tokens={merger.num_tokens} -> {merger.num_merged}"
                f"{' (varies per prompt; this is prompt 0)' if args.random_mask_center else ''}\n"
                f"merge_mlp={args.merge_mlp} (double blocks only; single blocks always merge both)\n"
                f"enable_tome={args.enable_tome}\n"
                f"model={args.model_path or args.model_id}\n"
                f"lora={args.lora_checkpoint}\n"
                f"steps={args.num_inference_steps} cfg={args.guidance_scale} seed={args.seed}\n"
            )
        # Same artifact RALU_fov_inference_diffsynth.py writes, for auditing that both
        # baselines really foveated the same region. Under --random_mask_center the mask
        # moves per prompt, so this is prompt 0's; the rest are reproducible from the
        # center_x/center_y columns in metadata_*.csv.
        mask_name = "mask_0000000000.png" if args.random_mask_center else "mask.png"
        Image.fromarray(
            (torch.nn.functional.interpolate(
                foveation_mask[None, None].float().cpu(),
                size=(args.height, args.width), mode="nearest")[0, 0].numpy() * 255
             ).astype("uint8")
        ).save(os.path.join(output_dir, mask_name))

    if not args.full_eval:
        generate(pipe, args, output_dir, args.prompt, 0)
        return

    prompts = load_prompt_dataset(args.prompt_dataset_path)
    num_subsets = max(1, args.num_subsets)
    # Append-only manifest with resume, for preemptible Slurm arrays. See shard_manifest.py.
    columns = ["image", "prompt"] + (["center_x", "center_y"] if args.random_mask_center else [])
    manifest = ShardManifest(output_dir, args.subset_idx, columns=columns)
    if len(manifest):
        print(f"Resuming: {len(manifest)} sample(s) already complete in this shard")
    generated = 0
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
        image_name = f"img_{global_idx:010d}.png"
        if manifest.is_done(image_name):
            continue          # row and image both present: nothing to redo
        center = center_for(global_idx)
        if args.random_mask_center:
            # All blocks share this one state dict, so replacing the merger rebinds every
            # block at once. merged_rope is rebuilt from it on every forward (patch_flux2:230),
            # so nothing stale survives the swap.
            state["merger"] = BlockMerger(
                mask_for(center), latent_height, latent_width,
                lr_factor=args.lr_downsample_factor,
            )
        generate(pipe, args, output_dir, prompt, global_idx)
        row = {"image": image_name, "prompt": prompt}
        if args.random_mask_center:
            row["center_x"], row["center_y"] = center[0], center[1]
        manifest.append(row)      # image is on disk, so the row may be flushed
        generated += 1

    manifest.mark_complete()
    manifest.close()
    print(f"Shard {args.subset_idx}/{num_subsets}: {generated} generated this run, "
          f"{len(manifest)} complete total -> {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ToMe-2x2 foveated baseline for FLUX.2")
    # Model
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B",
                   help="Hub id, resolved under $DIFFSYNTH_MODEL_BASE_PATH. Ignored if --model_path is set.")
    p.add_argument("--model_path", type=str, default=None,
                   help="Local model directory containing transformer/ text_encoder/ vae/ tokenizer/.")
    p.add_argument("--lora_checkpoint", type=str, default=None)
    # Generation (defaults mirror eval/inference.py)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", type=str, default="A lion lounges calmly in golden afternoon light.")
    p.add_argument("--output_dir", type=str, default="../../results/tome_eval")
    # Foveation mask (must match our runs)
    p.add_argument("--full_eval_mask", type=str, default="circular",
                   choices=["circular", "square", "checkerboard"])
    p.add_argument("--mask_radius", type=float, default=0.30)
    p.add_argument("--lr_downsample_factor", type=int, default=2)
    p.add_argument("--random_mask_center", type=_str_to_bool, default=False,
                   help="Randomize the mask center per prompt instead of centering it. Uses "
                        "inference.py:foveation_center_for_index, so ours/naive/RALU produce "
                        "the identical mask for the same prompt index and settings.")
    p.add_argument("--mask_center_seed", type=int, default=42,
                   help="Seed for --random_mask_center; must match the other baselines.")
    p.add_argument("--mask_center_range", type=float, default=0.3,
                   help="Centers drawn uniformly from [-range, range]^2 in normalized coords, "
                        "capped per axis by --clamp_mask_to_frame.")
    p.add_argument("--clamp_mask_to_frame", type=_str_to_bool, default=True,
                   help="Cap the center range so a randomized mask never cuts the frame edge; the cap "
                        "is a function of --mask_radius and --full_eval_mask. false allows clipping.")
    # ToMe
    p.add_argument("--merge_mlp", type=_str_to_bool, default=False,
                   help="Merge the image MLP in double blocks too. Single (parallel) blocks "
                        "always merge attention and MLP together regardless of this flag.")
    p.add_argument("--enable_tome", type=_str_to_bool, default=True,
                   help="Set false for the ratio=0 identity control (should match high_res).")
    # Full eval
    p.add_argument("--full_eval", action="store_true")
    p.add_argument("--prompt_dataset_path", type=str, default=None)
    p.add_argument("--num_prompts", type=int, default=None)
    p.add_argument("--subset_idx", type=int, default=0)
    p.add_argument("--num_subsets", type=int, default=1)

    main(p.parse_args())
