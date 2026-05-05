#!/usr/bin/env python3
"""CLIPScore evaluation for multi_object experiment (high_res / random / naive).

Calls _clip_sim_batch_loop per-image to get individual scores, averages seeds
per prompt, then reports mean±std over 100 prompts.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

PROMPTS_CSV = Path("/home/brianchc/foveated_diffusion/eval/prompts/saliency_multi_object.csv")
BASE = Path("/home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/multi_object")


def load_prompts(csv_path):
    with open(csv_path, newline="") as f:
        return [row["prompt"] for row in csv.DictReader(f)]


def load_flat(images_dir):
    imgs = sorted(p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix == ".png" and p.name.startswith("img_"))
    return [[img] for img in imgs]


def load_folders(images_dir):
    subdirs = sorted(d for d in images_dir.iterdir() if d.is_dir() and d.name.isdigit())
    grouped = []
    for subdir in subdirs:
        seeds = sorted(p for p in subdir.iterdir()
                       if p.name.startswith("img_") and p.suffix == ".png")
        if seeds:
            grouped.append(seeds)
    return grouped


def score_per_image(grouped, prompts, model, tokenizer, image_size, image_mean, image_std, dtype, device):
    """Score each image individually and return per-prompt averaged scores."""
    from metric_computation.CLIP import _clip_sim_batch_loop

    per_prompt = []
    for seeds, prompt in zip(grouped, prompts):
        seed_scores = []
        for img_path in seeds:
            s = _clip_sim_batch_loop(
                [str(img_path)], [prompt], model, tokenizer,
                image_size, image_mean, image_std, dtype,
                batch_size=1, device=device,
            )
            seed_scores.append(s)
        per_prompt.append(sum(seed_scores) / len(seed_scores))
    return per_prompt


def summary_stats(per_prompt):
    mean = sum(per_prompt) / len(per_prompt)
    std = math.sqrt(sum((s - mean) ** 2 for s in per_prompt) / len(per_prompt))
    return mean, std


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from metric_computation.CLIP import _load_openclip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading CLIP model...")
    model, tokenizer = _load_openclip(device=device)
    model.eval()

    image_size = model.visual.image_size
    if isinstance(image_size, tuple):
        image_size = image_size[0]
    image_size = int(image_size)
    image_mean = torch.tensor(model.visual.image_mean, device=device).view(3, 1, 1)
    image_std  = torch.tensor(model.visual.image_std,  device=device).view(3, 1, 1)
    dtype = torch.float16

    prompts = load_prompts(PROMPTS_CSV)

    conditions = {
        "high_res": ("flat",   BASE / "high_res"),
        "random":   ("folder", BASE / "random"),
        "naive":    ("folder", BASE / "naive"),
    }

    all_results = {}
    for name, (layout, images_dir) in conditions.items():
        print(f"\n=== {name} ===")
        grouped = load_flat(images_dir) if layout == "flat" else load_folders(images_dir)
        assert len(grouped) == len(prompts), f"{name}: {len(grouped)} groups vs {len(prompts)} prompts"
        print(f"  {len(grouped)} prompts, {len(grouped[0])} seed(s) each")

        per_prompt = score_per_image(grouped, prompts, model, tokenizer,
                                     image_size, image_mean, image_std, dtype, device)
        mean, std = summary_stats(per_prompt)

        all_results[name] = {"mean": mean, "std": std, "per_prompt": per_prompt}

        out_dir = images_dir / "iqa_results"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "clipscore.json", "w") as f:
            json.dump(all_results[name], f, indent=2)
        print(f"  mean = {mean:.4f}  std = {std:.4f}")
        print(f"  saved → {out_dir / 'clipscore.json'}")

    print("\n=== Summary ===")
    print(f"{'Condition':<12}  {'Mean':>8}  {'Std':>8}")
    print("-" * 32)
    for name, r in all_results.items():
        print(f"{name:<12}  {r['mean']:>8.4f}  {r['std']:>8.4f}")


if __name__ == "__main__":
    main()
