#!/usr/bin/env python3
"""HPSv2 evaluation for multi_object experiment (high_res / random / naive).

For random/naive: averages 3 seed scores per prompt, then reports mean±std over 100 prompts.
For high_res: 1 seed per prompt, reports mean±std over 100 prompts.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROMPTS_CSV = Path("/home/brianchc/foveated_diffusion/eval/prompts/saliency_multi_object.csv")
BASE = Path("/home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/multi_object")


def load_prompts(csv_path):
    with open(csv_path, newline="") as f:
        return [row["prompt"] for row in csv.DictReader(f)]


def load_flat(images_dir):
    """Returns (image_paths, prompts, grouped) for flat layout via metadata CSV."""
    csv_files = sorted(images_dir.glob("metadata_*.csv"))
    rows = []
    for f in csv_files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                img = images_dir / row["image"]
                if img.exists():
                    rows.append((str(img), row["prompt"]))
    image_paths = [r[0] for r in rows]
    prompts_flat = [r[1] for r in rows]
    grouped = [[Path(p)] for p in image_paths]
    return image_paths, prompts_flat, grouped


def load_folders(images_dir, prompts):
    """Returns (image_paths, prompts_flat, grouped) for folder layout (img_* only)."""
    subdirs = sorted(d for d in images_dir.iterdir() if d.is_dir() and d.name.isdigit())
    assert len(subdirs) == len(prompts)
    image_paths, prompts_flat, grouped = [], [], []
    for subdir, prompt in zip(subdirs, prompts):
        seeds = sorted(p for p in subdir.iterdir()
                       if p.name.startswith("img_") and p.suffix == ".png")
        grouped.append(seeds)
        for s in seeds:
            image_paths.append(str(s))
            prompts_flat.append(prompt)
    return image_paths, prompts_flat, grouped


def per_prompt_stats(grouped, all_scores):
    """Average seed scores per prompt, return list of per-prompt means."""
    idx = 0
    per_prompt = []
    for seeds in grouped:
        n = len(seeds)
        per_prompt.append(sum(all_scores[idx:idx + n]) / n)
        idx += n
    return per_prompt


def summary_stats(per_prompt):
    mean = sum(per_prompt) / len(per_prompt)
    variance = sum((s - mean) ** 2 for s in per_prompt) / len(per_prompt)
    return mean, math.sqrt(variance)


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from metric_computation.HPSV2 import compute_hpsv2

    prompts = load_prompts(PROMPTS_CSV)

    conditions = {
        "high_res": ("flat",   BASE / "high_res"),
        "random":   ("folder", BASE / "random"),
        "naive":    ("folder", BASE / "naive"),
    }

    all_results = {}
    for name, (layout, images_dir) in conditions.items():
        print(f"\n=== {name} ===")
        if layout == "flat":
            image_paths, prompts_flat, grouped = load_flat(images_dir)
        else:
            image_paths, prompts_flat, grouped = load_folders(images_dir, prompts)

        print(f"  {len(grouped)} prompts, {len(grouped[0])} seed(s) each, {len(image_paths)} images total")

        result = compute_hpsv2(image_paths, prompts_flat, batch_size=32, desc=f"HPSv2 {name}")
        per_prompt = per_prompt_stats(grouped, result["scores"])
        mean, std = summary_stats(per_prompt)

        all_results[name] = {"mean": mean, "std": std, "per_prompt": per_prompt}

        out_dir = images_dir / "iqa_results"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "hpsv2.json", "w") as f:
            json.dump(all_results[name], f, indent=2)
        print(f"  mean = {mean:.4f}  std = {std:.4f}")
        print(f"  saved → {out_dir / 'hpsv2.json'}")

    print("\n=== Summary ===")
    print(f"{'Condition':<12}  {'Mean':>8}  {'Std':>8}")
    print("-" * 32)
    for name, r in all_results.items():
        print(f"{name:<12}  {r['mean']:>8.4f}  {r['std']:>8.4f}")


if __name__ == "__main__":
    main()
