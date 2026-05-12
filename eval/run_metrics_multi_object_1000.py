#!/usr/bin/env python3
"""ImageReward + HPSv2 evaluation for multi_object_1000_prompts.

Layouts:
  high_res/seed_{42,123,456}/img_*.png    (1000 imgs × 3 seed dirs, flat)
  random/{000..999}/img_*.png             (1000 prompts × 3 img seeds)
  naive/{000..999}/img_*.png              (1000 prompts × 3 img seeds)

Reports per-prompt mean (averaged over 3 seeds) ± std across 1000 prompts,
for both ImageReward and HPSv2.1.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

# GPU 1 — must be set BEFORE importing torch / hpsv2.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

BASE = Path("/home/brianchc/foveated_diffusion/results/eccv2026_rebuttal/multi_object_1000_prompts")
PROMPTS_CSV = BASE / "high_res" / "seed_42" / "metadata_00000.csv"


def load_prompts(csv_path):
    with open(csv_path, newline="") as f:
        return [row["prompt"] for row in csv.DictReader(f)]


def load_high_res(images_dir, prompts):
    """high_res/seed_*/img_*.png — return list[list[Path]] grouped by prompt with 3 seeds each."""
    seed_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    per_seed_imgs = []
    for sd in seed_dirs:
        imgs = sorted(p for p in sd.iterdir()
                      if p.is_file() and p.name.startswith("img_") and p.suffix == ".png")
        per_seed_imgs.append(imgs)
    n = len(prompts)
    for sd_imgs, sd in zip(per_seed_imgs, seed_dirs):
        assert len(sd_imgs) == n, f"{sd}: {len(sd_imgs)} imgs vs {n} prompts"
    return [[per_seed_imgs[s][i] for s in range(len(seed_dirs))] for i in range(n)]


def load_folders(images_dir, prompts):
    """random/000/img_*.png — return list[list[Path]] grouped by prompt with 3 img_ seeds."""
    subdirs = sorted(d for d in images_dir.iterdir() if d.is_dir() and d.name.isdigit())
    assert len(subdirs) == len(prompts), f"{images_dir}: {len(subdirs)} dirs vs {len(prompts)} prompts"
    grouped = []
    for sd in subdirs:
        seeds = sorted(p for p in sd.iterdir()
                       if p.name.startswith("img_") and p.suffix == ".png")
        grouped.append(seeds)
    return grouped


def summary_stats(per_prompt):
    mean = sum(per_prompt) / len(per_prompt)
    std = math.sqrt(sum((s - mean) ** 2 for s in per_prompt) / len(per_prompt))
    return mean, std


# ---------- ImageReward ----------
def run_image_reward(grouped_per_cond, prompts):
    import ImageReward as RM
    print("\nLoading ImageReward-v1.0 …")
    model = RM.load("ImageReward-v1.0")

    results = {}
    for name, grouped in grouped_per_cond.items():
        print(f"\n[ImageReward] {name}")
        per_prompt = []
        for seeds, prompt in tqdm(zip(grouped, prompts), total=len(prompts), desc=name):
            scores = [float(model.score(prompt, str(p))) for p in seeds]
            per_prompt.append(sum(scores) / len(scores))
        mean, std = summary_stats(per_prompt)
        results[name] = {"mean": mean, "std": std, "per_prompt": per_prompt}
        print(f"  mean = {mean:.4f}  std = {std:.4f}")

    # release model
    del model
    torch.cuda.empty_cache()
    return results


# ---------- HPSv2 ----------
def run_hpsv2(grouped_per_cond, prompts):
    from metric_computation.HPSV2 import compute_hpsv2

    results = {}
    for name, grouped in grouped_per_cond.items():
        print(f"\n[HPSv2] {name}")
        flat_paths, flat_prompts = [], []
        for seeds, prompt in zip(grouped, prompts):
            for s in seeds:
                flat_paths.append(str(s))
                flat_prompts.append(prompt)

        out = compute_hpsv2(flat_paths, flat_prompts, batch_size=32, desc=f"HPSv2 {name}")
        scores = out["scores"]

        # Average seed scores per prompt
        per_prompt, idx = [], 0
        for seeds in grouped:
            n = len(seeds)
            per_prompt.append(sum(scores[idx:idx + n]) / n)
            idx += n
        mean, std = summary_stats(per_prompt)
        results[name] = {"mean": mean, "std": std, "per_prompt": per_prompt}
        print(f"  mean = {mean:.4f}  std = {std:.4f}")
    return results


def main():
    prompts = load_prompts(PROMPTS_CSV)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_CSV}")
    print(f"Device: cuda (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    grouped_per_cond = {
        "high_res": load_high_res(BASE / "high_res", prompts),
        "random":   load_folders(BASE / "random",   prompts),
        "naive":    load_folders(BASE / "naive",    prompts),
    }
    for name, g in grouped_per_cond.items():
        print(f"  {name}: {len(g)} prompts × {len(g[0])} seeds = {sum(len(s) for s in g)} images")

    # 1) ImageReward
    ir_results = run_image_reward(grouped_per_cond, prompts)

    # 2) HPSv2
    hps_results = run_hpsv2(grouped_per_cond, prompts)

    # Save per condition
    for name in grouped_per_cond:
        cond_dir = BASE / name
        out_dir = cond_dir / "iqa_results"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "image_reward.json", "w") as f:
            json.dump(ir_results[name], f, indent=2)
        with open(out_dir / "hpsv2.json", "w") as f:
            json.dump(hps_results[name], f, indent=2)
        print(f"saved → {out_dir}")

    print("\n=== Summary ===")
    print(f"{'Condition':<12}  {'IR mean':>10}  {'IR std':>8}  {'HPSv2 mean':>12}  {'HPSv2 std':>10}")
    print("-" * 60)
    for name in grouped_per_cond:
        ir = ir_results[name]; hp = hps_results[name]
        print(f"{name:<12}  {ir['mean']:>10.4f}  {ir['std']:>8.4f}  {hp['mean']:>12.4f}  {hp['std']:>10.4f}")


if __name__ == "__main__":
    main()
