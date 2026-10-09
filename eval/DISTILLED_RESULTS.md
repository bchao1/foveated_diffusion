# Foveated LoRA on the distilled FLUX.2-klein-4B — results and file paths

Every path in this doc is **absolute** and is **readable by all users** on
`apollo.stanford.edu` (checked 2026-10-09: no file or directory under these paths is
missing world read / execute). `/local` is a disk local to apollo, so these paths do
not exist on other machines. Copy the data off apollo if you need it somewhere else.

Images are **not** in git. `results/` and `paper_figures/` are gitignored, and the
10k-image eval trees live outside the repo on `/local`.

## What was run

`FLUX.2-klein-4B` is `FLUX.2-klein-base-4B` after BFL applied LADD: step-distilled to
4 steps **and** guidance-distilled (no CFG). The architecture is identical (same 169
tensors), so the foveated LoRA trained on base loads on it unchanged. No distillation
training was done on our side.

| | |
|---|---|
| launcher | `MODEL=distilled ./eval/eval.sh all` |
| checkpoint | `black-forest-labs/FLUX.2-klein-4B` |
| generation | 1024x1024, 4 steps, cfg 1.0, seed 42, `lr_factor` 2 |
| masks | circular r = 0.1 / 0.25 / 0.5, random center per prompt (seed 42, ±0.3, clamped to frame) |
| prompts | `/local/brian/Aesthetic-Train-V2/metadata_test.csv`, 10,000 prompts |
| fov LoRA (`ours`) | `/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov_more_steps/step-5000.safetensors` |
| no-fov LoRA (other arms) | `/local/brian/foveated_diffusion/models/eccv2026_final/finetuned_no_fov/step-15000.safetensors` |
| launch log | `/local/brian/foveated_diffusion/results/iclr_2027_launch.log` |

## Main eval (10k images per arm)

Root: **`/local/brian/foveated_diffusion/results/iclr_2027/`**

The directory name has no "distilled" suffix. The **base-model** run of the same
sweep (50 steps, cfg 4.0) is `/local/brian/foveated_diffusion/results/iclr_2027_base/`.

```
iclr_2027/
  iqa_comparison.md / .csv              all metrics, every arm x radius
  high_res/                             full-res reference, shared by all radii
  circular_r{0.1,0.25,0.5}_randcenter42/
    ours/                               <- foveated LoRA
    naive/  tome/  ralu/                baselines
  logs/<method>_<mask_tag>_shard<i>.log
```

Each method folder has `img_%010d.png` (10,000), `metadata_%05d.csv` (prompt, mask
center, `dit_time_s`, `vae_time_s` per image) and an `iqa_results/` folder.

### Metrics — distilled, `ours` vs `high_res`

From `iclr_2027/iqa_comparison.md`:

| arm | FID | pFID | MUSIQ | CLIP-IQA | ImageReward | HPSv2.1 |
|---|---|---|---|---|---|---|
| ours r0.1 | 13.82 | 22.27 | 58.57 | 0.571 | 1.074 | 0.287 |
| ours r0.25 | 13.21 | 18.42 | 60.46 | 0.579 | 1.055 | 0.286 |
| ours r0.5 | 12.52 | 17.92 | 67.00 | 0.611 | 1.027 | 0.286 |
| high_res | 12.34 | 16.86 | 69.74 | 0.631 | 1.063 | 0.288 |

The naive / tome / ralu rows are in the same file.

### DiT time per image (mean seconds, from `metadata_*.csv`)

| radius | high_res | ours | naive | tome | ralu |
|---|---|---|---|---|---|
| 0.1 | 1.613 | 0.648 | 0.650 | 0.623 | 0.662 |
| 0.25 | 1.613 | 0.726 | 0.728 | 0.711 | 0.684 |
| 0.5 | 1.613 | 1.025 | 1.025 | 1.017 | 0.759 |

### Caveat: RALU is not compute-matched here

At 4 steps, `--match_compute` collapses RALU to `N=[1, 1, 1]` (3 steps total), and it
still costs 1.23x `ours@4` (logged as `cost_ratio: 1.2256` in `logs/ralu_*.log`).
Treat the distilled RALU arm as "minimum RALU schedule", not as compute-matched.

## Qualitative runs (in the repo's `results/`, gitignored)

| path | what |
|---|---|
| `/home/brianchc/foveated_diffusion-private/results/iclr_2027_qual_distilled/` | 9 prompts, all 5 arms, circular r0.33 random center. Source for `make_supp_distilled.py`. |
| `/home/brianchc/foveated_diffusion-private/results/iclr_2027_distilled_all/` | 61 prompts, same settings. `tile_index.csv` maps each image to its prompt set and row. |
| `/home/brianchc/foveated_diffusion-private/results/distill_transfer_test/` | First single-prompt transfer test (lion, square r0.5): base vs distilled, with and without LoRA. `README.md` + `comparison_grid.png`. Script: `eval/distill_transfer_test.sh`. |

Layout inside the two qual dirs: `high_res/` and `circular_r0.33_randcenter42/{naive,ours,ralu,tome}/`.
The prompts for each are in `_prompts_distilled_qual.csv`.

## Figures

`/home/brianchc/foveated_diffusion-private/paper_figures/supp/`:

- `baseline_distilled.png`: 9 rows x 5 cols, from `eval/iclr_2027/make_supp_distilled.py`
- `baseline_{101,1,2,3}_{a,b}_distilled.png`: distilled versions of the supp baseline figures

## Reproduce

```bash
cd /home/brianchc/foveated_diffusion-private/eval
MODEL=distilled RANDOM_MASK_CENTER=true MASK_CONFIGS="circular:0.1 circular:0.25 circular:0.5" \
  OUT_ROOT=/local/brian/foveated_diffusion/results/iclr_2027 GPUS="1 2 3 4 5 6 9" ./eval.sh all
MODEL=distilled ./eval.sh plan     # resolved config + image counts, no GPU
```

The GPU box is shared. Check `nvidia-smi` for other users' jobs before picking `GPUS`.
