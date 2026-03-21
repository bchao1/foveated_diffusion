# Foveated Diffusion — Eval Directory

## `make_video.py`

Video composition utilities. Use the `imp_tome` conda environment to run.

| Function | Description |
|---|---|
| `stack_videos(video_paths, output_path, grid_x, grid_y, margin, bg_color)` | Arrange videos in a `grid_x × grid_y` grid with pixel margin between cells. Shorter clips hold their last frame. |
| `concat_videos(video_paths, output_path, sleep_duration, fps)` | Concatenate clips sequentially with a black pause of `sleep_duration` seconds between each. |

**CLI:**
```bash
# Grid layout (2 columns × 2 rows, 20px margin)
conda run -n imp_tome python make_video.py stack a.mp4 b.mp4 c.mp4 d.mp4 \
  --output stacked.mp4 --grid-x 2 --grid-y 2 --margin 20

# Sequential with 2-second black gap
conda run -n imp_tome python make_video.py concat a.mp4 b.mp4 c.mp4 \
  --output concat.mp4 --sleep 2.0
```

Invoke `/make_video` to have Claude pick the right function and run it.

---

## Agent Instructions

Whenever `inference.py` is edited (arguments added/removed, experiments added, mask types changed,
output layout modified, etc.), update this `CLAUDE.md` to reflect those changes. Specifically:

- If a new `--experiment` value is added or removed, update the Experiments table.
- If a new `--foveation_trajectory_type` is added or removed, update the Foveation Mask Trajectories table.
- If a new CLI argument is added, changed, or removed, update the relevant argument table.
- If the output directory layout changes for any experiment, update the Output Layout section.
- After updating `CLAUDE.md`, also save a memory entry summarizing what changed (type: `project`).

---

## `inference.py`

FLUX2 foveated image generation script built on DiffSynth. Generates images where a high-resolution
"foveal" region (defined by a binary mask) coexists with a spatially downsampled periphery, processed
by a learned Token AE (upsampler + downsampler). Supports several experiment modes, distributed
evaluation, and multiple foveation mask trajectories.

---

### Experiments (`--experiment`)

| Value | Description |
|---|---|
| `high_res` | Baseline: full-resolution generation, no foveation, no TokenAE. |
| `naive_mixed_res` | Mixed-resolution generation using naive (bilinear/nearest) up/downsamplers, no learned TokenAE. |
| `ours` | Our proposed method: mixed-resolution generation with the learned TokenAE (loaded from `--tokenAE_folder`). |
| `circular_traj` | Single prompt, circular orbit of foveation mask center over `--num_frames` steps. Saves per-frame images and masks. |
| `vary_radius` | Single prompt, sweeps foveation radius from 0.1 to 1.0 over `--num_frames` steps. |
| `runtime` | Runtime benchmark: sweeps number of HR tokens, runs each config 5× and records mean time. Saves `runtime_and_radius_list.npz`. |
| `foveation_trajectory_grid` | Grid figure: N prompts × M foveation masks. Each row is a prompt, each column a mask from the trajectory. |
| `user_study` | Generates triplets (high_res, naive, ours) per prompt with a random foveation mask. LoRA is applied only for "ours". |

---

### Foveation Mask Trajectories (`--foveation_trajectory_type`)

Used by `foveation_trajectory_grid`. One trajectory is generated once and reused for all prompts.

| Value | Description |
|---|---|
| `circular` | (default) Mask center orbits in a circle of radius `--orbit_radius`. Mask shape/size fixed. |
| `radius` | Mask stays at center; radius sweeps linearly from 0.3 to 0.7 across columns. |
| `grid` | Mask center steps over an M×N grid (`--grid_rows` × `--grid_cols`). Center range auto-computed so the circle never exits the frame. |
| `spiral` | Two-pass: outward spiral (r grows 0→0.5, orbit grows 0→max) then inward spiral (r grows 0.5→1.0, orbit shrinks). Two full rotations total. |
| `polygons` | Sequence cycling through triangle, square, hexagon, star with random center and scale per column. |
| `multi_circle` | Each mask is 2–3 non-overlapping circles distributed across image quadrants with random positions. |

---

### Mask Shapes (`--mask_shape`)

For `circular_traj`, `vary_radius`, and trajectories that use per-pixel masks:

| Value | Description |
|---|---|
| `circular` | Circular region; radius relative to half the image diagonal. |
| `square` | Square region centered at the fixation point; side length = `r * min(H, W)`. |
| `checkerboard` | Interleaved HR/LR token pattern (full-image, ignores center/r). |

Polygon trajectories (`polygons`) additionally support triangle, square, hexagon, star.

---

### Key Arguments

#### Model
| Arg | Default | Description |
|---|---|---|
| `--model_id` | `black-forest-labs/FLUX.2-klein-base-4B` | HuggingFace model ID. |
| `--lora_checkpoint` | None | Path to `.safetensors` LoRA checkpoint. |
| `--dit_checkpoint` | None | Path to a full DiT checkpoint (replaces base weights). |
| `--tokenAE_folder` | None | Folder with `token_upsampler.pt`, `token_downsampler.pt`, and `tokenAE_config.yaml`. |
| `--tokenAE_arch` | `adapter` | TokenAE architecture: `adapter` (CNN ResNet), `flux2` (Flux2-style), `dit` (DiT blocks). Overridden by `tokenAE_config.yaml` when found. |

#### Generation
| Arg | Default | Description |
|---|---|---|
| `--height` / `--width` | 1024 | Output image resolution. |
| `--num_inference_steps` | 50 | Denoising steps. |
| `--guidance_scale` | 4.0 | CFG scale. |
| `--seed` | 0 | Global random seed (controls PyTorch, numpy, random, cudnn). |
| `--decode_mode` | `direct` | VAE decode mode: `direct` (HR only) or `merge` (blend HR and LR decoded latents). |
| `--soft_foveation_blend` | false | Gaussian-blur the foveation mask boundary before blending in merge decode. |
| `--lr_downsample_factor` | 2 | Spatial downsampling factor for low-res periphery (2 = 2×, 4 = 4×). |
| `--prediction_type` | `clean` | Token AE prediction target: `clean` or `flow`. |

#### Full Evaluation
| Arg | Default | Description |
|---|---|---|
| `--full_eval` | off | Load prompts from CSV instead of using `--prompt`. |
| `--prompt_dataset_path` | — | CSV with `prompt` (and optionally `image`) columns. |
| `--num_prompts` | None | Cap on number of prompts to run. |
| `--full_eval_mask` | `square` | Mask shape for `high_res`/`naive_mixed_res`/`ours` full-eval runs. |
| `--mask_radius` | 0.30 | Foveation radius for full-eval and trajectory experiments. |
| `--subset_idx` | 0 | Shard index for distributed eval (0 ≤ idx < num_subsets). |
| `--num_subsets` | 1 | Total shards (one per GPU for distributed runs). |

#### Trajectory Grid (`foveation_trajectory_grid`)
| Arg | Default | Description |
|---|---|---|
| `--num_cols` | 4 | Number of foveation positions (columns) per prompt row. |
| `--foveation_trajectory_type` | `circular` | See table above. |
| `--orbit_radius` | 0.25 | Orbit radius for `circular` trajectory (normalized, 0–0.5). |
| `--grid_rows` / `--grid_cols` | 3 / 3 | Grid dimensions for `grid` trajectory. |
| `--prompt_ids` | None | Explicit list of CSV row indices to use (overrides `--num_prompts`). |
| `--foveation_outline` | false | Draw white foveation boundary outline on saved images. |
| `--outline_width_frac` | 0.005 | Outline width as fraction of image width. |
| `--outline_color` | `255,0,0` | Outline color as `R,G,B`. |

---

### Output Layout

**`high_res` / `naive_mixed_res` / `ours` (full eval):**
```
output_dir/
  img_0000000000.png   # indexed by global prompt idx
  img_0000000001.png
  ...
  metadata_00000.csv   # image filename + prompt per row (one file per subset)
```

**`foveation_trajectory_grid`:**
```
output_dir/
  tokenization_masks/
    tokenization_mask_0000.png  # HR/LR token grid visualization per column
  000/
    prompt.txt
    img_0000.png   # column 0
    img_0001.png   # column 1
    ...
  001/
    ...
```

**`user_study`:**
```
output_dir/
  00000/
    prompt.txt
    img_high_res.png
    img_naive.png       # with foveation outline if --foveation_outline
    img_ours.png        # with foveation outline if --foveation_outline
    mask.png            # full-res foveation mask (grayscale)
    fixation_point.png  # red dot at fixation center
  00001/
    ...
```

**`circular_traj` / `vary_radius`:**
```
output_dir/
  img_000.png
  foveation_mask_000.png
  ...
```

**`runtime`:**
```
output_dir/
  img_00.png
  runtime_and_radius_list.npz   # arrays: runtime_list, r_list, token_ratio_list
```

---

### Distributed Evaluation

Split a large prompt set across multiple GPUs by running one process per GPU with different `--subset_idx` values:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python inference.py \
    --experiment ours \
    --full_eval \
    --num_subsets 8 \
    --subset_idx $i \
    ... &
done
wait
```

Each process writes its own `metadata_<subset_idx>.csv`. Image filenames use the global prompt index so they remain consistent across shards.

---

### Example Invocations

**Baseline full-resolution (single GPU):**
```bash
python inference.py \
  --experiment high_res \
  --decode_mode direct \
  --full_eval \
  --prompt_dataset_path prompts/claude_prompts.csv \
  --num_prompts 100 \
  --seed 42 \
  --output_dir ../results/baseline_eval
```

**Our method with LoRA:**
```bash
python inference.py \
  --experiment ours \
  --decode_mode merge \
  --soft_foveation_blend true \
  --full_eval \
  --full_eval_mask square \
  --mask_radius 0.5 \
  --lora_checkpoint /path/to/step-5000.safetensors \
  --tokenAE_folder /path/to/tokenAE/ \
  --prompt_dataset_path prompts/claude_prompts.csv \
  --num_prompts 100 \
  --seed 42 \
  --output_dir ../results/ours_eval
```

**Spiral trajectory grid:**
```bash
python inference.py \
  --experiment foveation_trajectory_grid \
  --foveation_trajectory_type spiral \
  --decode_mode merge \
  --soft_foveation_blend true \
  --num_cols 12 \
  --foveation_outline true \
  --lora_checkpoint /path/to/step-5000.safetensors \
  --prompt_dataset_path prompts/website_prompts_landscapes_v2.csv \
  --prompt_ids 3 6 11 12 13 \
  --seed 42 \
  --output_dir ../results/spiral
```

**User study:**
```bash
python inference.py \
  --experiment user_study \
  --decode_mode merge \
  --soft_foveation_blend true \
  --foveation_outline true \
  --lora_checkpoint /path/to/step-5000.safetensors \
  --tokenAE_folder /path/to/tokenAE/ \
  --prompt_dataset_path prompts/user_study.csv \
  --num_prompts 50 \
  --seed 42 \
  --output_dir ../results/user_study
```
