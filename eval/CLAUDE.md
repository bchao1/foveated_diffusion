# Foveated Diffusion — Eval Directory

## `eval.sh` — the entry point for large-scale evaluation

One script, five methods, one configuration block. Every method function receives the same
mask flags, prompt set, seed/steps/cfg/resolution and sharding; only the method and its
model differ. Runnable from any directory (it resolves the repo from its own path).

| Method | Script | Model |
|---|---|---|
| `high_res` | `inference.py --experiment high_res` | no-fov LoRA |
| `naive` | `inference.py --experiment naive_mixed_res` | no-fov LoRA |
| `ours` | `inference.py --experiment ours` | fov LoRA |
| `tome` | `importance-token-merge/tome_fov_inference.py` | no-fov LoRA |
| `ralu` | `RALU/RALU_fov_inference_diffsynth.py` | no-fov LoRA |

```bash
./eval.sh plan            # resolved config, output dirs, image counts — no GPU
./eval.sh masks           # prove every arm derives an identical mask — no GPU
./eval.sh ours            # one method, every MASK_CONFIGS entry
./eval.sh all             # all five, sequentially
./eval.sh check           # PNG counts vs the prompt count

GPUS="0 1 2 3 4 5 6 7" RANDOM_MASK_CENTER=true ./eval.sh all
NUM_PROMPTS=2 MASK_CONFIGS="circular:0.25" GPUS="5" ./eval.sh tome     # smoke test
```

Knobs (all env-overridable, see the `CONFIGURE` block): `MASK_CONFIGS` (`shape:radius`
pairs), `METHODS`, `PROMPTS`, `NUM_PROMPTS`, `GPUS`, `HEIGHT`/`WIDTH`/`STEPS`/`SEED`/`CFG`/
`LR_FACTOR`, `RANDOM_MASK_CENTER`/`MASK_CENTER_SEED`/`MASK_CENTER_RANGE`/
`CLAMP_MASK_TO_FRAME`, `NOFOV_LORA`/`FOV_LORA`/`TOKENAE_FOLDER`, `MERGE_MLP`,
`RALU_STAGES`/`RALU_LEVEL`/`RALU_Z_MAX`/`OURS_STEPS`, `OUT_ROOT`, `SKIP_COMPLETE`.

Output layout — arms grouped per mask config, so a comparison is one directory listing:

```
$OUT_ROOT/<shape>_r<radius>[_randcenter<seed>]/<method>/img_%010d.png + metadata_%05d.csv
$OUT_ROOT/high_res/                     # mask-independent, generated once, shared
$OUT_ROOT/logs/<method>_<mask_tag>_shard<i>.log
```

`make_eval_summary.py` renders a methods-as-columns / prompts-as-rows figure from that
tree, with the foveal boundary drawn on each tile:

```bash
python make_eval_summary.py --out_root ../results/eval_smoke_5arm \
    --mask_tag circular_r0.25_randcenter42 --num_prompts 2
```

The per-baseline launchers (`naive.sh`, `ours.sh`, `RALU/ralu_eval_10k.sh`,
`importance-token-merge/tome_eval_10k.sh`) still work and remain the record of the existing
`/local/brian/foveated_diffusion/results/*_eval` dirs; `eval.sh` supersedes them for new runs.

## `make_video.py`

Video composition utilities. Use the `imp_tome` conda environment to run.

| Function | Description |
|---|---|
| `stack_videos(video_paths, output_path, grid_x, grid_y, margin, bg_color)` | Arrange videos in a `grid_x × grid_y` grid with pixel margin between cells. Shorter clips hold their last frame. |
| `concat_videos(video_paths, output_path, sleep_duration, fps, freeze)` | Concatenate clips sequentially with a black pause or frozen last frame between each. |
| `add_text_overlay(video_path, output_path, text, font_path, font_size, color, position, padding, outline_width, header, header_height)` | Overlay text on every frame, or prepend a black header bar. When `header=True`, text is pixel-centered in a black bar prepended above the video. Use `header_height` to enforce matching bar heights across multiple videos. |
| `images_to_video(folder, output_path, fps, image_order, pattern, trajectory, num_frames, grid_rows, grid_cols, lissajous_a, lissajous_b, lissajous_delta)` | Compile images from a folder into a video. Supports smooth trajectory curves through image grids via `trajectory=`. |
| `wipe_video(img_folder, mask_folder, output_path, fps, num_frames, grid_rows, grid_cols, trajectory, wipe_frames, hold_frames, freeze_frames, slider_width, img_label, mask_label, ...)` | Trajectory animation that wipes to a tokenization mask at the midpoint and back. Text labels slide with the wipe slider. |

**Trajectory types** (`--trajectory`): `lissajous`, `spiral`, `rose`, `epitrochoid`, `hypotrochoid`, `butterfly`

**CLI subcommands:** `stack`, `concat`, `text`, `images`, `wipe`

**CLI:**
```bash
# Grid layout (2 columns × 2 rows, 20px margin)
conda run -n imp_tome python make_video.py stack a.mp4 b.mp4 c.mp4 d.mp4 \
  --output stacked.mp4 --grid-x 2 --grid-y 2 --margin 20

# Sequential with 2-second frozen-frame gap
conda run -n imp_tome python make_video.py concat a.mp4 b.mp4 c.mp4 \
  --output concat.mp4 --sleep 2.0 --freeze

# Add black header bar above video
conda run -n imp_tome python make_video.py text input.mp4 \
  --output out.mp4 --text "My Label" --font /home/brianchc/fonts/cmu-serif/cmunbx.ttf \
  --font-size 82 --color 255,255,255 --padding 20 --header --header-height 114

# Rose trajectory through 10×10 image grid (240 frames, 30fps)
conda run -n imp_tome python make_video.py images /path/to/folder \
  --output out.mp4 --fps 30 --trajectory rose --num-frames 240 --grid-rows 10 --grid-cols 10
```

Invoke `/make_video` to have Claude pick the right function and run it.

---

## Animation Workflows

Standard parameters shared across all workflows unless noted:
- Font: CMU Serif Bold (`/home/brianchc/fonts/cmu-serif/cmunbx.ttf`)
- Header font size: **82px** (8% of 1024px image height), header bar height: **114px** (forced via `--header-height`)
- Bottom-center overlay font size: **77px** (5% × 1.5 of 1024px), white, black outline (`--outline-width 3`)
- Stack margin: **20px**, `--target-size none`
- Trajectory: **rose**, 240 frames at 30fps (unless a different duration is specified)
- Default output: `/home/brianchc/foveated_diffusion/twitter_videos/<descriptive_name>.mp4`
- Temp files: `/tmp/make_video_<name>/`

---

### Workflow 1 — Foveation Trajectory Grid (2×2 stack)

Stack the first four `video_with_circle.mp4` clips from a `fov_traj` result folder into a 2×2 grid video.

```
Input:  /home/brianchc/fov_video/results/fov_traj/{ID}/000/video_with_circle.mp4
        /home/brianchc/fov_video/results/fov_traj/{ID}/001/video_with_circle.mp4
        /home/brianchc/fov_video/results/fov_traj/{ID}/002/video_with_circle.mp4
        /home/brianchc/fov_video/results/fov_traj/{ID}/003/video_with_circle.mp4
Output: twitter_videos/fov_traj_{ID}_2x2.mp4
```

```bash
conda run -n imp_tome python make_video.py stack \
  /home/brianchc/fov_video/results/fov_traj/{ID}/000/video_with_circle.mp4 \
  /home/brianchc/fov_video/results/fov_traj/{ID}/001/video_with_circle.mp4 \
  /home/brianchc/fov_video/results/fov_traj/{ID}/002/video_with_circle.mp4 \
  /home/brianchc/fov_video/results/fov_traj/{ID}/003/video_with_circle.mp4 \
  --output .../twitter_videos/fov_traj_{ID}_2x2.mp4 \
  --grid-x 2 --grid-y 2 --margin 10
```

Output: `1862×1078 @ 15fps` (source clips are 832×480, scaled ~1.11×, 10px margin between cells).

---

### Workflow 2 — Naive vs Ours Baseline Comparison (1×2 with headers)

Side-by-side comparison of naive mixed-resolution and our method for a given video ID,
with header bars and optional bottom-center text overlays.

```
Input:  baseline_comparison/naive_720p/videos_with_circle/{ID}.mp4
        baseline_comparison/ours_720p/videos_with_circle/{ID}.mp4
Output: twitter_videos/naive_vs_ours_id{ID}_header_1x2.mp4
```

Pipeline:
1. Add header "Naive mixed-resolution" → `naive_labeled.mp4` (font 55px = 5%×1.5×736, header-height 96)
2. Add header "Foveated Diffusion (Ours)" → `ours_labeled.mp4` (same height)
3. Stack 1×2 with 15px margin, `--target-size none`

---

### Workflow 3 — Saliency 3×3 Grid (circle videos, rose trajectory)

Stack nine 720p saliency circle videos in a 3×3 grid (VR Gaming, Autonomous Driving, Robotics).

```
Input:  saliency/{games,av,robotics}_fix_radius_720p/videos_with_circle/{ID}.mp4
IDs:    games=[8,15,25], av=[2,6,11], robotics=[11,67,65]
Output: twitter_videos/saliency_circle_3x3.mp4
```

```bash
conda run -n imp_tome python make_video.py stack \
  games_fix_radius_720p/videos_with_circle/8.mp4  ... (9 videos total) \
  --output .../saliency_circle_3x3.mp4 \
  --grid-x 3 --grid-y 3 --margin 15
```

---

### Workflow 4 — Foveation Trajectory + Tokenization (1×/2×/4× concat with rose curve)

Full pipeline: rose trajectory video for each resolution, paired 1×2 with tokenization masks,
labeled with resolution at bottom-center, concatenated across resolutions with 1s freeze.

```
Input (images):
  fov_traj_grid_10x10_1x_v2/random/{ID}/          → 1x (full res)
  fov_traj_grid_10x10_2x_v2/random/{ID}/          → 2x
  fov_traj_grid_10x10_4x_v2/random_4x_distill/{ID}/ → 4x

Input (tokenization masks):
  fov_traj_grid_10x10_1x_v2/random/tokenization_masks/
  fov_traj_grid_10x10_2x_v2/random/tokenization_masks/
  teaser_grid_10x10_4x_v2/random_4x_distill/tokenization_masks/

Output: twitter_videos/fov_traj_{ID}_rose_1x_2x_4x_tok_labeled_concat.mp4
```

Per-resolution pipeline (repeat for 1×/2×/4×, with `--num-frames 90` for 1× and `240` for 2×/4×):
1. `images` with `--trajectory rose --num-frames N --grid-rows 10 --grid-cols 10` → image rose video
2. Same for tokenization masks → tok rose video
3. `text --header --header-height 114 --text "Generated Image"` → img_h.mp4
4. `text --header --header-height 114 --text "Tokenization"` → tok_h.mp4
5. `stack --grid-x 2 --grid-y 1 --margin 20 --target-size none` → stacked.mp4
6. `text --position bottom-center --outline-width 3 --text "Full high-resolution"/"1/2x periphery"/"1/4x periphery"` → final.mp4

Then: `concat 1x_final.mp4 2x_final.mp4 4x_final.mp4 --sleep 1.0 --freeze`

---

### Workflow 5 — Saliency-guided vs Randomized Mask (1×2 with headers)

Side-by-side rose trajectory comparison of random vs saliency-guided foveation positioning
for a given prompt ID. No tokenization masks, no bottom-center overlay.

```
Input:  vary_position_grid_10x10/random/{ID}/     → Randomized Mask (left)
        vary_position_grid_10x10/saliency/{ID}/   → Saliency-guided (right)
Output: twitter_videos/vary_position_{ID}_rose_1x2.mp4
```

Pipeline:
1. `images --trajectory rose --num-frames 240 --fps 30 --grid-rows 10 --grid-cols 10` for each folder
2. `text --header --header-height 114 --text "Randomized Mask"` (font 82px, CMU Bold)
3. `text --header --header-height 114 --text "Saliency-guided"` (same)
4. `stack --grid-x 2 --grid-y 1 --margin 20 --target-size none`

---

### Workflow 6 — Rose Trajectory with Wipe Transition to Tokenization Mask

Rose animation for a single prompt ID that wipes to the tokenization mask at the midpoint
and back, with text labels transitioning alongside the content via the slider.

```
Input (images):   .../website/fov_traj_grid_10x10_4x_v2/random_4x_distill/{ID}/
Input (masks):    .../website/teaser_grid_10x10_4x_v2/random_4x_distill/tokenization_masks/
Script:           /tmp/make_rose_wipe_{ID}.py  (custom per-run)
Output:           twitter_videos/fov_traj_{ID}_rose_wipe.mp4
```

**Video structure** (390 frames = 13s @ 30fps):

| Segment | Frames | Duration | Content |
|---|---|---|---|
| Image rose animation | 0–119 | 4s | rose trajectory over images — "Generated Image" label |
| Freeze image | 120–134 | 0.5s | frozen image — "Generated Image" label |
| Wipe L→R | 135–164 | 1s | mask grows from left, image retreats right — both labels slide with content |
| Hold mask | 165–224 | 2s | frozen mask — "Tokenization" label |
| Freeze mask | 225–239 | 0.5s | frozen mask — "Tokenization" label |
| Wipe R→L | 240–269 | 1s | mask shrinks from right (left edge fixed) — both labels slide with content |
| Continue image rose | 270–389 | 4s | rose trajectory continues — "Generated Image" label |

**Key parameters:**
- Rose trajectory: 240 frames, 10×10 grid, `k=4` (8 petals)
- Slider bar: 10px white vertical bar at the wipe edge
- Text: CMU Serif Bold 82px, white, black outline (3px), top-center, padding 20px
- Labels pre-baked onto each side before compositing so they transition with the content

```bash
conda run -n imp_tome python make_video.py wipe \
  .../website/fov_traj_grid_10x10_4x_v2/random_4x_distill/{ID} \
  .../website/teaser_grid_10x10_4x_v2/random_4x_distill/tokenization_masks \
  --output .../twitter_videos/fov_traj_{ID}_rose_wipe.mp4
```

All parameters (fps, num-frames, grid-rows/cols, wipe-frames, hold-frames, freeze-frames, slider-width, img-label, mask-label, font, font-size, color, outline-width, padding) use the defaults shown above unless overridden.

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
| `random_circular` | Same as `circular` but angles are sampled uniformly at random (seeded) instead of evenly spaced. |
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
| `--random_mask_center` | false | Randomize the mask center per prompt in full eval instead of centering it. Prompt `i` uses `foveation_center_for_index(i, mask_center_seed, mask_center_range)`, which is index-addressable, so the mask does not depend on shard count and is identical across ours/naive/ToMe/RALU. Adds `center_x,center_y` columns to `metadata_*.csv`. |
| `--mask_center_seed` | 42 | Seed for `--random_mask_center`, independent of `--seed`. Every baseline in a comparison must be given the same value. |
| `--mask_center_range` | 0.3 | Centers drawn uniformly from `[-range, range]^2` in normalized coords (0.3 matches the `user_study` path). Large radii clip the frame edge at large ranges, which shrinks the foveal area. |
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
  metadata_00000.csv   # image filename + prompt per row (one file per subset).
                       # With --random_mask_center, also center_x,center_y per row.
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
