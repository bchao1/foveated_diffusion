#!/usr/bin/env python
"""Shared machinery for the ICLR-2027 qualitative baseline figures.

Three entry points sit on top of this module, one per figure:

    make_main_qual.py          paper_figures/baseline_comparison_randomized.png
    make_supp_101.py           paper_figures/supp/baseline_101_{a,b}.png
    make_supp_123.py           paper_figures/supp/baseline_{1,2,3}_{a,b}.png

Each one does both halves of the job: it re-runs the ToMe and RALU arms for the tiles
that figure needs, then composes the figure. `high_res`, `naive` and `ours` are never
regenerated — they are read from the frozen `user_study` run in
`results/eccv2026_final_v2/baseline_comparison_randomized_{1,2,3}/`.

Why the two regenerated arms are configured the way they are
------------------------------------------------------------
The frozen columns come from `inference.py --experiment user_study`, which

  * draws one fixation point per prompt from `random.uniform(-0.3, 0.3)` on the `random`
    module seeded with `--seed 42`, at r=0.33, circular, lr_factor 2 (inference.py:1150).
    Those draws are the RNG's first consumers, so replaying the seed reproduces them;
    `verify_centers()` checks the replay against the saved `fixation_point.png`.
  * clears LoRA for the high_res and naive passes and loads the foveation LoRA only for
    `ours` (inference.py:1141, 1215).

So the ToMe/RALU columns run on the **base** model with **no LoRA** — matching the
columns they sit beside — and on the **same per-prompt mask**. Everything else (50 steps,
cfg 4.0, seed 42, `merge_mlp=true`, RALU 3-stage level 4 compute-matched to ours@50 with
z_max 6.5) is exactly what `eval.sh`'s `eval_tome` / `eval_ralu` pass today.

This differs from `eval.sh` in two deliberate places, both forced by the frozen columns:
`eval.sh` gives every training-free arm the no-fov LoRA, and it takes its mask from
`MASK_CONFIGS` + `foveation_center_for_index`. Neither is available here without also
regenerating high_res/naive/ours.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(EVAL_DIR)

RESULTS = os.path.join(REPO, "results", "eccv2026_final_v2")
FIGURES = os.path.join(REPO, "paper_figures")
# Regenerated ToMe/RALU land here, never on top of the Jul-27 `_extra` tree, so the two
# can be compared.
REGEN_ROOT = os.path.join(REPO, "results", "iclr_2027_qual")

PROMPT_SETS = {
    1: "prompts/claude_prompts.csv",
    2: "prompts/claude_prompts_extended.csv",
    3: "prompts/claude_prompts_extended_v2.csv",
}
NUM_PROMPTS = 100          # every set is exactly 100 rows; folder index == CSV row index

# Fixed by run_user_study — not knobs.
MASK_RADIUS = 0.33
MASK_SHAPE = "circular"
LR_FACTOR = 2
HEIGHT = WIDTH = 1024

# eval.sh's `base` model settings, shared by every arm.
MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
STEPS = 50
CFG = 4.0
SEED = 42

# eval.sh's method-specific knobs.
MERGE_MLP = True                 # tome
RALU_STAGES = 3
RALU_LEVEL = 4
RALU_Z_MAX = 6.5
OURS_STEPS = 50                  # ralu compute-match target

# Outline drawn on the mixed-resolution arms, matching img_naive / img_ours.
OUTLINE_WIDTH_FRAC = 0.005
OUTLINE_COLOR = (255, 0, 0)

PY = sys.executable

# ---------------------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------------------
# arm -> (filename in the tile directory, column label). "tokenization" is synthesized.
ARMS = {
    "high_res": ("img_high_res.png", "Full resolution"),
    "tome":     ("img_tome.png",     "Token merging"),
    "ralu":     ("img_ralu.png",     "RALU"),
    "naive":    ("img_naive.png",    "Naive mixed-res."),
    "ours":     ("img_ours.png",     "Ours"),
    "tokenization": (None,           "Tokenization"),
}
REGEN_ARMS = ("tome", "ralu")


def label_for(arm):
    return ARMS[arm][1]


def setup_env():
    """What eval.sh exports before invoking any driver."""
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/miele/brian/modelscope")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
    # The imp_tome env otherwise resolves `diffsynth` to /home/brianchc/DiffSynth-Studio,
    # a different checkout without the foveated pipeline. Force this repo's copy.
    for p in (os.path.join(REPO, "DiffSynth-Studio"),
              EVAL_DIR,
              os.path.join(EVAL_DIR, "importance-token-merge"),
              os.path.join(EVAL_DIR, "RALU")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------------------
# Prompts, masks
# ---------------------------------------------------------------------------------------

def load_prompts(prompt_set):
    import pandas as pd
    path = os.path.join(EVAL_DIR, PROMPT_SETS[prompt_set])
    df = pd.read_csv(path)
    assert len(df) == NUM_PROMPTS, f"{path}: {len(df)} rows, expected {NUM_PROMPTS}"
    return list(df["prompt"])


def user_study_centers(seed=SEED, n=NUM_PROMPTS):
    """Replay run_user_study's per-prompt fixation points (inference.py:1150)."""
    import random
    rng = random.Random(seed)
    return [(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)) for _ in range(n)]


def build_masks(indices, device="cpu"):
    """{idx: (tome_mask, ralu_mask, full_res_mask)} for the prompt indices we need.

    `tome_mask` is the (H/16, W/16) token grid our pipeline uses; `ralu_mask` is the
    (H/32, W/32) coarse block grid RALU's stage 1 works on. Same geometry either way.
    """
    from inference import create_foveation_mask, create_foveation_mask_full_res
    from RALU_fov_inference_diffsynth import coarse_foveation_mask

    centers = user_study_centers()
    out = {}
    for idx in sorted(set(indices)):
        c = centers[idx]
        out[idx] = (
            create_foveation_mask(HEIGHT, WIDTH, center=c, r=MASK_RADIUS, shape=MASK_SHAPE,
                                  device=device, lr_factor=LR_FACTOR),
            coarse_foveation_mask(HEIGHT, WIDTH, center=c, r=MASK_RADIUS, shape=MASK_SHAPE,
                                  lr_factor=LR_FACTOR),
            create_foveation_mask_full_res(HEIGHT, WIDTH, center=c, r=MASK_RADIUS,
                                           shape=MASK_SHAPE, device=device),
        )
    return out


def verify_centers(tiles):
    """Check the replayed centers against the saved fixation_point.png of every tile.
    CPU only, loads no model. Raises if the replay drifted."""
    import imageio.v2 as imageio

    centers = user_study_centers()
    worst, checked = 0.0, 0
    for prompt_set, idx in tiles:
        p = os.path.join(RESULTS, f"baseline_comparison_randomized_{prompt_set}",
                         f"{idx:05d}", "fixation_point.png")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        im = imageio.imread(p)
        ys, xs = np.nonzero(im[..., 0] > 128)
        h, w = im.shape[:2]
        worst = max(worst, abs(xs.mean() / w - 0.5 - centers[idx][0]),
                    abs(ys.mean() / h - 0.5 - centers[idx][1]))
        checked += 1
    print(f"verify_centers: {checked} tiles, worst |center error| = {worst:.2e} "
          f"(dot rasterization alone is ~1e-4)")
    if worst > 1e-3:
        raise SystemExit("center replay does not match the saved fixation points")


# ---------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------

def frozen_dir(prompt_set, idx):
    return os.path.join(RESULTS, f"baseline_comparison_randomized_{prompt_set}",
                        f"{idx:05d}")


def old_extra_dir(prompt_set, idx):
    """The Jul-27 ToMe/RALU tiles, kept only for the regression diff."""
    return os.path.join(RESULTS, f"baseline_comparison_randomized_{prompt_set}_extra",
                        f"{idx:05d}")


def regen_dir(prompt_set, idx):
    return os.path.join(REGEN_ROOT, f"set{prompt_set}", f"{idx:05d}")


def arm_path(arm, prompt_set, idx):
    """Where a column's image lives. Regenerated arms come from REGEN_ROOT, falling back
    to the frozen `_extra` tree so a figure can still be built without a GPU."""
    fname = ARMS[arm][0]
    if arm in REGEN_ARMS:
        p = os.path.join(regen_dir(prompt_set, idx), fname)
        if os.path.isfile(p):
            return p
        return os.path.join(old_extra_dir(prompt_set, idx), fname)
    return os.path.join(frozen_dir(prompt_set, idx), fname)


def missing(arms, tiles, resolver=None):
    if resolver is not None:
        return [(a, s, i) for s, i in tiles for a in arms if not resolver.has(a, s, i)]
    return [(a, s, i) for s, i in tiles for a in arms
            if not os.path.isfile(arm_path(a, s, i))]


# ---------------------------------------------------------------------------------------
# Distilled tiles
# ---------------------------------------------------------------------------------------

DISTILLED_ROOT = os.path.join(REPO, "results", "iclr_2027_distilled_all")
DISTILLED_TAG = f"{MASK_SHAPE}_r{MASK_RADIUS}_randcenter42"


class DistilledTiles:
    """Resolve (arm, prompt_set, csv_row) to an image from an eval.sh distilled tree.

    eval.sh names images by their position in the prompt csv, not by the prompt set and
    row the base-model figures are keyed on, so `tile_index.csv` (written alongside the
    prompts) is the join. The eval drivers write un-outlined images, so the foveal
    boundary is drawn here from the `center_x`/`center_y` in the manifests, matching the
    white circle on the base-model figures.
    """

    def __init__(self, root=DISTILLED_ROOT, tag=DISTILLED_TAG, radius=MASK_RADIUS,
                 outline=True):
        import pandas as pd
        self.root, self.tag, self.radius, self.outline = root, tag, radius, outline
        index_path = os.path.join(root, "tile_index.csv")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{index_path} — run make_supp_distilled.py --tiles all first")
        idx = pd.read_csv(index_path)
        self.row_of = {(int(r.prompt_set), int(r.prompt_row)): int(r.row)
                       for r in idx.itertuples()}
        self.centers = self._read_centers()

    def _dir(self, arm):
        return (os.path.join(self.root, "high_res") if arm == "high_res"
                else os.path.join(self.root, self.tag, arm))

    def _read_centers(self):
        import glob as _glob
        import pandas as pd
        for arm in ("ours", "naive", "tome", "ralu"):     # high_res records no center
            files = sorted(_glob.glob(os.path.join(self._dir(arm), "metadata_*.csv")))
            if not files:
                continue
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            if "center_x" not in df.columns:
                continue
            return {int(r["image"].split("_")[1].split(".")[0]):
                    (r["center_x"], r["center_y"]) for _, r in df.iterrows()}
        print("WARNING: no center_x/center_y found; the foveal outline will not be drawn")
        return {}

    def path(self, arm, prompt_set, idx):
        row = self.row_of.get((prompt_set, idx))
        if row is None:
            raise KeyError(f"set {prompt_set} row {idx} is not in {self.root}/tile_index.csv")
        return os.path.join(self._dir(arm), f"img_{row:010d}.png"), row

    def has(self, arm, prompt_set, idx):
        if (prompt_set, idx) not in self.row_of:
            return False
        return os.path.isfile(self.path(arm, prompt_set, idx)[0])

    def __call__(self, arm, prompt_set, idx):
        from inference import create_foveation_mask_full_res, draw_foveation_outline
        path, row = self.path(arm, prompt_set, idx)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        img = np.asarray(Image.open(path).convert("RGB")).copy()
        if self.outline and arm != "high_res" and row in self.centers:
            mask = create_foveation_mask_full_res(HEIGHT, WIDTH, center=self.centers[row],
                                                  r=self.radius, shape=MASK_SHAPE)
            draw_foveation_outline(img, mask, HEIGHT, WIDTH,
                                   outline_width_frac=OUTLINE_WIDTH_FRAC,
                                   color=OUTLINE_COLOR)
        return img


# ---------------------------------------------------------------------------------------
# Inference — the two regenerated arms
# ---------------------------------------------------------------------------------------

def _model_configs(ModelConfig):
    return (
        [
            ModelConfig(model_id=MODEL_ID, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=MODEL_ID, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=MODEL_ID, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="tokenizer/"),
    )


def _save(image, out_dir, arm, full_res_mask):
    """Write the plain image plus an outlined copy, matching img_naive / img_ours."""
    import imageio.v2 as imageio
    from inference import draw_foveation_outline

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"img_{arm}_raw.png")
    image.save(raw_path)
    img = imageio.imread(raw_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    draw_foveation_outline(img, full_res_mask, HEIGHT, WIDTH,
                           outline_width_frac=OUTLINE_WIDTH_FRAC, color=OUTLINE_COLOR)
    imageio.imwrite(os.path.join(out_dir, f"img_{arm}.png"), img)


def run_tome(tasks, masks):
    """tasks: [(prompt_set, idx, prompt)]. Mirrors eval.sh's eval_tome, minus the LoRA."""
    import torch
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    from flux2_tome import apply_patch

    torch.manual_seed(SEED)
    model_configs, tokenizer_config = _model_configs(ModelConfig)
    print("[tome] loading stock FLUX.2 pipeline (no foveated pipeline, no TokenAE)...")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )
    lh, lw = HEIGHT // 16, WIDTH // 16
    for prompt_set, idx, prompt in tasks:
        t0 = time.time()
        tome_mask, _, full_mask = masks[idx]
        # Re-patching per prompt is cheap next to a 50-step sample.
        state = apply_patch(pipe.dit, tome_mask.to(pipe.device), lh, lw,
                            lr_factor=LR_FACTOR, merge_mlp=MERGE_MLP, enabled=True)
        image = pipe(prompt=prompt, height=HEIGHT, width=WIDTH, seed=SEED,
                     rand_device="cuda", num_inference_steps=STEPS, cfg_scale=CFG)
        _save(image, regen_dir(prompt_set, idx), "tome", full_mask)
        print(f"[set {prompt_set} {idx:05d}] tome {time.time() - t0:.1f}s  "
              f"{state['merger'].num_merged}/{state['merger'].num_tokens} tokens  "
              f"{prompt[:55]}", flush=True)
    del pipe
    torch.cuda.empty_cache()


def run_ralu(tasks, masks):
    """tasks: [(prompt_set, idx, prompt)]. Mirrors eval.sh's eval_ralu, minus the LoRA."""
    import torch
    from diffsynth.pipelines.flux2_image import ModelConfig
    from pipeline_flux2_diffsynth_RALU_fov import (
        Flux2DiffSynthRALUFovPipeline, ralu_schedule, stage_token_counts,
    )

    model_configs, tokenizer_config = _model_configs(ModelConfig)
    print("[ralu] loading FLUX.2 RALU-foveated pipeline...")
    pipe = Flux2DiffSynthRALUFovPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )
    pipe.id_frame = "coarse"

    seq_len = (HEIGHT // 16) * (WIDTH // 16)
    for prompt_set, idx, prompt in tasks:
        _, ralu_mask, full_mask = masks[idx]
        m = int(ralu_mask.sum())
        # The mask moves with the fixation point, so m moves too: re-solve the matched
        # schedule, the NT-DM params and the correlated noise for every prompt.
        N, e, info = ralu_schedule(
            total_steps=STEPS, level=RALU_LEVEL, num_stages=RALU_STAGES,
            match_compute=True, height=HEIGHT, width=WIDTH, f0=LR_FACTOR,
            n_hr_blocks=m, ours_steps=OURS_STEPS,
        )
        pipe.set_foveation_mask(ralu_mask)
        pipe.set_params(N=N, e=e, seq_len=seq_len, f0=LR_FACTOR, z_max=RALU_Z_MAX,
                        shift_mode="ntdm",
                        stage_tokens=stage_token_counts(HEIGHT, WIDTH, LR_FACTOR, m,
                                                        RALU_STAGES))
        pipe.generate_ralu_noise(HEIGHT, WIDTH, seed=SEED)
        t0 = time.time()
        image = pipe(prompt=prompt, cfg_scale=CFG, embedded_guidance=CFG,
                     height=HEIGHT, width=WIDTH, seed=SEED)
        _save(image, regen_dir(prompt_set, idx), "ralu", full_mask)
        print(f"[set {prompt_set} {idx:05d}] ralu {time.time() - t0:.1f}s N={N} m={m} "
              f"ratio={info.get('cost_ratio')}  {prompt[:50]}", flush=True)
    del pipe
    torch.cuda.empty_cache()


def run_worker(tiles, arms, subset_idx, num_subsets):
    """One shard, in this process, on whatever GPU the caller pinned."""
    setup_env()
    masks = build_masks([i for _, i in tiles])
    prompts = {s: load_prompts(s) for s in sorted({s for s, _ in tiles})}
    mine = [(s, i, prompts[s][i]) for k, (s, i) in enumerate(sorted(tiles))
            if k % num_subsets == subset_idx]
    print(f"shard {subset_idx}/{num_subsets}: {len(mine)} tiles x {len(arms)} arms")
    for arm in arms:
        {"tome": run_tome, "ralu": run_ralu}[arm](mine, masks)


def launch(script, tiles, gpus, log_dir, arms=REGEN_ARMS):
    """Fork one worker per GPU, each pinned with CUDA_VISIBLE_DEVICES, and wait.

    Skips tiles whose regenerated images are all present, so a re-run resumes.
    """
    todo = sorted({(s, i) for s, i in tiles
                   if any(not os.path.isfile(os.path.join(regen_dir(s, i), ARMS[a][0]))
                          for a in arms)})
    if not todo:
        print(f"regen: all {len(set(tiles))} tiles already present in {REGEN_ROOT}")
        return 0
    os.makedirs(log_dir, exist_ok=True)
    gpu_list = [g for g in str(gpus).split() if g]
    print(f"regen: {len(todo)}/{len(set(tiles))} tiles x {len(arms)} arms "
          f"over GPUs {' '.join(gpu_list)} -> {REGEN_ROOT}")
    spec = ",".join(f"{s}:{i}" for s, i in todo)
    procs = []
    for k, gpu in enumerate(gpu_list):
        log = os.path.join(log_dir, f"regen_shard{k}_gpu{gpu}.log")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [PY, script, "--worker", "--tiles", spec, "--arms", ",".join(arms),
               "--subset_idx", str(k), "--num_subsets", str(len(gpu_list))]
        print(f"  shard {k} -> GPU {gpu}  ({log})")
        procs.append((subprocess.Popen(cmd, env=env, stdout=open(log, "w"),
                                       stderr=subprocess.STDOUT), log))
    rc = 0
    for p, log in procs:
        if p.wait() != 0:
            rc = 1
            print(f"  !! shard failed, see {log}", file=sys.stderr)
    return rc


def parse_tiles(spec):
    return [tuple(int(x) for x in t.split(":")) for t in spec.split(",") if t]


def diff_against_old(tiles, arms=REGEN_ARMS):
    """Pixel-diff the regenerated arms against the Jul-27 `_extra` tree.

    The ToMe/RALU code lived in submodule gitlinks with no nested `.git` until 38ab179
    (Jul 29), two days after those images were written, so there is no history for that
    window. This is the empirical answer to "did anything change?".
    """
    print("\nregression vs the existing _extra images (raw, pre-outline):")
    worst = 0.0
    for arm in arms:
        diffs = []
        for s, i in sorted(set(tiles)):
            new = os.path.join(regen_dir(s, i), f"img_{arm}_raw.png")
            old = os.path.join(old_extra_dir(s, i), f"img_{arm}_raw.png")
            if not (os.path.isfile(new) and os.path.isfile(old)):
                continue
            a = np.asarray(Image.open(new).convert("RGB")).astype(np.int16)
            b = np.asarray(Image.open(old).convert("RGB")).astype(np.int16)
            diffs.append((np.abs(a - b).mean(), np.abs(a - b).max()))
        if not diffs:
            print(f"  {arm:5s}: no overlapping tiles to compare")
            continue
        mean = max(d[0] for d in diffs)
        mx = max(d[1] for d in diffs)
        worst = max(worst, mean)
        verdict = ("identical" if mx == 0 else
                   "differs" if mean > 1.0 else "near-identical (PNG/round-off)")
        print(f"  {arm:5s}: {len(diffs)} tiles, worst mean |diff| {mean:6.3f}/255, "
              f"worst max {mx:3d}/255  -> {verdict}")
    return worst


# ---------------------------------------------------------------------------------------
# Figure composition
# ---------------------------------------------------------------------------------------

FONT_CANDIDATES = [
    "/home/brianchc/fonts/cmu-serif/cmunbx.ttf",          # CMU Serif Bold
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
]
# Label height as a fraction of ONE tile's height.
FONT_FRAC = 0.125

# Token-size ramp from LoT (precompute/flux/vrs_image.py:_TOKEN_RAMP): warm, muted,
# dark = fine tokens, light = coarse. Sampled at log2(extent)/log2(size_max=8) so a given
# token size is the same color in every figure.
_TOKEN_RAMP = (
    (124, 58, 45), (181, 101, 74), (208, 140, 96),
    (224, 172, 126), (239, 207, 160), (245, 230, 200),
)


def _ramp_color(f):
    x = f * (len(_TOKEN_RAMP) - 1)
    i = min(int(x), len(_TOKEN_RAMP) - 2)
    a, b, w = _TOKEN_RAMP[i], _TOKEN_RAMP[i + 1], x - i
    return tuple(int(round(a[c] + (b[c] - a[c]) * w)) for c in range(3))


def tokenization_mask_rgb(fov_mask, lr_factor=LR_FACTOR, size_max=8, border_px=4):
    """The HR/LR token grid, colored by token size with LoT's absolute palette.

    Foveal tokens span 1 latent cell (16 px), peripheral tokens span `lr_factor` cells,
    so they take ramp(0) and ramp(log2(lr_factor)/log2(size_max)) respectively. Black
    borders `border_px` wide, drawn on the grid each region actually uses.
    """
    mask = fov_mask.detach().cpu().numpy() if hasattr(fov_mask, "detach") else np.asarray(fov_mask)
    up = np.repeat(np.repeat(mask, 16, axis=0), 16, axis=1)[:HEIGHT, :WIDTH]
    hr_region = up > 0.5

    denom = float(np.log2(size_max))
    vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    vis[hr_region] = _ramp_color(0.0)
    vis[~hr_region] = _ramp_color(min(float(np.log2(lr_factor)) / denom, 1.0))

    yy = np.arange(HEIGHT)[:, None]
    xx = np.arange(WIDTH)[None, :]
    hr_grid = ((yy % 16) < border_px) | ((xx % 16) < border_px)
    lr_grid = ((yy % (16 * lr_factor)) < border_px) | ((xx % (16 * lr_factor)) < border_px)
    vis[(hr_grid & hr_region) | (lr_grid & ~hr_region)] = 0

    vis[:border_px, :] = vis[-border_px:, :] = 0
    vis[:, :border_px] = vis[:, -border_px:] = 0
    return vis


def load_font(px):
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return ImageFont.truetype(path, px)
    print("WARNING: no serif TTF found, falling back to PIL's bitmap font")
    return ImageFont.load_default()


def downsample(arr, factor):
    from skimage.transform import resize
    if factor <= 1:
        return arr
    h, w = arr.shape[:2]
    out = resize(arr, (h // factor, w // factor) + arr.shape[2:],
                 order=1, anti_aliasing=True, preserve_range=True)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def white(h, w):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def gap_list(n_cols, pad, gaps=None):
    """Per-gutter widths. `gaps` (len n_cols-1) overrides the uniform `pad`."""
    if gaps is None:
        return [pad] * (n_cols - 1)
    assert len(gaps) == n_cols - 1, f"{len(gaps)} gaps for {n_cols} columns"
    return [max(1, int(g)) for g in gaps]


def col_offsets(col_width, gaps):
    """Left x of every column, given per-gutter widths."""
    xs = [0]
    for g in gaps:
        xs.append(xs[-1] + col_width + g)
    return xs


def hstack(images, pad):
    """`pad` is a single gutter width or a per-gutter list."""
    h = max(im.shape[0] for im in images)
    gaps = pad if isinstance(pad, (list, tuple)) else [pad] * (len(images) - 1)
    parts = []
    for k, im in enumerate(images):
        if k:
            parts.append(white(h, gaps[k - 1]))
        parts.append(im)
    return np.concatenate(parts, axis=1)


def vstack(rows, pad):
    w = max(r.shape[1] for r in rows)
    parts = []
    for k, r in enumerate(rows):
        if k:
            parts.append(white(pad, w))
        parts.append(r)
    return np.concatenate(parts, axis=0)


def label_band(labels, col_width, pad, font_px, pad_below=None):
    """A white band with one label centered over each column. "" leaves a column blank.
    `pad` is a single gutter width or a per-gutter list."""
    gaps = gap_list(len(labels), pad if isinstance(pad, int) else 0,
                    pad if isinstance(pad, (list, tuple)) else None)
    pad_below = (min(gaps) if pad_below is None else pad_below)
    font = load_font(max(8, int(font_px)))
    xs = col_offsets(col_width, gaps)
    height = int(font_px * 1.35) + pad_below
    band = Image.new("RGB", (xs[-1] + col_width, height), (255, 255, 255))
    draw = ImageDraw.Draw(band)
    for i, text in enumerate(labels):
        if not text:
            continue
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((xs[i] + (col_width - tw) / 2 - bbox[0],
                   (height - pad_below - th) / 2 - bbox[1]),
                  text, font=font, fill=(0, 0, 0))
    return np.asarray(band)


def span_band(groups, n_cols, col_width, pad, font_px, pad_below=None, stroke=None):
    """A band carrying a bracketed caption over a *range* of columns.

    `groups` is a list of (first_col, last_col, text), inclusive. Each becomes a
    horizontal rule with downward end ticks spanning those columns, captioned above it —
    the standard way to say "this label describes these columns and not the others".
    """
    gaps = gap_list(n_cols, pad if isinstance(pad, int) else 0,
                    pad if isinstance(pad, (list, tuple)) else None)
    pad_below = (min(gaps) if pad_below is None else pad_below)
    stroke = max(1, int(font_px * 0.05)) if stroke is None else stroke
    tick = max(2, int(font_px * 0.30))
    font = load_font(max(8, int(font_px)))
    xs = col_offsets(col_width, gaps)
    height = int(font_px * 1.35) + tick + pad_below
    band = Image.new("RGB", (xs[-1] + col_width, height), (255, 255, 255))
    draw = ImageDraw.Draw(band)
    for first, last, text in groups:
        x0, x1 = xs[first], xs[last] + col_width
        y = height - pad_below - tick
        draw.rectangle([x0, y, x1, y + stroke], fill=(0, 0, 0))
        for x in (x0, x1 - stroke):
            draw.rectangle([x, y, x + stroke, y + tick], fill=(0, 0, 0))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((x0 + x1 - tw) / 2 - bbox[0], (y - th) / 2 - bbox[1]),
                  text, font=font, fill=(0, 0, 0))
    return np.asarray(band)


def load_cell(cell, ds, masks=None, resolver=None):
    """One grid cell: an (arm, prompt_set, idx) triple, "blank", or a ready ndarray.

    `resolver` overrides where an arm's image comes from (see DistilledTiles); the
    synthesized tokenization column is unaffected by it.
    """
    # ndarray first: `cell == "blank"` on an array is an elementwise compare, not a test.
    if isinstance(cell, np.ndarray):
        return downsample(cell, ds)
    if cell == "blank":
        return None
    arm, prompt_set, idx = cell
    if arm == "tokenization":
        return downsample(tokenization_mask_rgb(masks[idx][0]), ds)
    if resolver is not None:
        return downsample(resolver(arm, prompt_set, idx), ds)
    path = arm_path(arm, prompt_set, idx)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return downsample(np.asarray(Image.open(path).convert("RGB")), ds)


def compose(blocks, out_path, downsample_factor, pad_px=20, font_frac=FONT_FRAC,
            gaps_px=None, groups=None, rules=(), rule_span="rows", resolver=None):
    """Stack labelled blocks vertically.

    `blocks` is a list of (labels, rows), where `labels` is one string per column ("" for
    a blank column, None for no band at all) and `rows` is a list of cell lists. Every
    row in a block must have the same number of columns. `font_frac` is the label height
    as a fraction of one tile's height.

    `gaps_px` gives per-gutter widths at full resolution (len = n_cols - 1) instead of a
    uniform `pad_px`; `groups` is a list of (first_col, last_col, text) rendered as a
    bracketed caption above the column labels; `rules` names gutter indices to draw a
    vertical divider through, spanning either the image rows (`rule_span="rows"`) or the
    whole figure including the label band (`"full"`). Together these separate a reference
    column from a group of columns that share a condition.
    """
    ds = max(1, downsample_factor)
    pad = max(1, pad_px // ds)
    masks = build_masks([c[2] for labels, rows in blocks for row in rows for c in row
                         if isinstance(c, tuple) and c[0] == "tokenization"]) or {}
    n_cols = len(blocks[0][1][0])
    gaps = gap_list(n_cols, pad, None if gaps_px is None
                    else [max(1, g // ds) for g in gaps_px])
    pieces = []
    y = 0
    first_row_y = None
    for labels, rows in blocks:
        tiles = [[load_cell(c, ds, masks, resolver) for c in row] for row in rows]
        side = next(t.shape[0] for row in tiles for t in row if t is not None)
        tiles = [[t if t is not None else white(side, side) for t in row] for row in tiles]
        col_w = tiles[0][0].shape[1]
        for band in ([span_band(groups, n_cols, col_w, gaps, font_frac * side)] if groups
                     else []) + ([label_band(labels, col_w, gaps, font_frac * side)]
                                 if labels is not None else []):
            pieces.append(band)
            y += band.shape[0] + pad
        for row in tiles:
            if first_row_y is None:
                first_row_y = y
            pieces.append(hstack(row, gaps))
            y += side + pad
    fig = vstack(pieces, pad)

    # Divider through the named gutters, spanning the image rows or the whole figure.
    top = 0 if rule_span == "full" else (first_row_y or 0)
    for gi in rules:
        xs = col_offsets(col_w, gaps)
        centre = xs[gi] + col_w + gaps[gi] // 2
        w = max(1, gaps[gi] // 6)
        fig[top:, centre - w // 2: centre - w // 2 + w] = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(fig).save(out_path)
    print(f"wrote {out_path}  {fig.shape[1]}x{fig.shape[0]}  "
          f"downsample {downsample_factor}  {os.path.getsize(out_path) / 1e6:.1f} MB")


def add_common_args(ap, default_gpus="6 9"):
    ap.add_argument("--stage", default="all", choices=["all", "regen", "figure", "verify"],
                    help="all = regen then figure; verify = mask replay + old/new diff only")
    ap.add_argument("--gpus", default=default_gpus, help="space-separated GPU ids")
    ap.add_argument("--pad_px", type=int, default=20)
    ap.add_argument("--downsample", type=int, default=None, help="override the default")
    # worker mode, set by launch()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tiles", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arms", default=",".join(REGEN_ARMS), help=argparse.SUPPRESS)
    ap.add_argument("--subset_idx", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--num_subsets", type=int, default=1, help=argparse.SUPPRESS)
    return ap


def main_flow(args, script, tiles, build_figures, log_dir=None):
    """The body every figure script shares: worker dispatch, verify, regen, compose."""
    if args.worker:
        run_worker(parse_tiles(args.tiles), args.arms.split(","),
                   args.subset_idx, args.num_subsets)
        return 0

    setup_env()
    verify_centers(tiles)

    rc = 0
    if args.stage in ("all", "regen"):
        rc = launch(script, tiles, args.gpus,
                    log_dir or os.path.join(REGEN_ROOT, "logs"))
    if args.stage in ("all", "regen", "verify"):
        diff_against_old(tiles)
    if args.stage in ("all", "figure"):
        gaps = missing(["high_res", "naive", "ours", "tome", "ralu"], tiles)
        if gaps:
            raise SystemExit("missing inputs:\n  " +
                             "\n  ".join(f"{a} set{s} {i:05d}" for a, s, i in gaps))
        build_figures()
    return rc
