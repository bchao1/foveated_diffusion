#!/usr/bin/env python
"""Distilled-model supplementary baseline figure — paper_figures/supp/baseline_distilled.png.

One figure, 9 rows x 5 columns, the same structure and row count as each
`baseline_{1,2,3}_{a,b}.png`:

    Full resolution | Token merging | RALU | Naive mixed-res. | Ours

Every arm here runs on BFL's step- and guidance-distilled `FLUX.2-klein-4B` at 4 steps,
cfg 1.0 — so unlike the other qualitative figures, nothing is frozen and all five columns
are generated. Generation is delegated wholesale to `eval/eval.sh` with `MODEL=distilled`,
which already carries the distilled profile for all five arms; this script only chooses
the prompts, invokes it, and composes the result. Nothing about the five methods is
reimplemented here, so the figure cannot drift from the large-scale eval.

Prompts are sampled from the tiles the base-model supplementary figures already use
(`baseline_figure_tiles.json`, figures 1-3), three from each prompt set, with a fixed
seed. So a reader can put this figure next to `baseline_{1,2,3}_*` and see the same kind
of content under the distilled model.

The foveal boundary is drawn at composition time from the `center_x`/`center_y` columns
`eval.sh` records in `metadata_*.csv`, because the eval drivers write un-outlined images;
this matches the white circle on the other supplementary figures.

    ./make_supp_distilled.py                    # generate all five arms, then compose
    ./make_supp_distilled.py --stage figure     # compose only, no GPU
    ./make_supp_distilled.py --stage prompts    # print the sampled prompts, no GPU
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_common as qc  # noqa: E402

TILES_JSON = os.path.join(qc.EVAL_DIR, "baseline_figure_tiles.json")
EVAL_SH = os.path.join(qc.EVAL_DIR, "eval.sh")

COLUMNS = ["high_res", "tome", "ralu", "naive", "ours"]
# eval.sh's method name -> its output subdirectory under the mask tag. high_res is
# mask-independent and lives at the root.
OUT_SUBDIR = {m: (m if m != "high_res" else None) for m in COLUMNS}

OUT_ROOT = os.path.join(qc.REPO, "results", "iclr_2027_qual_distilled")
OUT_FIG = os.path.join(qc.FIGURES, "supp", "baseline_distilled.png")

# `--tiles all` generates every tile the base-model figures use, rather than the 9 the
# supplementary figure shows, so rows can be re-picked later without another GPU run. It
# writes to its own root: image indices are positions in the prompt csv, so a different
# tile list would silently re-key the sampled run's img_%010d files.
OUT_ROOT_ALL = os.path.join(qc.REPO, "results", "iclr_2027_distilled_all")
ALL_FIGURES = ["1", "2", "3", "101"]
MAIN_TILES = [(1, 8), (1, 10), (1, 3)]      # the main qualitative figure's rows

N_ROWS = 9                    # same row count as each baseline_{1,2,3}_{a,b}.png
SAMPLE_SEED = 0
SOURCE_FIGURES = ["1", "2", "3"]

# Mask geometry, matching the base-model supplementary figures so the two sit side by
# side. eval.sh randomizes the fixation point per prompt from its own index-addressable
# helper, which every arm shares.
MASK_CONFIG = f"{qc.MASK_SHAPE}:{qc.MASK_RADIUS}"
MASK_CENTER_SEED = 42

DOWNSAMPLE = 4                # 256 px per tile, matching baseline_{1,2,3}_{a,b}.png

# Local mirror of the distilled checkpoint. See --model_base_path: reading klein-4B
# straight off /miele is unworkable while that mount is busy. Falls back to eval.sh's
# default when the mirror is absent.
MODEL_BASE_PATH = ("/local/brian/models/hub"
                   if os.path.isdir("/local/brian/models/hub/black-forest-labs/FLUX.2-klein-4B")
                   else "")


def sample_tiles(seed=SAMPLE_SEED, n_rows=N_ROWS):
    """`n_rows` tiles drawn from the base-model supplementary figures, evenly across the
    three prompt sets and deterministic in `seed`."""
    with open(TILES_JSON) as f:
        raw = json.load(f)
    pool = {}
    for name in SOURCE_FIGURES:
        for prompt_set, idx in raw[name]:
            pool.setdefault(prompt_set, set()).add(idx)

    rng = random.Random(seed)
    per_set, extra = divmod(n_rows, len(pool))
    picked = []
    for k, prompt_set in enumerate(sorted(pool)):
        take = per_set + (1 if k < extra else 0)
        picked += [(prompt_set, i) for i in rng.sample(sorted(pool[prompt_set]), take)]
    return sorted(picked)


def all_tiles(figures=ALL_FIGURES, include_main=True):
    """Every distinct tile used by the named base-model figures, in a stable order."""
    with open(TILES_JSON) as f:
        raw = json.load(f)
    seen = {tuple(t) for name in figures for t in raw[name]}
    if include_main:
        seen |= set(MAIN_TILES)
    return sorted(seen)


def write_prompt_csv(tiles, path):
    """The prompts as an eval.sh-shaped csv. The row index is what every arm's
    img_%010d.png is keyed on, so it is the join key for everything downstream."""
    import pandas as pd
    rows = [{"image": f"images/distilled_{k:05d}.jpg",
             "prompt": qc.load_prompts(prompt_set)[idx]}
            for k, (prompt_set, idx) in enumerate(tiles)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

    # The map back to the base-model figures: which (prompt set, csv row) each generated
    # index came from. Without this, img_0000000042.png is unattributable.
    index = [{"row": k, "image": f"img_{k:010d}.png", "prompt_set": s, "prompt_row": i,
              "source_csv": qc.PROMPT_SETS[s], "prompt": rows[k]["prompt"]}
             for k, (s, i) in enumerate(tiles)]
    idx_path = os.path.join(os.path.dirname(path), "tile_index.csv")
    pd.DataFrame(index).to_csv(idx_path, index=False)
    print(f"tile index -> {idx_path}")
    return [r["prompt"] for r in rows]


def mask_tag(mask_config=MASK_CONFIG, center_seed=MASK_CENTER_SEED, random_center=True):
    shape, radius = mask_config.split(":")
    tag = f"{shape}_r{radius}"
    return f"{tag}_randcenter{center_seed}" if random_center else tag


def method_dir(out_root, method, tag):
    return (os.path.join(out_root, "high_res") if method == "high_res"
            else os.path.join(out_root, tag, method))


def run_eval_sh(args, csv_path, n_prompts):
    """Invoke eval.sh once for all five arms. Every generation flag comes from eval.sh's
    own `distilled` profile; only the prompt set, mask config and output root are ours."""
    env = dict(os.environ,
               MODEL="distilled",
               METHODS=" ".join(COLUMNS),
               MASK_CONFIGS=args.mask_config,
               PROMPTS=csv_path,
               NUM_PROMPTS=str(n_prompts),
               RANDOM_MASK_CENTER="true" if args.random_center else "false",
               MASK_CENTER_SEED=str(args.mask_center_seed),
               GPUS=args.gpus,
               OUT_ROOT=args.out_root)
    if args.ralu_n:
        env["RALU_N"] = args.ralu_n
    if args.model_base_path:
        # /miele is an NFS mount that has been observed at ~4 MB/s under load, and
        # safetensors mmaps its weights, so a cold klein-4B loads by 4 KB page faults and
        # effectively never finishes. Point the loader at a local copy instead; the path
        # must contain <org>/<model> since model_id resolves beneath it.
        env["DIFFSYNTH_MODEL_BASE_PATH"] = args.model_base_path
    print(f"$ MODEL=distilled MASK_CONFIGS={args.mask_config} GPUS='{args.gpus}' "
          f"OUT_ROOT={args.out_root} {EVAL_SH} all")
    return subprocess.call(["bash", EVAL_SH, "all"], env=env, cwd=qc.EVAL_DIR)


def read_centers(out_root, tag):
    """{row index: (cx, cy)} from the manifests eval.sh writes. Every arm records the same
    centers, so the first method that has manifests answers for all of them."""
    import pandas as pd
    for method in COLUMNS:
        # high_res is mask-independent, so inference.py records no center for it; keep
        # looking rather than concluding the run has no centers at all.
        if method == "high_res":
            continue
        files = sorted(glob.glob(os.path.join(method_dir(out_root, method, tag),
                                              "metadata_*.csv")))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if "center_x" not in df.columns:
            continue
        return {int(r["image"].split("_")[1].split(".")[0]): (r["center_x"], r["center_y"])
                for _, r in df.iterrows()}
    print("WARNING: no center_x/center_y found; the foveal outline will not be drawn")
    return {}


def load_tile(out_root, method, tag, row, centers, radius, outline):
    """One cell, with the foveal boundary drawn on the mixed-resolution arms."""
    from inference import create_foveation_mask_full_res, draw_foveation_outline

    path = os.path.join(method_dir(out_root, method, tag), f"img_{row:010d}.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = np.asarray(Image.open(path).convert("RGB")).copy()
    if outline and method != "high_res" and row in centers:
        mask = create_foveation_mask_full_res(qc.HEIGHT, qc.WIDTH, center=centers[row],
                                              r=radius, shape=qc.MASK_SHAPE)
        draw_foveation_outline(img, mask, qc.HEIGHT, qc.WIDTH,
                               outline_width_frac=qc.OUTLINE_WIDTH_FRAC,
                               color=qc.OUTLINE_COLOR)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "generate", "figure", "prompts"])
    ap.add_argument("--gpus", default="6 9")
    ap.add_argument("--tiles", default="sampled", choices=["sampled", "all"],
                    help="sampled = the 9 rows the supp figure shows; all = every tile "
                         "used by baselines 1/2/3/101 (+ the main figure), for re-picking "
                         "rows later without another GPU run")
    ap.add_argument("--figures", nargs="+", default=ALL_FIGURES,
                    help="source figures for --tiles all")
    ap.add_argument("--out_root", default=None,
                    help="defaults to the per-mode root (sampled and all must not share "
                         "one: image indices are csv positions)")
    ap.add_argument("--out", default=OUT_FIG)
    ap.add_argument("--rows", type=int, default=N_ROWS)
    ap.add_argument("--sample_seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--mask_config", default=MASK_CONFIG, help="shape:radius, eval.sh semantics")
    ap.add_argument("--mask_center_seed", type=int, default=MASK_CENTER_SEED)
    ap.add_argument("--random_center", type=lambda s: str(s).lower() != "false", default=True)
    ap.add_argument("--ralu_n", default=None,
                    help="explicit RALU per-stage steps, e.g. '1 2 1'. Unset = eval.sh's "
                         "compute match, which at 4 steps floors at N=[1,1,1]")
    ap.add_argument("--model_base_path", default=MODEL_BASE_PATH,
                    help="overrides DIFFSYNTH_MODEL_BASE_PATH for the eval.sh run; must "
                         "contain <org>/<model>. Empty = eval.sh's default (/miele)")
    ap.add_argument("--downsample", type=int, default=DOWNSAMPLE)
    ap.add_argument("--pad_px", type=int, default=20)
    ap.add_argument("--font_frac", type=float, default=qc.FONT_FRAC)
    ap.add_argument("--no_outline", action="store_true",
                    help="skip the foveal boundary the other supp figures carry")
    args = ap.parse_args()

    qc.setup_env()
    if args.tiles == "all":
        tiles = all_tiles(args.figures)
        args.out_root = args.out_root or OUT_ROOT_ALL
    else:
        tiles = sample_tiles(args.sample_seed, args.rows)
        args.out_root = args.out_root or OUT_ROOT
    csv_path = os.path.join(args.out_root, "_prompts_distilled_qual.csv")
    prompts = write_prompt_csv(tiles, csv_path)
    tag = mask_tag(args.mask_config, args.mask_center_seed, args.random_center)
    radius = float(args.mask_config.split(":")[1])

    if args.tiles == "all":
        print(f"{len(tiles)} tiles (every row of baseline_{{{','.join(args.figures)}}} "
              f"+ the main figure) x 5 arms = {len(tiles) * 5} images -> {csv_path}")
    else:
        print(f"{len(tiles)} prompts sampled (seed {args.sample_seed}) from the "
              f"baseline_{{{','.join(SOURCE_FIGURES)}}} tiles -> {csv_path}")
    for k, ((prompt_set, idx), p) in enumerate(zip(tiles, prompts)):
        print(f"  row {k}: set {prompt_set} row {idx:3d}  {p[:78]}")
    if args.stage == "prompts":
        return 0

    rc = 0
    if args.stage in ("all", "generate"):
        rc = run_eval_sh(args, csv_path, len(tiles))
        if rc:
            print("eval.sh reported a failure; see its logs", file=sys.stderr)

    if args.stage in ("all", "figure"):
        centers = read_centers(args.out_root, tag)
        rows = [[load_tile(args.out_root, m, tag, k, centers, radius,
                           not args.no_outline)
                 for m in COLUMNS] for k in range(len(tiles))]
        qc.compose([([qc.label_for(m) for m in COLUMNS], rows)],
                   args.out, args.downsample, args.pad_px, font_frac=args.font_frac)
    return rc


if __name__ == "__main__":
    sys.exit(main())
