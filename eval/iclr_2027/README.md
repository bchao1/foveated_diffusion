# `eval/iclr_2027` — qualitative figure generation

Four scripts, one shared module, one figure each. Every script does both halves of the
job — regenerate the arms it needs on GPU, then compose the PNG — so a figure in
`paper_figures/` is reproducible from one command with no manual steps in between.

| Script | Output | Layout |
|---|---|---|
| `make_main_qual.py` | `paper_figures/baseline_comparison_randomized{,_3row}.png` | 2 **and** 3 rows x 6 cols |
| `make_supp_101.py` | `paper_figures/supp/baseline_101_{a,b}.png` | 2 prompt blocks of 2x3 |
| `make_supp_123.py` | `paper_figures/supp/baseline_{1,2,3}_{a,b}.png` | 9 rows x 5 cols, six figures |
| `make_supp_distilled.py` | `paper_figures/supp/baseline_distilled.png` | 9 rows x 5 cols |

`qual_common.py` holds everything shared: the ToMe/RALU inference configuration, the
mask replay, the tokenization renderer, and the figure composer. The per-figure scripts
are thin — they name their tiles, their columns and their downsample factor, then call
`qc.main_flow`.

```bash
# imp_tome is the only env with the ToMe/RALU dependencies on the path
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python

$PY make_main_qual.py                     # regen ToMe/RALU on --gpus, then compose
$PY make_main_qual.py --stage figure       # compose only, no GPU, no model
$PY make_main_qual.py --stage verify       # mask replay + diff vs the old images, no GPU
$PY make_main_qual.py --stage regen        # regenerate only
$PY make_main_qual.py --gpus "2 3 4"       # default is "6 9"
$PY make_main_qual.py --variants 3row      # one variant only; default builds both
```

`--stage figure` is the one to reach for after any styling change: it reads only saved
PNGs and re-derives the masks on CPU, so it takes seconds and touches no GPU. Re-check
GPU occupancy before `--stage regen` or `--stage all` — this box is shared.

## What is regenerated and what is frozen

Only `tome` and `ralu` are regenerated (`REGEN_ARMS`). `high_res`, `naive` and `ours` are
read from the frozen `user_study` run under
`results/eccv2026_final_v2/baseline_comparison_randomized_{1,2,3}/<idx>/`, and are never
re-run — the published figures' identity depends on them. Regenerated images land in
`results/iclr_2027_qual/` (never on top of the old `_extra` tree), so old and new can be
diffed; `diff_against_old` prints that comparison during `--stage verify`.

`make_supp_distilled.py` is the exception: nothing is frozen there, all five arms are
generated, and generation is delegated wholesale to `eval/eval.sh` with
`MODEL=distilled` rather than reimplemented.

The ToMe/RALU columns run on the **base** model with **no LoRA**, matching the frozen
columns beside them. This deliberately differs from `eval.sh`, which gives every
training-free arm the no-fov LoRA and takes its mask from `MASK_CONFIGS` +
`foveation_center_for_index`. Neither is available here without also regenerating
high_res/naive/ours. Everything else — 50 steps, cfg 4.0, seed 42, `merge_mlp=true`,
RALU 3-stage level 4 compute-matched to ours@50 at `z_max` 6.5 — is what `eval.sh`'s
`eval_tome` / `eval_ralu` pass today.

## Tile identity

Tiles are `(prompt_set, csv_row)` pairs; the folder index equals the CSV row index, and
each of the three prompt sets is exactly 100 rows.

- `make_main_qual.py` hard-codes `TILES` in the file.
- The supplementary scripts read `eval/baseline_figure_tiles.json`, which recovered the
  published figures' composition by pixel-matching against the frozen `user_study`
  tiles. That recovery was necessary because `make_supp_figures.py` shuffled with an
  unseeded `np.random.shuffle`, so the assignment is not otherwise reproducible.

The per-prompt fixation point is **not** stored — it is replayed from the seed.
`run_user_study` draws it from `random.uniform(-0.3, 0.3)` on the `random` module seeded
with 42, and those draws are the RNG's first consumers, so the replay is exact.
`verify_centers()` checks the replay against each tile's saved `fixation_point.png` and
raises if the worst error exceeds 1e-3 (dot rasterization alone is ~1e-4). It runs on
every stage, including `figure`, so a drifted mask can never reach a figure silently.

## The main figure's two row counts

`make_main_qual.py` emits **both** row counts, and builds both by default:

```python
VARIANTS = {
    "2row": ([DOG, MAN],        "baseline_comparison_randomized.png"),
    "3row": ([HORSE, DOG, MAN], "baseline_comparison_randomized_3row.png"),
}
```

The published figure had all three rows; the 2-row cut drops the racehorse. Both are kept
because that is a page-budget decision and can flip late, and the choice costs nothing to
defer: `tome`/`ralu` are regenerated for `ALL_TILES` (the union), and center verification
covers the union too, so either variant composes with `--stage figure` and no GPU.

`tiles` is the **only** per-variant input — `build_variant()` fixes everything that
defines the format (columns, wide gutter, divider, downsample, padding, label band), so
the two figures cannot drift apart in style. They are byte-identical outside the extra
row: same width (3157 px), same label band, and the dog+man block is pixel-for-pixel the
same in both. If you add a knob, put it in `build_variant`'s signature with one value at
the call site, not in `VARIANTS`.

The `high_res` reference does not consume the token layout while the four foveated
methods do, so it is set off by a wide gutter (`WIDE_GUTTER_PX = 90`) with a vertical
divider drawn through it, rather than sitting flush against the tokenization column.

## The tokenization column

Column 2 of the main figure is synthesized, not loaded — `tokenization_mask_rgb()`
renders the HR/LR token grid from the same replayed foveation mask the methods use.

The shading uses LoT's token-size gray palette
`{0: 248, 1: 220, 2: 190, 3: 155, 4: 118, 5: 78, 6: 38}`, keyed on `log2(leaf area)`,
copied verbatim from `GRAY_LEVELS` in that repo's `scripts/export_qual_figures.py`
(`/home/brianchc/LoT-george-branch`). **Light = fine, dark = coarse.** The fovea takes
`FOVEA_LEVEL = 0` (gray 248), its true level for a 1x1 token; the periphery takes
`PERIPHERY_LEVEL = 4` (gray 118), which is **darker than its own area implies** — a 2x2
token at `lr_factor=2` is area 4, level 2, gray 190.

Notes on getting the match right:

- It used to use LoT's *warm* `_TOKEN_RAMP` from `precompute/flux/vrs_image.py`, where
  the convention is inverted (dark = fine). The two palettes disagree on direction, so
  matching the grays meant flipping which region is dark.
- `border_px` is 4, which puts the grid lines at **2 px in the composed figure** (the main
  figure downsamples by 2). That is LoT's line weight: its `render_layout` outlines every
  leaf at width 1 and adjacent leaves double up, so its interior lines measure 2 px at any
  cell size — 420 px panels over 6.5 px cells included. Measure before changing this; the
  intuition that a thinner line matches a smaller cell is wrong for LoT, whose line weight
  is absolute.
- That line weight is what forces `PERIPHERY_LEVEL` off the absolute mapping, and it is
  the one place this panel deviates from LoT. At 2 px on an 8 px foveal cell the fovea
  carries ~43% black ink against the periphery's ~24%, which drags the fovea's *mean*
  luminance to 142 against the periphery's 147 — the fill difference is cancelled outright
  and the panel has no tonal contrast, so a reader has to resolve individual cells to see
  where the budget went. Level 4 puts the periphery at mean 91 against the fovea's 141.
  LoT gets away with 190 because its panels are 420 px for the whole image, so its coarse
  leaves are 26 px cells at ~14% ink where ours are 16 px at 24%.
- Level 3 (gray 155 → mean 108) is the softer choice if the panel ever reads too heavy
  next to the photographic tiles. Thinning `border_px` instead also separates the two
  regions, but then the lines are half LoT's weight — that trade was made the other way.

## Composition

`qc.compose(blocks, out, downsample, ...)` stacks labelled blocks vertically. Notes on
the knobs the figures actually use:

- `blocks` is a list of `(labels, rows)`. A cell is an `(arm, prompt_set, idx)` triple,
  the string `"blank"`, or a ready ndarray. `labels=None` suppresses the band entirely;
  `make_supp_101.py` uses one band per block because its two rows have different columns.
- `downsample` sets the tile size: 1 = 1024 px (supp 101), 2 = 512 px (main), 4 = 256 px
  (supp 1-3). These match the published figures they supersede.
- `gaps_px` gives per-gutter widths instead of a uniform `pad_px`; `rules` names gutters
  to draw a divider through, `rule_span="full"` extending it through the label band.
- `font_frac` is the label height as a fraction of **one tile's** height, 1/8 by default.
  Labels are CMU Serif Bold, falling back to DejaVu Serif Bold.

## Gotchas

- `--stage figure` still imports `inference` and RALU to rebuild masks, so it needs
  `setup_env()`'s path surgery — run it via the scripts, not by importing the composer
  directly. `setup_env` forces this repo's `DiffSynth-Studio`, because the `imp_tome`
  env otherwise resolves `diffsynth` to a different checkout without the foveated
  pipeline.
- The figure stage fails loudly on missing inputs (`missing()` lists every absent
  arm/tile) rather than composing a partial figure.
- `cell == "blank"` on an ndarray is an elementwise compare, not a test — `load_cell`
  checks for ndarray first for this reason.
