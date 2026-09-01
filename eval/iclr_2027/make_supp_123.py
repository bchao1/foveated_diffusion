#!/usr/bin/env python
"""Supplementary baseline figures 1-3 — paper_figures/supp/baseline_{1,2,3}_{a,b}.png.

Supersedes the three 18-row `baseline_{1,2,3}_5col.png`. Each figure's tiles are split in
half, giving six figures of 9 rows x 5 columns:

    Full resolution | Token merging | RALU | Naive mixed-res. | Ours

An odd tile count would drop its last tile so the halves are equal; all three sets have
18, so nothing is dropped. Tile identity and order come from
`eval/baseline_figure_tiles.json`, which recovered the published figures' composition by
pixel-matching (make_supp_figures.py shuffles with an unseeded np.random.shuffle, so the
assignment is not otherwise reproducible). `high_res`, `naive` and `ours` are the frozen
`user_study` images; `tome` and `ralu` are regenerated here.

    ./make_supp_123.py                     # regen ToMe/RALU on --gpus, then compose
    ./make_supp_123.py --figures 1         # one source figure only
    ./make_supp_123.py --stage figure      # compose only, no GPU
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_common as qc  # noqa: E402

TILES_JSON = os.path.join(qc.EVAL_DIR, "baseline_figure_tiles.json")

COLUMNS = ["high_res", "tome", "ralu", "naive", "ours"]

OUT_DIR = os.path.join(qc.FIGURES, "supp")
DOWNSAMPLE = 4          # 256 px per tile, matching the published baseline_{1,2,3}_5col.png


def load_tiles(names):
    with open(TILES_JSON) as f:
        raw = json.load(f)
    return {n: [tuple(t) for t in raw[n]] for n in names}


def split_in_half(tiles):
    """Two equal halves; an odd count drops its last tile."""
    n = len(tiles) // 2
    if len(tiles) % 2:
        print(f"  odd tile count ({len(tiles)}): dropping the last one so the halves match")
    return tiles[:n], tiles[n:2 * n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    qc.add_common_args(ap)
    ap.add_argument("--figures", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--source", default="base", choices=["base", "distilled"],
                    help="base = the frozen user_study columns + regenerated ToMe/RALU; "
                         "distilled = every arm from the klein-4B tree written by "
                         "make_supp_distilled.py --tiles all")
    ap.add_argument("--suffix", default=None,
                    help="output filename suffix (default: '' for base, "
                         "'_distilled' for distilled)")
    args = ap.parse_args()
    ds = args.downsample or DOWNSAMPLE
    suffix = args.suffix if args.suffix is not None else (
        "_distilled" if args.source == "distilled" else "")

    by_figure = load_tiles(args.figures)
    halves = {name: split_in_half(t) for name, t in by_figure.items()}
    tiles = [t for name in args.figures for half in halves[name] for t in half]

    qc.setup_env()          # the distilled path skips main_flow, which sets this up
    resolver = qc.DistilledTiles() if args.source == "distilled" else None

    def build():
        labels = [qc.label_for(a) for a in COLUMNS]
        for name in args.figures:
            for half_id, half in zip("ab", halves[name]):
                rows = [[(arm, s, i) for arm in COLUMNS] for s, i in half]
                qc.compose(
                    [(labels, rows)],
                    os.path.join(args.out_dir,
                                 f"baseline_{name}_{half_id}{suffix}.png"),
                    ds, args.pad_px, resolver=resolver)

    if resolver is not None:
        # Every arm already exists in the distilled tree; there is nothing to regenerate
        # and no user_study fixation point to verify against.
        gaps = qc.missing(COLUMNS, tiles, resolver)
        if gaps:
            raise SystemExit("missing distilled inputs:\n  " +
                             "\n  ".join(f"{a} set{s} {i:05d}" for a, s, i in gaps))
        build()
        return 0
    return qc.main_flow(args, os.path.abspath(__file__), tiles, build)


if __name__ == "__main__":
    sys.exit(main())
