#!/usr/bin/env python
"""Supplementary hero baseline figures — paper_figures/supp/baseline_101_{a,b}.png.

Supersedes the single 4-row `baseline_101_5col.png`. Its four tiles are split across two
figures, two prompts each, and every prompt becomes a 2x3 block:

    | Full resolution | Token merging | RALU |
    |                 | Naive mixed-res. | Ours |

so the five arms read as "reference on top left, the four foveated methods around it"
rather than one long row. Two blocks stack vertically per figure, giving 4 rows x 3 cols.
Each block carries its own two label bands, since the columns differ between its rows.

Tile identity comes from `eval/baseline_figure_tiles.json` ("101"), which recovered the
published figure's composition by pixel-matching. `high_res`, `naive` and `ours` are the
frozen `user_study` images; `tome` and `ralu` are regenerated here.

    ./make_supp_101.py                     # regen ToMe/RALU on --gpus, then compose
    ./make_supp_101.py --stage figure      # compose only, no GPU
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_common as qc  # noqa: E402

TILES_JSON = os.path.join(qc.EVAL_DIR, "baseline_figure_tiles.json")

TOP = ["high_res", "tome", "ralu"]
BOTTOM = ["blank", "naive", "ours"]

OUT_DIR = os.path.join(qc.FIGURES, "supp")
DOWNSAMPLE = 1          # full resolution, matching the published baseline_101_5col.png
# Half the shared label size: each prompt block carries two bands, so at the common
# 1/8-of-a-tile size the labels crowd the figure.
FONT_FRAC = qc.FONT_FRAC / 2


def load_tiles():
    with open(TILES_JSON) as f:
        return [tuple(t) for t in json.load(f)["101"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    qc.add_common_args(ap)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--prefix", default="baseline_101")
    ap.add_argument("--font_frac", type=float, default=FONT_FRAC,
                    help="label height as a fraction of one tile's height")
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

    tiles = load_tiles()
    assert len(tiles) == 4, f"expected 4 tiles for figure 101, got {len(tiles)}"
    halves = [tiles[:2], tiles[2:]]

    def block(prompt_set, idx):
        """One prompt: two labelled rows, |HR|ToMe|RALU| over |blank|Naive|Ours|."""
        top = [(a, prompt_set, idx) for a in TOP]
        bottom = [c if c == "blank" else (c, prompt_set, idx) for c in BOTTOM]
        return [([qc.label_for(a) for a in TOP], [top]),
                (["" if c == "blank" else qc.label_for(c) for c in BOTTOM], [bottom])]

    qc.setup_env()          # the distilled path skips main_flow, which sets this up
    resolver = qc.DistilledTiles() if args.source == "distilled" else None

    def build():
        for name, half in zip("ab", halves):
            blocks = [b for s, i in half for b in block(s, i)]
            qc.compose(blocks,
                       os.path.join(args.out_dir, f"{args.prefix}_{name}{suffix}.png"),
                       ds, args.pad_px, font_frac=args.font_frac, resolver=resolver)

    if resolver is not None:
        # Every arm already exists in the distilled tree; there is nothing to regenerate
        # and no user_study fixation point to verify against.
        gaps = qc.missing(TOP + BOTTOM[1:], tiles, resolver)
        if gaps:
            raise SystemExit("missing distilled inputs:\n  " +
                             "\n  ".join(f"{a} set{s} {i:05d}" for a, s, i in gaps))
        build()
        return 0
    return qc.main_flow(args, os.path.abspath(__file__), tiles, build)


if __name__ == "__main__":
    sys.exit(main())
