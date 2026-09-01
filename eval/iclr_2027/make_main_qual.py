#!/usr/bin/env python
"""Main qualitative baseline figure — paper_figures/baseline_comparison_randomized.png.

Three prompts (racehorse, German Shepherd, elderly man) x six columns:

    Full resolution | Tokenization | Token merging | RALU | Naive mixed-res. | Ours

The rows are the same three examples as the published figure: prompt set 1
(prompts/claude_prompts.csv) rows 8, 10 and 3, recovered by pixel-matching the published
PNG against the frozen `user_study` tiles. `high_res`, `naive` and `ours` are read from
that frozen run; `tome` and `ralu` are regenerated here. The tokenization column is
synthesized from the same per-prompt foveation mask, colored with LoT's absolute
token-size palette (dark = fine foveal tokens, light = coarse peripheral tokens).

    ./make_main_qual.py                     # regen ToMe/RALU on --gpus, then compose
    ./make_main_qual.py --stage figure      # compose only, no GPU
    ./make_main_qual.py --stage verify      # mask replay + diff vs the old images, no GPU
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_common as qc  # noqa: E402

# (prompt_set, csv row). Reading order = figure rows.
TILES = [(1, 8), (1, 10), (1, 3)]

COLUMNS = ["high_res", "tokenization", "tome", "ralu", "naive", "ours"]

# The full-resolution reference does NOT consume the token layout — the four foveated
# methods do. Sitting flush against the mask column it reads as though it did, so the
# reference is set off by a wide gutter with a vertical divider running the height of the
# image rows.
WIDE_GUTTER_PX = 90          # between the reference and the rest, at full resolution
RULE_GUTTER = 0              # gutter index the divider is drawn through

OUT = os.path.join(qc.FIGURES, "baseline_comparison_randomized.png")
DOWNSAMPLE = 2          # 512 px per tile, matching the published figure


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    qc.add_common_args(ap)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--wide_gutter", type=int, default=WIDE_GUTTER_PX,
                    help="gutter between the full-resolution reference and the group")
    args = ap.parse_args()
    ds = args.downsample or DOWNSAMPLE

    def build():
        rows = [[(arm, s, i) for arm in COLUMNS] for s, i in TILES]
        gaps = [args.wide_gutter] + [args.pad_px] * (len(COLUMNS) - 2)
        qc.compose([([qc.label_for(a) for a in COLUMNS], rows)],
                   args.out, ds, args.pad_px, gaps_px=gaps, rules=[RULE_GUTTER],
                   rule_span="full")

    return qc.main_flow(args, os.path.abspath(__file__), TILES, build)


if __name__ == "__main__":
    sys.exit(main())
