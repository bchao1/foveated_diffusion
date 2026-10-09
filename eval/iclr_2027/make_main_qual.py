#!/usr/bin/env python
"""Main qualitative baseline figure — two row counts, identical in every other respect.

Six columns either way:

    Full resolution | Tokenization | Token merging | RALU | Naive mixed-res. | Ours

    VARIANTS["2row"]  ->  baseline_comparison_randomized.png        dog, man
    VARIANTS["3row"]  ->  baseline_comparison_randomized_3row.png   horse, dog, man

Both are built by default, from one shared `build_variant`, so the two cannot drift in
format: same columns, same wide gutter and divider, same downsample, same label band,
same tokenization palette. They differ only in which tiles are listed.

The tiles are prompt set 1 (prompts/claude_prompts.csv) rows 8, 10 and 3, recovered by
pixel-matching the published PNG against the frozen `user_study` tiles. The published
figure had all three; the 2-row cut drops the racehorse (row 8) for the camera-ready.
Both row counts are kept because that choice is a page-budget one, so it can flip late.

`high_res`, `naive` and `ours` are read from that frozen run; `tome` and `ralu` are
regenerated here — for the union of both variants' tiles, so switching between them never
needs a GPU. The tokenization column is synthesized from the same per-prompt foveation
mask, shaded with LoT's token-size grays (light = fine foveal tokens, dark = coarse
peripheral tokens) at LoT's line weight.

    ./make_main_qual.py                     # regen ToMe/RALU on --gpus, then compose both
    ./make_main_qual.py --stage figure      # compose both, no GPU
    ./make_main_qual.py --variants 3row     # just the 3-row figure
    ./make_main_qual.py --stage verify      # mask replay + diff vs the old images, no GPU
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_common as qc  # noqa: E402

# (prompt_set, csv row). Reading order = figure rows.
HORSE, DOG, MAN = (1, 8), (1, 10), (1, 3)

# variant -> (tiles, output filename). Every other aspect of the format is shared.
VARIANTS = {
    "2row": ([DOG, MAN], "baseline_comparison_randomized.png"),
    "3row": ([HORSE, DOG, MAN], "baseline_comparison_randomized_3row.png"),
}
# Regen and center-verification cover the union, so either variant composes without a GPU.
ALL_TILES = [HORSE, DOG, MAN]

COLUMNS = ["high_res", "tokenization", "tome", "ralu", "naive", "ours"]

# The full-resolution reference does NOT consume the token layout — the four foveated
# methods do. Sitting flush against the mask column it reads as though it did, so the
# reference is set off by a wide gutter with a vertical divider running the height of the
# image rows.
WIDE_GUTTER_PX = 90          # between the reference and the rest, at full resolution
RULE_GUTTER = 0              # gutter index the divider is drawn through

DOWNSAMPLE = 2          # 512 px per tile, matching the published figure


def build_variant(tiles, out_path, ds, pad_px, wide_gutter):
    """Compose one variant. The only per-variant input is `tiles`; everything that defines
    the format is fixed here, which is what keeps the two figures identical in style."""
    rows = [[(arm, s, i) for arm in COLUMNS] for s, i in tiles]
    gaps = [wide_gutter] + [pad_px] * (len(COLUMNS) - 2)
    qc.compose([([qc.label_for(a) for a in COLUMNS], rows)],
               out_path, ds, pad_px, gaps_px=gaps, rules=[RULE_GUTTER],
               rule_span="full")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    qc.add_common_args(ap)
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of %s" % ", ".join(VARIANTS))
    ap.add_argument("--out_dir", default=qc.FIGURES)
    ap.add_argument("--wide_gutter", type=int, default=WIDE_GUTTER_PX,
                    help="gutter between the full-resolution reference and the group")
    args = ap.parse_args()
    ds = args.downsample or DOWNSAMPLE

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")

    def build():
        for name in names:
            tiles, filename = VARIANTS[name]
            build_variant(tiles, os.path.join(args.out_dir, filename), ds,
                          args.pad_px, args.wide_gutter)

    return qc.main_flow(args, os.path.abspath(__file__), ALL_TILES, build)


if __name__ == "__main__":
    sys.exit(main())
