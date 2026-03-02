"""
Generate user-study pairwise preference bar plots from overall_pairwise_metrics.json.
Style is matched to runtime figure style in eval/generate_paper_figures.py.
"""

import json
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator

# Import shared style constants from the paper figures module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_paper_figures import FIGURE_RC, setup_academic_axes

# Tunable aesthetics
FIGSIZE = (8.0, 4.0)
Y_TICKS = np.arange(0, 101, 20)
Y_AXIS_MAX = 100.0
TOP_MARGIN_FRACTION = 0.85  # lower = more whitespace above axes for value labels

BAR_WIDTH = 0.08
PAIR_INNER_GAP = BAR_WIDTH  # center-to-center = bar width → bars touch (no gap)
PAIR_OUTER_GAP = 0.15  # extra gap between adjacent pairs (after second bar center)
BAR_CORNER_RADIUS_FRAC = 0.0

COLOR1 = "#549bcf"
COLOR2 = "#dc65aa"
BAR_EDGE_COLOR = "#555555"
BAR_EDGE_WIDTH = 0.6

ERROR_BAR_LINEWIDTH = 1.0
ERROR_BAR_CAPTHICK = 1.0
ERROR_BAR_CAPSIZE = 4
ERROR_BAR_COLOR = "#333333"

VALUE_LABEL_OFFSET_PT = 6.0

PAIR_LABEL_FONT_SIZE = 12
METHOD_LABEL_FONT_SIZE = 11
METHOD_LABEL_ROTATION = 25

PAIR_ORDER = [
    ("naive_vs_ours", "Naive vs Ours"),
    ("ours_vs_high_res", "Ours vs High-res"),
    ("naive_vs_high_res", "Naive vs High-res"),
]

NAME_MAP = {
    "naive": "Naive",
    "ours": "Ours",
    "high_res": "High-res",
}


def _display_name(name):
    return NAME_MAP.get(name, str(name).replace("_", " ").title())


def _extract_error_bars(pair_data):
    # Prefer CI95 if available, fallback to SEM.
    if "ci95_error_bar_length" in pair_data:
        return pair_data["ci95_error_bar_length"]
    return pair_data.get("error_bar_length", {})


def _add_rounded_bars(ax, bars):
    for bar in bars:
        x, y = bar.get_xy()
        w = bar.get_width()
        h = bar.get_height()
        fc = bar.get_facecolor()
        ec = bar.get_edgecolor()
        lw = bar.get_linewidth()
        bar.set_visible(False)
        # Stable corner radius in data-space, capped by width/height.
        corner_radius = max(0.0, min(BAR_CORNER_RADIUS_FRAC * w, 0.5 * w, 0.5 * h))
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            clip_on=True,
            antialiased=True,
            zorder=3,
        )
        ax.add_patch(patch)


def generate_user_study_figure(json_path, output_path=None):
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON metrics file not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    pairwise = data.get("pairwise_overall", {})
    missing = [k for k, _ in PAIR_ORDER if k not in pairwise]
    if missing:
        raise ValueError(f"Missing pairs in JSON: {missing}")

    with plt.rc_context(FIGURE_RC):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        setup_academic_axes(ax, grid=False)
        # Bar-chart-specific overrides (applied after the shared style)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.tick_params(axis="y", which="both", right=False)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # 1 minor tick between majors

        x_positions = []
        method_labels = []
        pair_centers = []
        pair_labels = []
        vals_all = []
        errs_all = []
        colors_all = []
        global_max = 0.0

        cursor = 0.0
        for key, pair_title in PAIR_ORDER:
            pair_data = pairwise[key]
            methods = pair_data["pair"]
            means = pair_data["mean_preference"]
            err_src = _extract_error_bars(pair_data)

            vals = np.array([100.0 * float(means[m]) for m in methods], dtype=float)
            errs = np.array([100.0 * float(err_src.get(m, 0.0)) for m in methods], dtype=float)

            if np.isclose(vals[0], vals[1]):
                colors = [COLOR2, COLOR1]
            elif vals[0] > vals[1]:
                colors = [COLOR1, COLOR2]
            else:
                colors = [COLOR2, COLOR1]

            x1 = cursor
            x2 = cursor + PAIR_INNER_GAP
            x_positions.extend([x1, x2])
            method_labels.extend([_display_name(methods[0]), _display_name(methods[1])])
            pair_centers.append((x1 + x2) / 2.0)
            pair_labels.append(pair_title)
            vals_all.extend(vals.tolist())
            errs_all.extend(errs.tolist())
            colors_all.extend(colors)
            global_max = max(global_max, float(np.max(vals + errs)))
            cursor = x2 + PAIR_OUTER_GAP

        vals_all = np.asarray(vals_all, dtype=float)
        errs_all = np.asarray(errs_all, dtype=float)
        ylim_top = Y_AXIS_MAX

        bars = ax.bar(
            x_positions,
            vals_all,
            width=BAR_WIDTH,
            color=colors_all,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )
        _add_rounded_bars(ax, bars)

        ax.errorbar(
            x_positions,
            vals_all,
            yerr=errs_all,
            fmt="none",
            ecolor=ERROR_BAR_COLOR,
            elinewidth=ERROR_BAR_LINEWIDTH,
            capsize=ERROR_BAR_CAPSIZE,
            capthick=ERROR_BAR_CAPTHICK,
            zorder=4,
        )

        for x, v, e in zip(x_positions, vals_all, errs_all):
            y_anchor = min(v + e, ylim_top)
            ax.annotate(
                f"{v:.1f}%",
                xy=(x, y_anchor),
                xytext=(0, VALUE_LABEL_OFFSET_PT),
                textcoords="offset points",
                ha="center",
                va="bottom",
                clip_on=False,
                zorder=5,
            )

        ax.set_ylabel("Preference (%)")
        ax.set_ylim(0.0, Y_AXIS_MAX)
        ax.set_yticks(Y_TICKS)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(method_labels, rotation=METHOD_LABEL_ROTATION, fontsize=METHOD_LABEL_FONT_SIZE)

        # Pair labels centered under each method pair.
        for xc, label in zip(pair_centers, pair_labels):
            ax.text(
                xc,
                -0.20,
                label,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=PAIR_LABEL_FONT_SIZE,
            )

        ax.set_xlim(min(x_positions) - BAR_WIDTH, max(x_positions) + BAR_WIDTH)
        fig.tight_layout(pad=1.0)
        fig.subplots_adjust(top=TOP_MARGIN_FRACTION)

        if output_path is None:
            output_path = os.path.join(os.path.dirname(json_path), "user_study_pairwise_bars.pdf")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        print(f"Saved: {output_path}")


def main():
    parser = ArgumentParser(description="Generate user-study pairwise bar plot figure.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="eval/user_study/overall_pairwise_metrics.json",
        help="Path to overall_pairwise_metrics.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: alongside JSON as user_study_pairwise_bars.pdf)",
    )
    args = parser.parse_args()
    generate_user_study_figure(args.json_path, args.output)


if __name__ == "__main__":
    main()
