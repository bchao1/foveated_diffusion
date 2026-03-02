import csv
import argparse
import os
import glob
import json
import math
from collections import defaultdict

PAIRS = [
    ("naive", "high_res"),
    ("high_res", "ours"),
    ("naive", "ours"),
]


def canonical_pair(a, b):
    for p in PAIRS:
        if set(p) == {a, b}:
            return p
    return tuple(sorted([a, b]))


def parse(csv_path):
    # counters[pair][question][method] = count
    counters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img1, img2 = row['img1'], row['img2']
            response = int(row['response'])
            question = row['question']
            chosen = img1 if response == 1 else img2
            pair = canonical_pair(img1, img2)
            counters[pair]['overall'][chosen] += 1
            counters[pair][question][chosen] += 1

    return counters


def empty_counters():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def merge_counters(dst, src):
    for pair, questions in src.items():
        for q, methods in questions.items():
            for method, count in methods.items():
                dst[pair][q][method] += count


def report(counters):
    for pair in PAIRS:
        m1, m2 = pair
        print(f"\n{'='*50}")
        print(f"Pair: {m1} vs {m2}")
        print(f"{'='*50}")

        data = counters[pair]['overall']
        total = data[m1] + data[m2]
        if total > 0:
            print(f"  Overall ({total} trials):")
            print(f"    {m1}: {data[m1]}/{total} = {data[m1]/total:.1%}")
            print(f"    {m2}: {data[m2]}/{total} = {data[m2]/total:.1%}")

        for q in ['quality', 'errors', 'natural']:
            data = counters[pair][q]
            total = data[m1] + data[m2]
            if total > 0:
                print(f"  {q.capitalize()} ({total} trials):")
                print(f"    {m1}: {data[m1]}/{total} = {data[m1]/total:.1%}")
                print(f"    {m2}: {data[m2]}/{total} = {data[m2]/total:.1%}")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def build_overall_summary(csv_files):
    # Collect per-file overall win rates and then average across files.
    per_pair_values = defaultdict(lambda: defaultdict(list))
    files_used = []

    for csv_file in csv_files:
        counters = parse(csv_file)
        file_used = False
        for pair in PAIRS:
            m1, m2 = pair
            data = counters[pair]["overall"]
            total = data[m1] + data[m2]
            if total > 0:
                per_pair_values[pair][m1].append(data[m1] / total)
                per_pair_values[pair][m2].append(data[m2] / total)
                file_used = True
        if file_used:
            files_used.append(os.path.basename(csv_file))

    summary = {
        "files_used_count": len(files_used),
        "files_used": files_used,
        "pairwise_overall": {},
    }

    output_pairs = [
        ("naive_vs_ours", ("naive", "ours")),
        ("ours_vs_high_res", ("ours", "high_res")),
        ("naive_vs_high_res", ("naive", "high_res")),
    ]

    for key, pair in output_pairs:
        m1, m2 = pair
        canonical = canonical_pair(m1, m2)
        v1 = per_pair_values[canonical][m1]
        v2 = per_pair_values[canonical][m2]

        m1_mean = mean(v1)
        m2_mean = mean(v2)
        m1_std = sample_std(v1)
        m2_std = sample_std(v2)
        n = max(len(v1), len(v2))

        m1_sem = m1_std / math.sqrt(n) if n > 0 else 0.0
        m2_sem = m2_std / math.sqrt(n) if n > 0 else 0.0
        ci95_multiplier = 1.96

        summary["pairwise_overall"][key] = {
            "pair": [m1, m2],
            "n_files": n,
            "mean_preference": {
                m1: m1_mean,
                m2: m2_mean,
            },
            "error_bar_length": {
                "type": "sem",
                m1: m1_sem,
                m2: m2_sem,
            },
            "ci95_error_bar_length": {
                "type": "95%_confidence_interval_half_width",
                m1: ci95_multiplier * m1_sem,
                m2: ci95_multiplier * m2_sem,
            },
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Parse user study pairwise preference results."
    )
    parser.add_argument(
        "directory",
        help="Directory containing CSV files to aggregate",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output JSON path. Defaults to <directory>/overall_pairwise_metrics.json",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        raise ValueError(f"Expected a directory, got: {args.directory}")

    csv_files = sorted(glob.glob(os.path.join(args.directory, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {args.directory}")
        return

    print(
        f"Aggregating {len(csv_files)} CSV file(s): "
        f"{[os.path.basename(f) for f in csv_files]}"
    )
    counters = empty_counters()
    for csv_file in csv_files:
        merge_counters(counters, parse(csv_file))

    report(counters)

    json_out = args.json_out or os.path.join(
        args.directory, "overall_pairwise_metrics.json"
    )
    summary = build_overall_summary(csv_files)
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nSaved overall pairwise JSON metrics to: {json_out}")


if __name__ == '__main__':
    main()
