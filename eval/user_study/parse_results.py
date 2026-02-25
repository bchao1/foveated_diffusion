import csv
import argparse
import os
import glob
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


def main():
    parser = argparse.ArgumentParser(
        description="Parse user study pairwise preference results."
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='eval/user_study/user_study_results.csv',
        help='Path to a CSV file or a directory containing CSV files',
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        csv_files = sorted(glob.glob(os.path.join(args.path, '*.csv')))
        if not csv_files:
            print(f"No CSV files found in {args.path}")
            return
        print(f"Aggregating {len(csv_files)} CSV file(s): {[os.path.basename(f) for f in csv_files]}")
        counters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for csv_file in csv_files:
            file_counters = parse(csv_file)
            for pair, questions in file_counters.items():
                for q, methods in questions.items():
                    for method, count in methods.items():
                        counters[pair][q][method] += count
    else:
        counters = parse(args.path)

    report(counters)


if __name__ == '__main__':
    main()
