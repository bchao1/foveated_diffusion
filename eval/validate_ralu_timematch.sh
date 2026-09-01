#!/usr/bin/env bash
# Validate the time-matched RALU schedules on an idle box.
#
# For each radius, run `ours` and `ralu` with the candidate N back to back under identical
# conditions, then compare mean (dit + vae) per image. Measuring both arms in the same
# window is the point: comparing a fresh RALU number against the 10k means collected days
# ago on a busier box would fold contention into the match.
#
# Sharding matches production (5 shards / 5 GPUs) so the calibration predicts the real run.
# Resolution stays 1024x1024 -- per-image time depends on it, so shrinking it would
# invalidate the whole measurement.
set -u

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_PROMPTS=${N_PROMPTS:-50}
GPUS_USE=${GPUS_USE:-"1 2 3 4 5"}
OUT=${OUT:-/local/brian/foveated_diffusion/results/iclr_2027_timematch_cal}

# radius | candidate RALU N (solved from k = ours_time / ralu_time on the 10k means)
CONFIGS=(
  "0.1|10 11 13"
  "0.25|11 12 14"
  "0.5|14 16 18"
)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] waiting for the box to go idle (main sweep + repair pass)"
while pgrep -f "eval.sh (all|ralu|shard)" >/dev/null || pgrep -f repair_base_runA.sh >/dev/null; do
    sleep 60
done
echo "[$(ts)] idle"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

for entry in "${CONFIGS[@]}"; do
    r="${entry%%|*}"; N="${entry##*|}"
    echo "[$(ts)] ===== r=$r  candidate RALU_N=[$N] ====="
    MODEL=base RANDOM_MASK_CENTER=true NUM_PROMPTS="$N_PROMPTS" \
    MASK_CONFIGS="circular:$r" GPUS="$GPUS_USE" OUT_ROOT="$OUT" \
    "$EVAL_DIR"/eval.sh ours
    MODEL=base RANDOM_MASK_CENTER=true NUM_PROMPTS="$N_PROMPTS" \
    MASK_CONFIGS="circular:$r" RALU_N="$N" GPUS="$GPUS_USE" OUT_ROOT="$OUT" \
    "$EVAL_DIR"/eval.sh ralu
done

echo "[$(ts)] ===== RESULTS ====="
/home/brianchc/miniconda3/envs/imp_tome/bin/python - "$OUT" <<'PY'
import sys, glob, os, collections
import pandas as pd
OUT = sys.argv[1]
rows = collections.defaultdict(list)
for f in glob.glob(OUT + '/**/metadata_*.csv', recursive=True):
    arm = os.path.basename(os.path.dirname(f))
    tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
    rows[(tag, arm)].append(pd.read_csv(f))
print(f"{'radius':8s} {'ours s':>9s} {'ralu s':>9s} {'ralu/ours':>10s} {'verdict':>10s}")
for r in ('0.1', '0.25', '0.5'):
    tag = f'circular_r{r}_randcenter42'
    if (tag, 'ours') not in rows or (tag, 'ralu') not in rows:
        print(f'{r:8s} MISSING'); continue
    o = pd.concat(rows[(tag, 'ours')]); a = pd.concat(rows[(tag, 'ralu')])
    ot = (o.dit_time_s + o.vae_time_s).mean(); at = (a.dit_time_s + a.vae_time_s).mean()
    ratio = at / ot
    verdict = 'MATCHED' if abs(ratio - 1) <= 0.02 else ('RALU FAST' if ratio < 1 else 'RALU SLOW')
    print(f'{r:8s} {ot:9.3f} {at:9.3f} {ratio:10.4f} {verdict:>10s}   (n={len(o)}/{len(a)})')
print()
print('MATCHED = within 2%. Otherwise rescale N by 1/ratio and re-validate.')
PY
echo "[$(ts)] done"
