#!/bin/bash
# Isolated latency benchmark across all arms at a matched token budget.
#
# Runs arms SEQUENTIALLY on a single GPU — concurrent arms contaminate the timing.
# One process per arm so only one pipeline is resident at a time.
#
#   ./bench_all.sh                      # defaults: 50 steps, r=0.3, 3 reps, GPU 0
#   STEPS=28 RADIUS=0.5 ./bench_all.sh
#   GPU=3 REPEATS=5 BENCH_OUT=../../results/bench_tome ./bench_all.sh
set -euo pipefail

GPU=${GPU:-0}
STEPS=${STEPS:-50}
RADIUS=${RADIUS:-0.3}
MASK=${MASK:-circular}
LR_FACTOR=${LR_FACTOR:-2}
SEED=${SEED:-42}
CFG=${CFG:-4.0}
WARMUP=${WARMUP:-1}
REPEATS=${REPEATS:-3}
ARMS=${ARMS:-"high_res tome_attn tome_mlp ours naive"}
BENCH_OUT=${BENCH_OUT:-../../results/bench_tome_vs_ours}

PY=${PY:-/home/brianchc/miniconda3/envs/imp_tome/bin/python}
REPO=${REPO:-/home/brianchc/foveated_diffusion-private}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/miele/brian/modelscope}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-True}
export PYTHONPATH=${REPO}/DiffSynth-Studio

cd "$(dirname "$0")"
mkdir -p "${BENCH_OUT}"

echo "GPU ${GPU} | ${STEPS} steps | ${MASK} r=${RADIUS} | lr_factor=${LR_FACTOR} | ${WARMUP} warmup + ${REPEATS} reps"
echo "arms (sequential): ${ARMS}"

for arm in ${ARMS}; do
    echo "--- ${arm} ---"
    CUDA_VISIBLE_DEVICES=${GPU} $PY bench_tome_vs_ours.py \
        --arm "${arm}" \
        --steps "${STEPS}" --cfg "${CFG}" --seed "${SEED}" \
        --mask "${MASK}" --radius "${RADIUS}" --lr_factor "${LR_FACTOR}" \
        --warmup "${WARMUP}" --repeats "${REPEATS}" \
        --out "${BENCH_OUT}/${arm}.json" > "${BENCH_OUT}/${arm}.log" 2>&1
    tail -1 "${BENCH_OUT}/${arm}.log" > /dev/null || true
    grep -h "rep .*dit" "${BENCH_OUT}/${arm}.log" || echo "  (see ${BENCH_OUT}/${arm}.log)"
done

$PY bench_tome_vs_ours.py --report "${BENCH_OUT}" | tee "${BENCH_OUT}/report.txt"
