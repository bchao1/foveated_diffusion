# Marlowe eval — exact launch commands

The same evaluation as `eval/eval.sh` on local GPUs, ported to Slurm following
[`marlowe_launch_rules.md`](marlowe_launch_rules.md), with the environment taken from
[`example.sh`](example.sh).

**Generation only** — no scoring job is submitted. Run the metric scripts on the output
directories afterwards ([below](#metrics-afterwards)).

| File | Role |
|---|---|
| `eval_env.sh` | every path and sweep setting, in one place. Sourced by all three scripts. |
| `eval_gen_worker.sh` | **shard mode**: one array element = one shard = one GPU, `--requeue` + `USR1` trap |
| `eval_node_worker.sh` | **node mode**: one job = whole node, all shards in parallel (`example.sh` shape) |
| `submit_eval.sh` | launcher: `preflight` / `check-env` / `print` / `submit` / `submit-node` / `status` |

Generation is not reimplemented here — both workers call
`eval/eval.sh shard <method> <shape:radius> <shard_id> <num_shards>`, so a cluster shard and
a local shard are the same computation with the same flags.

---

## 1. First time

```bash
cd /users/brianchc/foveated_diffusion/marlowe_scripts
./submit_eval.sh preflight      # every path, plus the submitted-job count
./submit_eval.sh check-env      # actually activates conda and imports torch/diffsynth
cd ../eval && ./eval.sh masks   # confirms all arms derive an identical mask (no GPU)
```

Environment, from `example.sh` (override any of it inline):

| Variable | Value |
|---|---|
| `ROOT` | `/scratch/m000051-pm05/brianchc_2` |
| `REPO` | `/users/brianchc/foveated_diffusion` |
| `CONDA_SH` | `/cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh` |
| `CONDA_ENV` | `/scratch/m000051/brian/conda_environments/camera_ctrl` (a prefix path) |
| `DIFFSYNTH_MODEL_BASE_PATH` | `$ROOT/models` |
| `ACCOUNT` / `PARTITION` | `marlowe-m000051` / `preempt` |
| `NOFOV_LORA` / `FOV_LORA` | `$ROOT/foveated_diffusion/experiments/{finetuned_no_fov/step-15000,rank_32_fov_more_steps/step-5000}.safetensors` |

> `REPO` must be the checkout that contains `eval/RALU`, `eval/importance-token-merge`,
> `eval/eval.sh` and `eval/shard_manifest.py`. `preflight` checks for each of those
> individually, so a stale checkout fails loudly instead of at the first worker.

---

## 2. Launch

### Shard mode — the rules' pattern (8 single-GPU array elements per config)

```bash
./submit_eval.sh submit ours circular:0.25     # one method + one config
./submit_eval.sh submit ours                   # one method, every config
./submit_eval.sh submit-all                    # every method x config
```

`submit ours circular:0.25` runs exactly:

```bash
sbatch --parsable --account marlowe-m000051 \
    --job-name fovgen_ours_circular_r0.25 \
    --partition preempt \
    --array 0-7 \
    --nodes 1 --ntasks-per-node 1 \
    --gpus-per-node 1 --cpus-per-task 8 --mem 128G \
    --time 01:00:00 \
    --output $ROOT/foveated_diffusion/logs/eval_v2/fovgen_ours_circular_r0.25_%A_%a.out \
    --error  $ROOT/foveated_diffusion/logs/eval_v2/fovgen_ours_circular_r0.25_%A_%a.err \
    --export NIL,METHOD=ours,MASK_CONFIG=circular:0.25,NUM_SHARDS=8 \
    /users/brianchc/foveated_diffusion/marlowe_scripts/eval_gen_worker.sh
```

### Node mode — `example.sh`'s shape (1 job per config, whole node)

```bash
./submit_eval.sh submit-node ours circular:0.25
./submit_eval.sh submit-node-all
```

```bash
sbatch --parsable --account marlowe-m000051 \
    --job-name fovnode_ours_circular_r0.25 \
    --partition preempt \
    --nodes 1 --ntasks-per-node 1 \
    --gpus-per-node 8 --cpus-per-task 64 --mem 1024G \
    --time 01:00:00 \
    --output $ROOT/foveated_diffusion/logs/eval_v2/fovnode_ours_circular_r0.25_%j.out \
    --error  $ROOT/foveated_diffusion/logs/eval_v2/fovnode_ours_circular_r0.25_%j.err \
    --export NIL,METHOD=ours,MASK_CONFIG=circular:0.25,NUM_SHARDS=8 \
    /users/brianchc/foveated_diffusion/marlowe_scripts/eval_node_worker.sh
```

| | shard mode | node mode |
|---|---|---|
| job records, full sweep | 21 x 8 = **168** | **21** |
| requeue granularity | one shard | all 8 shards together (each still resumes from its manifest) |
| queue wait | 8 small allocations | one whole node |

`print` / `print-node` emit these command lines without submitting, if you would rather
drive `sbatch` yourself.

### Randomized (frame-constrained) mask centers

```bash
RANDOM_MASK_CENTER=true ./submit_eval.sh submit-all
```

Output dirs gain a `_randcenter42` suffix, so randomized and centered sweeps never mix.

### Smoke test first

```bash
NUM_PROMPTS=16 NUM_SHARDS=2 OUT_ROOT=$ROOT/foveated_diffusion/results/eval_smoke \
    ./submit_eval.sh submit ours circular:0.25
```

### What else the launcher can set

Everything is an environment variable — no file edits needed:

```bash
NUM_SHARDS=16            # array size = concurrent GPUs (shard mode)
GEN_GPUS=1 GEN_CPUS=8 GEN_MEM=128G GEN_TIME=01:00:00        # per element
NODE_GPUS=8 NODE_CPUS=64 NODE_MEM=1024G NODE_TIME=04:00:00  # node mode
NODE_GPU_IDS="0 1 2 3"   # pin specific devices in node mode
OUT_ROOT=... LOG_DIR=... # where results and logs go
PARTITION=batch ACCOUNT=marlowe-m000051-pm05                # example.sh's batch pairing
METHODS="ours ralu" MASK_CONFIGS="circular:0.25 square:0.5"
NUM_PROMPTS=1000 STEPS=50 SEED=42 CFG=4.0 LR_FACTOR=2
MERGE_MLP=true RALU_STAGES=3 RALU_LEVEL=4 RALU_Z_MAX=6.5
MAIL_TYPE=END,FAIL MAIL_USER=brianchc@stanford.edu           # off by default
```

---

## 3. Monitor

```bash
./submit_eval.sh status         # per-shard markers + image counts per method/config
squeue -u $USER -o "%.18i %.30j %.8T %.10M %R"
tail -f $ROOT/foveated_diffusion/logs/eval_v2/fovgen_ours_circular_r0.25_*_0.out
```

A shard is finished when `.generation_shard<id>_complete` exists in its output dir. Nothing
else is written on exit, so a missing marker means real work is still pending.

---

## Metrics afterwards

Each output dir has `img_%010d.png` + `metadata_%05d.csv`, so the existing scripts consume
it unchanged:

```bash
D=$ROOT/foveated_diffusion/results/eval_v2/circular_r0.25/ours
python $REPO/eval/generate_metrics.py --generated_images_dir $D \
    --images_dataset_path $ROOT/Aesthetic-Train-V2 \
    --metric_output_dir $ROOT/foveated_diffusion/results/metrics_v2/circular_r0.25/ours
python $REPO/eval/compute_hpsv2_score.py --images_dir $D --output_dir <same metric dir>
python $REPO/eval/make_eval_summary.py --out_root $ROOT/foveated_diffusion/results/eval_v2 \
    --mask_tag circular_r0.25 --num_prompts 6
```

---

## How the rules are satisfied

| Rule | Where |
|---|---|
| Deterministic split | `eval.sh shard` passes `--num_subsets/--subset_idx`; drivers take prompts with `global_idx % num_shards == shard_id` |
| One GPU, preempt allocation | `eval_gen_worker.sh`: `--gpus-per-node 1 --time 01:00:00 --partition preempt` |
| Save each result immediately, flush the manifest row after the output | `eval/shard_manifest.py:append` writes + `fsync`s **after** the image is saved |
| Skip completed samples on start/requeue | `ShardManifest.is_done` requires the row **and** the image; rows without an image are dropped and regenerated (no duplicate rows) |
| Per-shard manifests, no concurrent writes | one `metadata_<shard>.csv` per element |
| `--export=NIL` | on every `sbatch`; the worker rebuilds its environment from `eval_env.sh` |
| Requeue the element, not the base array id | `scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"` |
| Node-local, disposable compile caches | `setup_node_scratch`: `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` under `${SLURM_TMPDIR:-/tmp/$USER}`, plus `example.sh`'s per-job MPS directories |
| Unique output dir per config | `$OUT_ROOT/<shape>_r<radius>[_randcenter<seed>]/<method>/`; `high_res` once, shared |
| Append Slurm logs | `--open-mode=append` |
| Completion marker only when nothing is pending | `ShardManifest.mark_complete()`, after the prompt loop drains |
| Scoring with `afterok` | **not used** — generation only, by choice |

Verified locally (paths overridden to this machine, `--export=NIL`-style clean env via
`env -i`):

- shard worker, 1 element on one GPU: 2/2 images, manifest + marker written, marker
  reported in the log
- node worker, 2 shards on two GPUs: 4/4 images, split as shard 0 → prompts 0,2 and
  shard 1 → 1,3, two independent manifests, both markers
- resume: untouched rerun generates **0**; delete one image and rerun regenerates exactly
  that one and leaves 2 manifest rows, not 3 — confirmed for `naive`, `tome`, `ralu`

## Sizing

At ~18 s/image, 10,000 prompts over 8 shards is ~1,250 images/shard ≈ **6-7 GPU-hours per
shard**, so on the rules' one-hour allocation expect ~7 requeues per element, each reloading
the model (~1-2 min, roughly 10-15% overhead). Raise `GEN_TIME` if preempt allows longer
(`example.sh` uses 12 h), or raise `NUM_SHARDS` to cut wall-clock.

Check the submitted-job limit before `submit-all` (168 records in shard mode):

```bash
sacctmgr show assoc where user=$USER format=account,maxsubmit
```

If it is lower, either use node mode (21 records) or submit per method as earlier pairs drain.
