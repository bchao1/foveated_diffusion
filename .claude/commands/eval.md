---
description: Run eval/eval.sh — translate a plain-English eval request into the right env vars + subcommand, check GPUs, launch
argument-hint: e.g. "smoke test ours on distilled, 2 prompts, gpu 7" | "all arms, random centers, 8 gpus" | "plan" | "check"
allowed-tools: Bash, Read, Edit
---

Run `/home/brianchc/foveated_diffusion-private/eval/eval.sh` according to this request:

**$ARGUMENTS**

Everything you need to construct the command is below. **Do not read `eval.sh`** unless
something below is contradicted by an error at runtime.

## Invocation shape

```
[ENV=VAL ...] /home/brianchc/foveated_diffusion-private/eval/eval.sh <subcommand>
```

Subcommands: `plan` (resolved config + current PNG counts, no GPU) · `masks` (prove all arms
share an identical mask, no GPU) · `check` (per-arm completeness) · `all` (all methods,
sequentially) · one of `high_res|naive|ours|tome|ralu` · `shard <method> <shape:radius> <id> <n>`
(Slurm only — never use locally).

The script forks one background worker per GPU in `GPUS`, pinned with `CUDA_VISIBLE_DEVICES`,
and `wait`s. It resolves the repo from its own path, so cwd doesn't matter.

## Env knobs (all optional, defaults shown)

| Var | Default | Notes |
|---|---|---|
| `MASK_CONFIGS` | `"square:0.1 square:0.25 square:0.5 circular:0.25 circular:0.75"` | space-separated `shape:radius`; every method runs every entry |
| `METHODS` | `"high_res naive ours tome ralu"` | only used by `all`/`plan`/`check` |
| `GPUS` | `"0 1 2 3 4 5 6 7"` | one shard per GPU |
| `MODEL` | `base` | `base` → klein-base-4B, 50 steps, cfg 4.0 · `distilled` → klein-4B, 4 steps, cfg 1.0 |
| `MODEL_ID` / `STEPS` / `CFG` | follow `MODEL` | override individually only if asked |
| `PROMPTS` | `/local/brian/Aesthetic-Train-V2/metadata_test.csv` | 10k-row csv, `image,prompt` header |
| `NUM_PROMPTS` | empty (= all) | cap |
| `PROMPT` / `PROMPT_LIST` / `PROMPT_PRESET` | — | ad-hoc prompts, materialised to a csv. `PROMPT_LIST` is `|`-separated. `PROMPT_PRESET=lion` is the only preset |
| `RANDOM_MASK_CENTER` | `false` | `true` = per-prompt fixation point, index-addressable so identical across arms |
| `MASK_CENTER_SEED` / `MASK_CENTER_RANGE` / `CLAMP_MASK_TO_FRAME` | `42` / `0.3` / `true` | |
| `HEIGHT` / `WIDTH` / `SEED` / `LR_FACTOR` | `1024` / `1024` / `42` / `2` | |
| `NOFOV_LORA` / `FOV_LORA` | eccv2026_final checkpoints | `""` = base model. `ours` uses fov; all other arms use no-fov |
| `TOKENAE_FOLDER` | empty | `ours` only |
| `MERGE_MLP` | `true` | tome |
| `RALU_N` / `RALU_STAGES` / `RALU_LEVEL` / `RALU_Z_MAX` / `OURS_STEPS` | empty / `3` / `4` / `6.5` / `=STEPS` | explicit `RALU_N` (e.g. `"1 2 1"`) beats compute-matching; needed at small budgets |
| `OUT_ROOT` | `/local/brian/foveated_diffusion/results/eval_v2` | |
| `LOG_DIR` | `$OUT_ROOT/logs` | |
| `SKIP_COMPLETE` | `1` | skips a (method, config) already at the expected PNG count — makes reruns resumable |

## Output layout

```
$OUT_ROOT/<shape>_r<radius>[_randcenter<seed>]/<method>/img_%010d.png + metadata_%05d.csv
$OUT_ROOT/high_res/                      # mask-independent, generated once, shared by all configs
$OUT_ROOT/logs/<method>_<mask_tag>_shard<i>.log
```

## Procedure

1. **Map the request to env vars + subcommand.** If it names no method, use `all`. Words like
   "smoke test" / "quick" mean a small `NUM_PROMPTS`, one `MASK_CONFIGS` entry, and few GPUs —
   pick sane values and say what you picked.
2. **Check GPUs before any run that touches one** (`plan`, `masks`, `check` don't). The box is
   shared: run `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv` (plus
   `nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv`) and set `GPUS` to only
   free devices. If the user named GPUs that are busy, say so and ask before proceeding.
3. **Dry-run first for anything large** (more than a few hundred images per arm): run `plan`,
   show the user the resolved config and total image count, and confirm before launching.
   Small/smoke runs: just go.
4. **Launch in the background** (`run_in_background: true`) for anything non-trivial, and
   report the log directory. Foreground is fine for `plan`/`masks`/`check`.
5. **Report** the exact command line you ran, then the OK/WARN lines. On WARN, tail the
   offending `$LOG_DIR/<method>_<tag>_shard*.log`.

## Examples

```bash
NUM_PROMPTS=2 MASK_CONFIGS="circular:0.25" GPUS="5" ./eval.sh tome
MODEL=distilled GPUS="0 1 2 3" ./eval.sh all
GPUS="0 1 2 3 4 5 6 7" RANDOM_MASK_CENTER=true ./eval.sh all
MODEL=distilled PROMPT_PRESET=lion MASK_CONFIGS="square:0.5" GPUS="7" ./eval.sh all
PROMPT_LIST="a red fox in snow|a lion at dusk" GPUS="0 1" ./eval.sh ours
```

If the request is about **summarizing results** rather than generating them, that's
`eval/make_eval_summary.py --out_root ... --mask_tag <shape>_r<radius>[_randcenter<seed>] --num_prompts N`,
not `eval.sh`.

If the request can't be expressed with the knobs above, say so and propose the closest thing —
don't invent flags.
