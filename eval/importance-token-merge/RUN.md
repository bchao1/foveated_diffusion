# Running the ToMe-2x2 baseline

Method and porting details are in [`TOME_FOVEATED.md`](TOME_FOVEATED.md). This file is
just how to launch it.

## Large-scale eval: `tome_eval_10k.sh`

For the paper's quantitative table, use **`tome_eval_10k.sh`** rather than `run_tome.sh eval`.
It is the ToMe counterpart of `eval/naive.sh`: the full 10,000-prompt Aesthetic-Train-V2 test
set, seed 42, the same `finetuned_no_fov/step-15000` LoRA, at every mask config in one go,
each sharded across the configured GPUs.

```bash
cd eval/importance-token-merge
./tome_eval_10k.sh plan                    # configs, token budgets, output dirs — no GPU
GPUS="1 7 8" ./tome_eval_10k.sh            # plan, then run
./tome_eval_10k.sh check                   # PNG counts vs the 10,000 expected
MASK_CONFIGS="square:0.1" NUM_PROMPTS=8 ./tome_eval_10k.sh run     # smoke test
```

Every setting lives in a commented `CONFIGURE` block at the top of the script and is
overridable inline. `MASK_CONFIGS` is a list of `shape:radius` pairs, defaulting to the five
that have completed `naive_mixed_res_*` dirs, so every ToMe dir has a naive/ours dir at the
identical mask. Output dirs follow the naive naming (`tome_2x2_r_0.1_eval`,
`tome_2x2_circle_r_0.25_eval`). `SKIP_COMPLETE=1` (default) means re-running the script
resumes an interrupted sweep, and each config is verified at 10,000 PNGs before moving on.

`tome_eval.sh` is the older single-config 1k-prompt run on `claude_prompts.csv`; it is kept
for the quick checks it was written for.

Everything goes through **`run_tome.sh`**, which wraps `tome_fov_inference.py`. Every knob
is an environment variable with a default, so you override only what you care about:

```bash
cd eval/importance-token-merge

./run_tome.sh single                                  # one prompt, defaults
STEPS=28 RADIUS=0.5 ./run_tome.sh single              # fewer steps, bigger fovea
MERGE_MLP=true ./run_tome.sh single                   # merge the image MLP too
MODEL_PATH=/local/models/FLUX.2-klein-base-4B ./run_tome.sh single
GPUS="0 1 2 3" NUM_PROMPTS=500 ./run_tome.sh eval     # sharded full eval
```

---

## Modes

| Mode | What it does | Output |
|---|---|---|
| `single` | one prompt on one GPU | `$OUT/single/img_0000000000.png` |
| `eval` | full prompt CSV, sharded across `$GPUS` | `$OUT/eval/img_%010d.png` + `metadata_%05d.csv` |
| `sweep` | one prompt at each radius in `$RADII`, in parallel | `$OUT/sweep/r0.1/`, `r0.2/`, … |
| `compare` | ToMe-attn, ToMe-attn+MLP, merging-off, and **ours** — same prompt, mask and step count | `$OUT/compare/{tome_attn,tome_attn_mlp,tome_off,ours}/` |

`eval` writes the same layout as `eval/inference.py`, so the existing metric scripts
consume it unchanged (see [Metrics](#metrics)).

---

## Knobs

### Model

| Var | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `black-forest-labs/FLUX.2-klein-base-4B` | Hub id, resolved under `$DIFFSYNTH_MODEL_BASE_PATH`. |
| `MODEL_PATH` | *(unset)* | **Local model directory**; overrides `MODEL_ID`. Must contain `transformer/`, `text_encoder/`, `vae/`, `tokenizer/`. Sharded `.safetensors` are globbed automatically. |
| `LORA` | `…/finetuned_no_fov/step-15000.safetensors` | LoRA for the ToMe arms. Set to `""` for no LoRA. |
| `FOV_LORA` | `…/rank_32_fov_more_steps/step-5000.safetensors` | LoRA for the "ours" arm in `compare` mode only. |

### Generation

| Var | Default | Meaning |
|---|---|---|
| `STEPS` | `50` | Denoising steps. Match whatever our runs use. |
| `SEED` | `42` | Global seed. |
| `CFG` | `4.0` | Guidance scale. |
| `HEIGHT` / `WIDTH` | `1024` | Output resolution. Must be divisible by `16 * LR_FACTOR`. |

### Foveation mask

| Var | Default | Meaning |
|---|---|---|
| `MASK` | `circular` | `circular`, `square`, or `checkerboard`. |
| `RADIUS` | `0.3` | Fovea size — the "mask ratio". For `circular`, radius relative to half the diagonal; for `square`, side as a fraction of the image. Ignored by `checkerboard`. |
| `LR_FACTOR` | `2` | Periphery downsample factor. `2` merges 2x2 blocks, `4` merges 4x4. |

The mask comes from `eval/inference.py`'s own `create_foveation_mask`, so for a given
`(MASK, RADIUS, LR_FACTOR, HEIGHT, WIDTH)` it is bit-identical to the mask our runs use.

Token counts at 1024², `LR_FACTOR=2` (printed at startup and saved to `tome_config.txt`):

| `MASK` | `RADIUS` | image tokens | kept |
|---|---|---|---|
| circular | 0.3 | 4096 → 1468 | 35.8% |
| circular | 0.4 | 4096 → 1792 | 43.8% |
| — | *(merging off)* | 4096 → 4096 | 100% |

### Token merging

| Var | Default | Meaning |
|---|---|---|
| `MERGE_MLP` | `false` | Also merge the image MLP in the 8 double-stream blocks. **Read the caveat below.** |
| `ENABLE_TOME` | `true` | `false` runs the identical code path with merging disabled — the identity control, which must match `high_res`. |

> **`MERGE_MLP` caveat.** FLUX.2-klein is 8 double blocks + 48 single blocks, and the single
> blocks are *parallel* blocks whose attention QKV and MLP input come from one fused
> projection. Merged tokens there merge the MLP whether you like it or not. So `MERGE_MLP`
> only controls 8 of 56 blocks, and `MERGE_MLP=false` is **not** a "no MLP savings" setting.
> See `TOME_FOVEATED.md`.

### Runtime

| Var | Default | Meaning |
|---|---|---|
| `GPUS` | `"0 1 2 3 4 5 6 7"` | GPUs to use. `single` uses the first; `eval` shards across all. |
| `OUT` | `../../results/tome` | Output root. Each mode adds its own subdirectory. |
| `PROMPT` | a lion prompt | Prompt for `single` / `sweep` / `compare`. |
| `PROMPTS_CSV` | `../prompts/claude_prompts.csv` | Prompt CSV for `eval` (needs a `prompt` column). |
| `NUM_PROMPTS` | `1000` | Cap on prompts in `eval`. |
| `RADII` | `"0.1 0.2 0.3 0.4 0.5"` | Radii for `sweep`. |
| `PY` | `…/envs/imp_tome/bin/python` | Interpreter. |
| `REPO` | `/home/brianchc/foveated_diffusion-private` | Repo root; sets `PYTHONPATH`. |

`run_tome.sh` sets `DIFFSYNTH_MODEL_BASE_PATH`, `DIFFSYNTH_SKIP_DOWNLOAD` and `PYTHONPATH`
itself. The `PYTHONPATH` line matters: the `imp_tome` env otherwise resolves `diffsynth` to
`/home/brianchc/DiffSynth-Studio`, a different checkout.

---

## Every run records its own config

Each output directory gets a `tome_config.txt`:

```
variant=tome_2x2_mask
mask=circular r=0.3 lr_factor=2
tokens=4096 -> 1468
merge_mlp=False (double blocks only; single blocks always merge both)
enable_tome=True
model=black-forest-labs/FLUX.2-klein-base-4B
lora=/local/brian/.../finetuned_no_fov/step-15000.safetensors
steps=50 cfg=4.0 seed=42
```

Check `tokens=` against our run's token count before trusting any comparison.

---

## Metrics

`eval` mode emits `img_%010d.png` + `metadata_%05d.csv`, matching `eval/inference.py`, so:

```bash
cd ..
python compute_clip_score.py  --image_dir ../results/tome/eval
python compute_hpsv2_score.py --image_dir ../results/tome/eval
python extract_inception_features.py --image_dir ../results/tome/eval
```

Metrics are computed on the **full image** for both arms. ToMe outputs uniform full
resolution while ours is mixed-resolution, so report token count and latency alongside
quality.

---

## Calling the driver directly

`run_tome.sh` is a convenience wrapper; `tome_fov_inference.py` takes the same flags as
`eval/inference.py` where they overlap.

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/miele/brian/modelscope DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH=/home/brianchc/foveated_diffusion-private/DiffSynth-Studio

CUDA_VISIBLE_DEVICES=0 python tome_fov_inference.py \
    --model_path /miele/brian/modelscope/black-forest-labs/FLUX.2-klein-base-4B \
    --lora_checkpoint /local/brian/.../finetuned_no_fov/step-15000.safetensors \
    --num_inference_steps 50 --seed 42 --guidance_scale 4.0 \
    --full_eval_mask circular --mask_radius 0.3 --lr_downsample_factor 2 \
    --merge_mlp false \
    --prompt "A lion lounges calmly..." \
    --output_dir ../../results/tome/manual

python tome_fov_inference.py --help    # full flag list
```

---

## Before trusting results

```bash
PYTHONPATH=$REPO/DiffSynth-Studio python -m flux2_tome.test_merge_2x2
PYTHONPATH=$REPO/DiffSynth-Studio python -m flux2_tome.test_patch_flux2
```

```bash
python verify_config_parity.py --arm tome --out parity/tome.json
python verify_config_parity.py --arm ours --out parity/ours.json
python verify_config_parity.py --compare parity/      # exit 0 == all match
```

The parity check confirms the ToMe arm receives the same `cfg_scale`, `embedded_guidance`,
timestep/sigma schedule and text embeddings as our foveated runs. Rerun it if you change
step count, guidance, resolution, or the text encoder.

Both must print all-pass. The merge tests check that our surviving-token set equals what
the foveated pipeline produces for the same mask; the patch tests check that merging off is
bit-exact against the unpatched model and that attention really sees the shortened sequence.

`ENABLE_TOME=false ./run_tome.sh single` is the same control end to end — it should match
the `high_res` baseline.

---

## Reference numbers

Measured with `./bench_all.sh` — arms run **sequentially** on a single idle RTX A6000,
1 warmup + 3 reps, 50 steps, 1024², circular `r=0.3`, `lr_factor=2`, seed 42.
`DiT s` is the summed CUDA-synced time of all 100 DiT forward calls (50 steps x 2 for CFG).

| Arm | Image tokens | DiT s | Speedup | Total s |
|---|---|---|---|---|
| `high_res` (merging off) | 4096 | 40.06 | 1.00x | 40.44 |
| ToMe-2x2, `MERGE_MLP=false` | 1468 | 20.43 | 1.96x | 20.80 |
| ToMe-2x2, `MERGE_MLP=true` | 1468 | 18.72 | 2.14x | 19.09 |
| **ours** | 1468 | 18.89 | 2.12x | 19.36 |
| `naive_mixed_res` | 1468 | 18.85 | 2.13x | 19.30 |

Rep-to-rep spread is +/-0.05s.

**Ours and ToMe-`MERGE_MLP=true` are within 0.9% of each other.** At a matched token budget
they cost the same, so the comparison is quality-at-equal-cost. Use `MERGE_MLP=true` as the
headline ToMe config — `MERGE_MLP=false` just handicaps it by 1.7s, which is the 8 double
blocks running their feed-forward at 4096 instead of 1468.

Why ToMe isn't slower despite keeping a full-length residual stream: 48 of 56 blocks are
parallel blocks, so merging before the fused `to_qkv_mlp_proj` puts both attention *and* MLP
at 1468. What stays at 4096 there is LayerNorm, the residual add, and the merge/unmerge
gather — memory-bound ops that are cheap next to the matmuls.

`naive` matches `ours` because with `tokenAE_training=false` neither calls the TokenAE; they
differ only in LoRA weights.

Reproduce:

```bash
./bench_all.sh                                   # defaults as above
GPU=3 STEPS=28 RADIUS=0.5 REPEATS=5 ./bench_all.sh
python bench_tome_vs_ours.py --report ../../results/bench_tome_vs_ours
```

`total s` is not comparable across arms — ours uses `decode_mode="merge"` and decodes the VAE
twice. Compare `DiT s`.
