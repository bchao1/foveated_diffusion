# RALU-foveated baseline

RALU ([CVPR 2026](https://arxiv.org/abs/2507.08422)) ported to FLUX.2-klein and driven by
**our foveation mask** instead of its edge-based token selection, so it can be compared
against foveated diffusion under matched conditions.

Upstream files (`pipeline_flux_RALU.py`, `RALU_NTDM.py`, `RALU_inference.py`, `README.md`)
are **unmodified**. Everything here is additive.

---

## What was changed, and why

The port keeps RALU's method intact. Exactly one methodological change:

| | Original RALU | Here |
|---|---|---|
| Which tokens get upsampled in stage 2 | top-`k` Canny edge blocks (`up_ratio`) | blocks selected by our foveation mask |

Everything else is theirs: the uniform low-res stage 1, the per-stage tailored shift
schedule, NT-DM noise-timestep rescheduling, correlated re-noising at each transition,
nearest-neighbour token upsampling, the `+0.5` RoPE id refinement, and the stage-3 rescale
plus raster sort.

Port-only differences, forced by the base model and none of them methodological:

- 4-column `(T,H,W,L)` position ids instead of FLUX.1's 3-column.
- Latent dim 128 (patchified 32-channel VAE latents) instead of 64.
- `h_ori = exp(compute_empirical_mu(seq_len, steps))`. FLUX.2's exponential-mu shift is the
  same functional family as FLUX.1's shift factor, so NT-DM transfers directly, but their
  published `shift`/`Z` constants assume FLUX.1-dev's `h_ori=3` and must be re-solved
  (FLUX.2 at 1024², 50 steps is `h_ori≈7.56`).
- NT-DM's shift bound widened. Their `(0.5, 10.0)` is calibrated to `h_ori=3`; at 7.56 the
  stage-1 shift pins against it.
- NT-DM generalized to K stages (`RALU_NTDM_general.py`) — theirs hardcodes 3.
- Correlated noise sampled analytically instead of via a dense 4096² `MultivariateNormal`.

`RALU_NTDM_general.py` is gated by `test_ntdm_regression.py`, which requires the K-stage
solver to reproduce their published constants:

```bash
python test_ntdm_regression.py      # must print PASS
```

It agrees with their 3-stage solver to 0.00e+00 and with their published constants to
2.6e-4 (level 4) / 4.7e-3 (level 7, their own rounding).

---

## Stage schedule

Everything RALU's own optimizer decides is left alone: `e` (stage boundaries) and `Z` /
`shift` (NT-DM's solve) are whatever RALU picks. Only the per-stage **step counts** are
set by us, since those encode a target speedup rather than a property of the method.

Two ways to set them:

### Step-matched (`--num_inference_steps 50`)

Their proportions rescaled to the given total, via largest-remainder allocation. Recovers
their published settings at their native totals:

| total steps | level 4 | level 7 |
|---|---|---|
| 10 | `[3,3,4]` | `[2,3,5]`  <- their published 7x |
| 18 | `[5,6,7]`  <- their published 4x | `[4,5,9]` |
| 50 | `[14,17,19]` | `[10,15,25]` |

### Compute-matched (`--match_compute`) — preferred

Equalizes token-steps with our method on the same mask, holding their stage proportions
fixed and solving for the total. Per-step token counts:

| stage | tokens | |
|---|---|---|
| 1 | `M_down` | uniform low-res |
| 2 | `M_down + 3m` | mixed — **identical to our method's token set** |
| 3 | `M = 4*M_down` | uniform full-res (3-stage only) |

with `m` = foveal blocks. Our cost is `50 * (M_down + 3m)`, since we run the mixed set
every step. Examples at 1024², `f0=2`:

| mask | m | area | ours cost | 2-stage N | 3-stage N |
|---|---|---|---|---|---|
| circular r=0.1 | 16 | 1.6% | 53600 | `[14,37]` (51) | `[7,8,9]` (24) |
| circular r=0.3 | 148 | 14.5% | 73400 | `[15,40]` (55) | `[9,10,12]` (31) |
| circular r=0.5 | 392 | 38.3% | 110000 | `[16,43]` (59) | `[12,14,16]` (42) |
| circular r=0.7 | 788 | 77.0% | 169400 | `[17,45]` (62) | `[15,19,22]` (56) |

Matching lands within ~1% in every configuration tested. 2-stage receives *more* steps
than ours because it never pays for a full-res stage; 3-stage receives *fewer* because
stage 3 at 4096 tokens is expensive. Both are the intended behaviour.

Two caveats on the cost model:

- It counts token-steps, a **linear** proxy that ignores attention's quadratic term. That
  slightly favours RALU's short full-res stage. At r=0.3 the matched runs measure 7.1s
  (3-stage) and 8.3s (2-stage) despite equal token-steps, so quote wall-clock as well.
- At very small masks with `lr_factor=4`, the 3-stage solve floors at the 1-step-per-stage
  minimum (r=0.1 gives 9 steps, ratio 0.886). Check `cost_ratio` in the stats JSON.

`up_ratio` no longer applies: the mask replaces it. Their level-4 `up_ratio` is 0.3 while
a circular r=0.3 mask is 14.45% of blocks, so RALU runs at roughly half the foveal budget
its proportions were tuned around.

### A note on NT-DM on FLUX.2

Their JSD objective is monotone in `Z` at FLUX.2's `h_ori=7.56` (it was well-conditioned
at FLUX.1-dev's 3), so the free solve lands near `Z≈28` with `beta≈[0.05, 0.06]` — the
stage transitions discard almost all of the previous stage, versus their published
`beta≈[0.21, 0.25]`. `--z_max` clamps it; `--z_max 6.5` restores their re-noising
strength (`beta≈[0.21, 0.25]`).

**Use `--z_max 6.5` for 3-stage.** At correct CFG it is better than the free solve on
every measurement taken — brighter, higher CLIP, and visibly cleaner at every radius (see
the rerun below and `results/ralu_compare/cfg4/SUMMARY_cfg4_lion.png`). An earlier version
of this doc reported a brightness-versus-ghosting trade-off with no good setting; that
sweep was run at `cfg_scale=1.0` and the "ghosting" it attributed to low `Z` does not
reproduce at 4.0.

## The comparison at matched guidance (current numbers)

Everything below is at `cfg_scale=4.0` on both sides. 16 prompts from
`prompts/claude_prompts.csv`, circular r=0.3, `lr_downsample_factor` 2, seed 42, RALU
compute-matched to ours@50. Reproduce with `./rerun_cfg4.sh`, summarize with
`analyze_cfg4.py [--clip]`. Raw output in `results/ralu_compare/cfg4/` (`summary.csv`,
`contact_sheet.png`).

| config | steps | Z | beta | mean px | paired ratio vs plain | CLIP | s/img |
|---|---|---|---|---|---|---|---|
| plain full-res, 50 steps | 50 | — | — | 77.5 +/- 34.5 | 1.00 | 0.311 | — |
| **ours** | 50 | — | — | 84.9 +/- 34.2 | **1.13** | 0.301 | — |
| RALU 3-stage, Z free | [9,10,12] | 28.2 | .05/.06 | 55.9 +/- 31.1 | 0.71 | 0.309 | 14.9 |
| **RALU 3-stage, `--z_max 6.5`** | [9,10,12] | 6.5 | .21/.25 | 60.3 +/- 32.4 | **0.78** | **0.313** | 15.0 |
| RALU 2-stage, Z free | [15,40] | 100 | .014 | 65.1 +/- 33.9 | 0.83 | 0.298 | 16.6 |
| RALU 2-stage, `--z_max 6.5` | [15,40] | 6.5 | .21 | 69.4 +/- 34.2 | 0.90 | 0.292 | 16.7 |
| RALU published level 4 verbatim | [5,6,7] | 6.32 | .21/.26 | 56.2 +/- 32.8 | 0.70 | 0.316 | 8.5 |

Compute matching holds: cost ratio 0.995 (3-stage) / 1.009 (2-stage).

Reading the CLIP column: RALU 3-stage renders the whole frame at full resolution while
ours deliberately leaves the periphery low-res, so a whole-image CLIP score structurally
favours RALU. `ours` 0.301 against `plain` 0.311 is the honest measure of what foveation
costs; `ours` against RALU's 0.313 is cross-modality (same caveat as the 2-stage section).
The 2-stage runs are the exception that shows brightness is a bad proxy on its own — they
are the *closest* to plain in brightness and the *worst* in CLIP, because their
weave artifacts are bright.

Per-radius, one prompt, `SUMMARY_cfg4_lion.png` (mean pixel value):

| r | 3-stage Z free | 3-stage Z=6.5 | ours | plain |
|---|---|---|---|---|
| 0.1 | 88.7 | 95.9 | 87.2 | 103.4 |
| 0.3 | 70.5 | 84.8 | 96.8 | 103.4 |
| 0.5 | 79.1 | 89.7 | 88.2 | 103.4 |
| 0.7 | 71.9 | 91.0 | 92.0 | 103.4 |

The `ours` column reproduces the pre-fix run exactly (96.8 / 55.8 at r=0.3), which
confirms only the RALU side moved.

## Superseded: "RALU does not transfer to FLUX.2 (dark output)"

> **Mostly a guidance mismatch.** Every FLUX.2 number in this section was measured at
> `cfg_scale=1.0` against baselines running at 4.0. At matched guidance the 3-stage
> compute-matched deficit goes from 0.36 to 0.71 (free Z) / 0.78 (Z=6.5) — so CFG was
> worth roughly half of it. A ~22-29% deficit remains and is still unexplained, but this
> is no longer a "2.8x too dark, does not transfer" result and should not be reported as
> one. The FLUX.1 rows below stand; the FLUX.1-vs-FLUX.2 ratios mix guidance regimes and
> are superseded by the table above. Kept for the mechanism and the FLUX.1 control.

RALU's output on FLUX.2 is systematically ~2.8x too dark. This was traced to the base
model's shift factor, not to this port.

### Evidence

Mean pixel value, 8-16 prompts from `prompts/claude_prompts.csv`, same prompts and seed
within each row:

| model | RALU | plain | ratio |
|---|---|---|---|
| FLUX.1-dev (their base model, their code, their level-4) | 72.3 | 68.0 @18 / 73.2 @50 | **0.99** |
| FLUX.2-klein (this port) | 27.8 | 77.5 | **0.36** |

On FLUX.1 the agreement is per-prompt, not just in the mean:

```
RALU     39.8  52.1  91.7  98.5  38.9  52.5  80.5  124.7
plain@18 37    44    97    97    25    40    79    124
```

Our foveation mask and our speedup-matched schedule are also correctly exposed on FLUX.1
(65.9, inside the plain spread) -- see `control_flux1_fov.py`, which runs our mask and N
through their pipeline by substituting the Canny edge map with the mask.

### Operating points measured on FLUX.2 (16 prompts, r=0.3, lr 2)

| config | base | mean | ratio to that model's plain |
|---|---|---|---|
| published level 4 verbatim (`--use_published`) | FLUX.1-dev | 72.3 | **0.99** |
| published level 4 verbatim | FLUX.2 | 45.1 +/- 26.2 | 0.58 |
| published level 7 verbatim | FLUX.2 | 29.4 +/- 23.5 | 0.38 |
| NT-DM re-solved for FLUX.2 (default) | FLUX.2 | 27.8 +/- 20.8 | 0.36 |
| speedup-matched N + their Z=5.54 | FLUX.2 | 52.1 +/- 26.1 | 0.67 |
| per-stage natural shift, Z free | FLUX.2 | 27.5 +/- 20.6 | 0.35 |

Two things follow. The identical published config gives 0.99 on FLUX.1 and 0.58 on
FLUX.2 -- same code, same parameters, same mask, only the base model differs. And
re-solving NT-DM for FLUX.2 makes it *worse*, not better.

Note their published level 4 is an 18-step, 4.8x speedup point while ours at r=0.3 is
2.79x, so that row gives RALU 58% of our compute. For a head-to-head against our method
use the compute-matched row (52.1); use the published rows for the transfer argument,
where compute is held constant across the FLUX.1/FLUX.2 comparison.

Substituting FLUX.2's per-stage natural shift, exp(mu(tokens_i, N_i)) = 7.35/7.59/9.44,
changes nothing (27.8 -> 27.5): shift only sets step spacing, not the meaning of sigma.
Z is the only effective lever and it trades brightness against ghosting -- no setting
gives both. Do not tune Z on brightness; the brightness-matched value (Z=2.1) produces
the worst images in the sweep.

### Mechanism

NT-DM is calibrated to the base model's shift. Same N=[9,10,12], same e:

| | h_ori | Z | beta |
|---|---|---|---|
| FLUX.1-dev | 3.00 | 5.54 (healthy; their published value is 6.32) | 0.24, 0.29 |
| FLUX.2-klein | 7.56 | 28.5 (degenerate) | 0.049, 0.062 |

At h_ori=7.56 the JSD objective is monotone in Z with no interior optimum, so the solve
runs to a regime where each transition discards ~95% of the previous stage instead of
~76%. Clamping Z to their operating point recovers ~40% of the deficit (27.8 -> 48.3),
confirming this is causal.

**Caveat:** the remaining ~60% is unaccounted for. The FLUX.1 result exonerates the mask
and schedule design, and every component of the FLUX.2 port tested clean individually
(ids exact, correlated noise correct, loop bit-identical under state transplant, stage
boundaries within 1% of expected sigma, shifts shown irrelevant across 4 values x 6
seeds), but the port is not proven bug-free.

### Consequence for the comparison

Superseded by the matched-guidance table above. Report the 3-stage `--z_max 6.5` row as
the RALU baseline, note the residual ~22% brightness deficit, and keep the FLUX.1 control
as evidence that the mask and schedule are not the cause.

### Methodological warning

Check that every arm of a comparison uses the same guidance before attributing a
difference to the method. Half of this investigation chased a base-model transfer failure
that was a `cfg_scale` default mismatch between two scripts.

Per-sample brightness variance is large (plain FLUX.2: 77.5 +/- 34.5, range 31.9-147.7).
Single-sample comparisons cannot localize anything and unpaired n=6 resolves only ~2
sigma. Two intermediate conclusions during this investigation were wrong for exactly this
reason. Use >=16 samples and prefer paired comparisons.

### Upstream issues found while running their code

1. `pipeline_flux_RALU.py` does not import as published: the `level == 5` block (commit
   87992fb) indents with 60 U+00A0 non-breaking spaces -> SyntaxError. See
   `pipeline_flux_RALU_importable.py` for a whitespace-only fixed copy.
2. That same block sets `Z = [7.522, 7.522]` as a list, which would fail at `1/(Z**2)`.
3. `RALU_NTDM.NT_DM` hardcodes 3 stages (indexes `E[1]`, `S[2]`), so `set_params` crashes
   on any other stage count.
4. `set_params` requires `level` positionally even when `use_RALU_default=False`.
5. `generate_noise` rebuilds a dense MultivariateNormal inside a `latent_dim`-iteration
   comprehension, recomputing the same 4096x4096 Cholesky 64x. Measured 2.19s at 1024^2,
   so slow but not a blocker; this port samples the same distribution analytically.

## Known limitation: 2-stage does not work

A 2-stage variant (drop stage 3 so the output stays mixed-resolution like ours) is
implemented but **does not produce usable images**. This is structural, not a bug.

RALU's stage-2 representation is transitional. Its low-res tokens are only coherent in the
coarse coordinate frame, and the method relies on stage 3 to lift everything into a valid
full-resolution frame. Ending at stage 2 leaves no valid frame:

- `--id_frame coarse` — ids as RALU leaves them (periphery at 0,1,2,…; fovea at 0.5 steps).
  Extent is 32, so the model generates a 512²-scale image: globally coherent, but half
  scale with weave artifacts.
- `--id_frame full` — apply their stage-3 `* f0` early. Extent 64 is correct and the fovea
  is sharp, but the untouched periphery lands on 0,2,4,… with empty gaps, so it collapses.

Verified in both the DiffSynth and diffusers ports, and re-verified at `cfg_scale=4.0`
with Z both free and clamped to 6.5 — the weave persists at every radius, worst at r=0.1
and 0.3. Guidance was never the cause here. RALU produces clean output **iff** its last stage ends in a valid full-resolution frame. Our
CRPA attention is what removes this constraint, by giving each resolution its own RoPE
frame simultaneously.

**Consequence for the comparison:** the working RALU baseline is 3-stage and outputs
uniform full resolution, while ours outputs mixed resolution. The headline comparison is
therefore cross-modality, and RALU spends full-resolution tokens on the periphery we
deliberately do not render. Report tokens and latency alongside quality.

---

## Running

Environment: `imp_tome`. `PYTHONPATH` must point at this repo's `DiffSynth-Studio`,
because `imp_tome` otherwise resolves `diffsynth` to `/home/brianchc/DiffSynth-Studio`,
a different checkout. `ralu.sh` and `ralu_eval_10k.sh` set this.

### Large-scale eval: `ralu_eval_10k.sh`

For the paper's quantitative table, use **`ralu_eval_10k.sh`** rather than `ralu.sh eval`.
It is the RALU counterpart of `eval/naive.sh`: the full 10,000-prompt Aesthetic-Train-V2 test
set, seed 42, cfg 4.0, the same `finetuned_no_fov/step-15000` LoRA, 3-stage compute-matched to
ours@50 with `--z_max 6.5`, at every mask config in one go, each sharded across the configured
GPUs.

```bash
cd eval/RALU
./ralu_eval_10k.sh plan                    # configs, solved schedules, cost ratios — no GPU
GPUS="1 7 8" ./ralu_eval_10k.sh            # plan, then run
./ralu_eval_10k.sh check                   # PNG counts vs the 10,000 expected
MASK_CONFIGS="square:0.1" NUM_PROMPTS=8 ./ralu_eval_10k.sh run      # smoke test
```

Every setting lives in a commented `CONFIGURE` block at the top of the script and is
overridable inline. `MASK_CONFIGS` is a list of `shape:radius` pairs, defaulting to the five
that have completed `naive_mixed_res_*` dirs, so every RALU dir has a naive/ours dir at the
identical mask. Output dirs follow the naive naming (`ralu_3stage_r_0.1_eval`,
`ralu_3stage_circle_r_0.25_eval`). `SKIP_COMPLETE=1` (default) means re-running the script
resumes an interrupted sweep, and each config is verified at 10,000 PNGs before moving on.

Compute matching at those five configs (from `plan`): `cost_ratio` 0.982-1.008.

### Mask reuse

`coarse_foveation_mask` now calls `eval/inference.py:create_foveation_mask` and inverts its
nearest upsample by a stride-`lr_factor` subsample, asserting the result round-trips to the
token mask bit-exactly. It previously re-derived the same geometry independently, which was
correct but free to drift. Verified identical to the old implementation for
{square, circular} x {0.1, 0.25, 0.3, 0.5, 0.7, 0.75} x `lr_factor` {2, 4}, and the emitted
`mask.png` is byte-identical to the ToMe baseline's at the same config.

```bash
cd eval/RALU

./ralu.sh single 0     # one prompt on GPU 0
./ralu.sh sanity 0     # all-ones mask; must produce a clean image
./ralu.sh eval         # full prompt set, sharded over 8 GPUs
```

Edit the configuration block at the top of `ralu.sh` (`MASK_SHAPE`, `MASK_RADIUS`,
`LR_FACTOR`, `NUM_STEPS`, `RALU_LEVEL`, `PROMPTS`, `OUT`).

Direct invocation:

```bash
CUDA_VISIBLE_DEVICES=0 python RALU_fov_inference_diffsynth.py \
    --output_dir ../../results/ralu_eval \
    --lora_checkpoint /path/to/finetuned_no_fov/step-15000.safetensors \
    --full_eval --prompt_dataset_path prompts/claude_prompts.csv --num_prompts 100 \
    --full_eval_mask circular --mask_radius 0.3 --lr_downsample_factor 2 \
    --num_inference_steps 50 --ralu_level 4 --seed 42
```

### Arguments

Mask, sharding, and output flags intentionally match `eval/inference.py`.

| Arg | Default | Description |
|---|---|---|
| `--full_eval_mask` | `circular` | Mask shape: `circular` or `square`. |
| `--mask_radius` | 0.3 | Foveation radius, same semantics as ours. |
| `--lr_downsample_factor` | 2 | Periphery downsample factor. |
| `--num_inference_steps` | 50 | Total steps, split in RALU's proportions. |
| `--ralu_level` | 4 | Published operating point supplying the `N`/`e` proportions. |
| `--N` / `--e` | None | Override the schedule directly; `e[-1]` must be 1.0. |
| `--id_frame` | `coarse` | 2-stage only; ignored for 3-stage. |
| `--full_eval` | off | Read prompts from a CSV with an `image,prompt` header. |
| `--prompt_dataset_path` | — | Prompt CSV. |
| `--num_prompts` | None | Cap on prompts. |
| `--subset_idx` / `--num_subsets` | 0 / 1 | Shard for distributed runs. |
| `--cfg_scale` | 4.0 | True CFG. Matches our `--guidance_scale 4.0`; see below. |
| `--embedded_guidance` | 4.0 | Ignored by klein-base; see below. |
| `--full_res_check` | off | All-ones mask, port sanity check. |

### Guidance

Two independent knobs, easy to conflate. They are **not** the same thing:

| knob | what it is | on klein-base | ours (`eval/inference.py`) | this port |
|---|---|---|---|---|
| `cfg_scale` | true CFG — a second forward pass on the negative prompt, `nega + s*(posi-nega)` (`base_pipeline.cfg_guided_model_fn`) | **active**; `s != 1.0` costs 2x forwards per step | `cfg_scale=args.guidance_scale`, default **4.0** | `--cfg_scale`, default **4.0** |
| `embedded_guidance` | FLUX.1-dev-style distillation guidance embedding, added to the timestep embedding inside the DiT | **inert** — klein-base 4B/9B are registered with `guidance_embeds: False` (`diffsynth/configs/model_configs.py`) and ship no `guidance_embedder`, so the tensor is built and dropped | pipeline default 4.0, never passed, no effect | `--embedded_guidance` 4.0, no effect |

So `--guidance_scale` in our CLI *is* `cfg_scale` — there is no third quantity. Our runs
are **CFG-guided at 4.0** with an empty negative prompt (`negative_prompt=""`, the pipeline
default; neither script overrides it). klein-**base** is the CFG-trained, non-distilled
variant, so 4.0 is the operating point, not a no-op.

Correction to an earlier version of this doc: it claimed our runs were unguided because
`embedded_guidance` is ignored. That conflated the two knobs — `cfg_scale=4.0` was always
active in our runs.

**This changes previously measured RALU numbers.** Until now `--cfg_scale` defaulted to
1.0 here while ours ran at 4.0, so every FLUX.2 RALU number in this file was measured
unguided against a guided baseline. The default is now 4.0 (`ralu.sh` exports `CFG`), and
the brightness table below must be re-measured before it is quoted anywhere.

The FLUX.1 controls are unaffected and internally consistent: `control_flux1_*.py` use
diffusers' `guidance_scale=3.5`, which on FLUX.1-dev is the *embedded* guidance (dev has
`guidance_embeds: True` and does not run true CFG), and both RALU and the plain reference
in those scripts use the same value.

Cost note: at `cfg_scale != 1.0` every step is two forwards. This cancels in the
compute-matching arithmetic only while both sides use the same `cfg_scale` — another
reason not to let the two defaults drift apart.

### Output layout

Matches `eval/inference.py`, so the metrics scripts work unchanged:

```
output_dir/
  img_0000000000.png        # indexed by global prompt idx
  metadata_00000.csv        # image,prompt — one per shard
  stats_00000.json          # schedule, token counts, timings
  mask.png                  # shard 0 only
```

---

## Files

| File | Role |
|---|---|
| `pipeline_flux2_diffsynth_RALU_fov.py` | **The baseline.** DiffSynth port; subclasses the plain `Flux2ImagePipeline` (not the foveated one), so it inherits no CRPA or Token AE. |
| `RALU_fov_inference_diffsynth.py` | Driver, CLI mirroring `eval/inference.py`. |
| `ralu.sh` | Launcher for single / full-eval / sanity runs. |
| `RALU_NTDM_general.py` | K-stage NT-DM. |
| `test_ntdm_regression.py` | Gate: must reproduce their published constants. |
| `diag_ntdm_k2.py` | Diagnostic for the 2-stage `Z` degeneracy. |
| `rerun_cfg4.sh` | The matched-guidance comparison: 7 configs x 16 prompts at r=0.3. |
| `analyze_cfg4.py` | Brightness / paired ratio / CLIP / schedule table + contact sheet. |
| `make_summary_grid_cfg4.sh` | Lion-prompt grid across radii x {ours, RALU 3/2-stage, Z free/6.5}. |
| `make_grid_fig_cfg4.py` | Assembles that grid into `SUMMARY_cfg4_lion.png`. |
| `pipeline_flux2_klein_RALU_fov.py` | Earlier diffusers port. Superseded by the DiffSynth one; kept only as a cross-framework check. Needs LoRA key renaming and does not share our text-encoder path. |

Use the **DiffSynth** pipeline for all real experiments: it shares our Qwen3 text-encoder
path, VAE wrapper, noise generator, and native LoRA loading, so the only difference from
our runs is the denoising method itself.
