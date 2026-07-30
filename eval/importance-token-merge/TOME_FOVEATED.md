# ToMe-2x2 foveated baseline

Importance-Based Token Merging ([ICCV 2025](https://arxiv.org/abs/2411.16720)) ported to
FLUX.2-klein and driven by **our foveation mask** instead of its error-map token selection,
so it can be compared against foveated diffusion at a matched token budget.

Upstream files (`tomesd/`, `pipelines/`, `pipe_sd.py`, `pipe_pixart.py`, `demo_*.py`,
`README.md`) are **unmodified**. Everything here is additive, and nothing in
`DiffSynth-Studio/` is edited or mutated.

```
flux2_tome/
  merge_2x2.py         mask-driven grouping + merge/unmerge
  patch_flux2.py       Flux2DiT / block subclasses, apply_patch / remove_patch
  test_merge_2x2.py    correctness gate for the grouping
  test_patch_flux2.py  structural gate for the patch
tome_fov_inference.py  driver (same CLI shape and output layout as eval/inference.py)
run_tome.sh            parameterized launcher: single | eval | sweep | compare
bench_tome_vs_ours.py  isolated latency benchmark (one arm per process)
bench_all.sh           runs all arms sequentially on one GPU + prints the table
verify_config_parity.py  checks cfg/guidance/schedule/text-embed parity vs our runs
RUN.md                 how to launch, full knob table
tome_compare_ours.sh   fixed-config single-prompt runs (superseded by `run_tome.sh compare`)
tome_eval.sh           fixed-config 8-way sharded eval (superseded by `run_tome.sh eval`)
```

---

## What was changed, and why

The port keeps ToMe's mechanism intact. Exactly one methodological change:

| | Original ToMe | Here |
|---|---|---|
| Which tokens merge into which | top-`M` by `\|ε_text − ε_uncond\|` heat map; src→nearest dst by `-cdist` | the `lr_factor²` tokens of each peripheral block merge into that block's top-left token; foveal tokens never merge |

Everything else is theirs: merge is the **mean** over each group (`scatter_reduce(reduce="mean")`,
which includes the dst token), unmerge is a **copy** of the group value back to every member,
tokens are merged *around* the sublayer and restored immediately after, and text tokens are
never merged.

The grouping change is what makes the comparison meaningful. Their DiT variant already merges
by spatial proximity — `patch_pixart.py` replaces the similarity metric with `(i,j)` coordinates
and scores by `-cdist` when `original_tome=False` — so local block merging is the degenerate
case of their own DiT rule under a hard binary importance mask. Feeding our mask in as their
heat map directly does *not* work: their dst set is drawn from the **important** tokens, so
every peripheral token would merge into a foveal one, collapsing the periphery onto the fovea
boundary rather than downsampling it.

Port-only differences, forced by the base model and none of them methodological:

- **RoPE gather.** SD and PixArt add position information at the input, so ToMe can change the
  sequence length inside a block freely. FLUX.2 builds `concat_rotary_emb` once at full
  `[txt + img]` length in `Flux2DiT.forward` and applies it *inside* attention, so a merged
  sequence needs a correspondingly gathered rope table. This is the only reason `Flux2DiT.forward`
  is overridden at all. Merged groups inherit their top-left token's position id — the same
  choice `_prepare_latent_image_ids_foveation` makes for a low-res block.
- **4-column `(T,H,W,L)` position ids** instead of SD/PixArt's 2D grid.
- **Joint text+image sequence.** Double blocks keep the streams separate; single blocks arrive
  pre-concatenated, so only the slice after `text_seq_len` is merged.
- **No `max_downsample` / `downsample` machinery.** That exists to skip ToMe on the UNet's
  downsampled stages; a DiT has one resolution.
- **No `B == 2·B0` CFG assumption.** Their matching functions require a paired uncond/cond batch
  because the heat map is derived from the CFG difference. Our grouping comes from the mask, so
  the constraint is dropped.
- **Grouping computed once, not per block.** It depends only on the mask, not on features.

---

## `merge_mlp` means less than it does upstream

Upstream, `merge_attn` and `merge_mlp` are independent. On FLUX.2 they are not.

FLUX.2-klein-base-4B is 8 double-stream blocks + **48 single-stream blocks**. The single blocks
are *parallel* blocks: `to_qkv_mlp_proj` produces the attention QKV and the MLP input in one
fused projection, then a single `to_out` consumes both. Feeding merged tokens into that block
necessarily merges the MLP as well — there is no way to merge attention but not the MLP without
running the block twice.

So `--merge_mlp` only controls the 8 double blocks, and the two configurations are identical in
48 of 56 blocks. Both are provided and reported, but the gap between them is small by
construction, and **`merge_mlp=false` is not a "ToMe with no MLP savings" configuration** — most
of this model's MLP compute is inside parallel blocks and is merged either way. Say this in the
paper rather than presenting the two numbers as an attn-vs-attn+MLP ablation.

---

## Matched token budget

The block-level high/low decision replicates the pipeline exactly: a block is high-res iff
*any* of its tokens is masked. `test_merge_2x2.py` asserts that our surviving-token set equals
the set `Flux2FoveatedImagePipeline._prepare_latent_image_ids_foveation` produces for the same
mask — same count *and* same position ids — including for a deliberately block-unaligned mask.

At 1024², circular `r=0.3`, `lr_factor=2`: **4096 → 1468 image tokens (35.8% kept)**, identical
to our foveated run.

Note that `create_foveation_mask` builds the mask at `(H/16)/lr_factor` and nearest-upsamples,
so in practice every mask is already block-aligned; the unaligned case is tested only for safety.

---

## Correctness gates

```bash
PYTHONPATH=<repo>/DiffSynth-Studio python -m flux2_tome.test_merge_2x2
PYTHONPATH=<repo>/DiffSynth-Studio python -m flux2_tome.test_patch_flux2
```

`test_merge_2x2.py` (CPU, no model):
- all-high-res mask → merge and unmerge are the identity
- all-low-res mask → merge equals `avg_pool2d(2)`, unmerge copies the block mean back
- mixed mask → foveal tokens are bit-exact through merge→unmerge
- token count and position ids match the foveated pipeline

`test_patch_flux2.py` (tiny random DiT, CPU):
- `enabled=False` is bit-exact against the unpatched model, and `remove_patch` restores it
- an all-high-res mask reproduces the unpatched model through the merging code path
- attention provably sees `text_seq_len + num_merged` tokens while the output stays full length
- `merge_mlp` changes the result

The `--enable_tome false` run in `tome_compare_ours.sh` is the same identity control end to end:
it exercises the patched classes but with merging off, and must match the `high_res` baseline.

---

## What the baseline actually produces

Unlike RALU, ToMe has no failure mode here — it stays in a valid full-resolution frame the whole
time, since merging is undone before the residual. Output is coherent, correctly composed, with
a sharp fovea and a visibly blurred periphery.

That blur is the substantive finding, and it is the comparison's point: ToMe's periphery is a
box filter (mean-pool merge, copy unmerge), whereas ours is a learned TokenAE with CRPA
attention giving each resolution its own RoPE frame. Same token budget, different reconstruction.

**Modality caveat for the writeup:** ToMe's output is uniform full resolution, ours is
mixed-resolution. Metrics are computed on the full image for both (per the eval config), so
report tokens and latency alongside quality, as the RALU writeup does.

---

## Running

See [`RUN.md`](RUN.md) for the launcher and the full knob table.

Environment: `imp_tome`. `PYTHONPATH` must point at this repo's `DiffSynth-Studio`, because
`imp_tome` otherwise resolves `diffsynth` to `/home/brianchc/DiffSynth-Studio`, a different
checkout. Both scripts set this.

```bash
cd eval/importance-token-merge
./tome_compare_ours.sh          # qualitative: tome_attn, tome_attn_mlp, tome_off, ours
./tome_eval.sh false            # full eval, attention-only in double blocks
./tome_eval.sh true             # full eval, image MLP merged in double blocks too
```

Then the existing metric scripts, unchanged:

```bash
cd ..
python compute_clip_score.py --image_dir ../results/tome_eval_mlp_false
python compute_hpsv2_score.py --image_dir ../results/tome_eval_mlp_false
```

Baseline weights are the `finetuned_no_fov` LoRA (`step-15000.safetensors`), matching the
`naive_mixed_res` and `high_res` arms of `eval/ralu_compare_ours.sh`, so base-model quality is
controlled across all baselines.

---

## Conditioning parity with our foveated runs

`Flux2ImagePipeline.__call__` and `Flux2FoveatedImagePipeline.__call__` are separate
implementations with separate defaults, so parity cannot be assumed. `verify_config_parity.py`
intercepts the real DiT calls in both and compares what actually arrives:

```bash
python verify_config_parity.py --arm tome --out parity/tome.json
python verify_config_parity.py --arm ours --out parity/ours.json
python verify_config_parity.py --compare parity/     # exit 0 == all match
```

Verified identical (results in `results/tome_config_parity/`):

| | ToMe | ours |
|---|---|---|
| `cfg_scale` | 4.0 | 4.0 |
| `embedded_guidance` | 4.0 | 4.0 |
| DiT calls / step | 2.0 (true CFG) | 2.0 (true CFG) |
| timesteps | `1000.0, 967.384, 908.144, 767.2` | identical |
| sigmas | `1.0, 0.967384, 0.908144, 0.7672` | identical |
| prompt embeds | `[1,512,7680]`, checksums match on both CFG branches | identical |

Three things that could have silently diverged, and did not:

- **`embedded_guidance` is distinct from `cfg_scale` in FLUX.2.** Neither `eval/inference.py`
  nor our driver passes it, so both fall through to the default — which is `4.0` in *both*
  `__call__` signatures. Had those defaults differed, the baseline would have run at a
  different distillation guidance with no error.
- **The sigma schedule uses the full-resolution token count in both.** Both call
  `set_timesteps(..., dynamic_shift_len=height//16*width//16)` — i.e. 4096, *not* the reduced
  count. This is the trap RALU hit, where a changed sequence length forced its shift constants
  to be re-solved. Here the foveated pipeline does not rescale, so the schedules coincide.
- **Text conditioning is the same encoder and the same embedding.** The Mistral
  `PromptEmbedder` has a different `system_message` default between the two files (a real
  system prompt in the base, `None` in the foveated one), but it is dead code for this model:
  it short-circuits with `if pipe.text_encoder_qwen3 is not None: return {}`, and FLUX.2-klein
  loads `z_image_text_encoder`. The Qwen3 embedder that actually runs is AST-identical between
  the two files, and the measured embedding checksums match exactly.

The one intended difference is the image sequence length reaching the DiT: ToMe passes 4096
and merges inside each block, ours passes 1468 directly. That is the method under comparison,
not a configuration drift.

---

## Latency: ToMe and ours cost the same

Isolated benchmark (`./bench_all.sh`, sequential arms on one idle RTX A6000, 1 warmup + 3 reps,
50 steps, 1024², circular `r=0.3`). `DiT s` = summed CUDA-synced time over all 100 DiT forwards.

| Arm | Image tokens | DiT s | Speedup |
|---|---|---|---|
| full resolution | 4096 | 40.06 | 1.00x |
| ToMe-2x2, `merge_mlp=false` | 1468 | 20.43 | 1.96x |
| ToMe-2x2, `merge_mlp=true` | 1468 | 18.72 | 2.14x |
| **ours** | 1468 | 18.89 | 2.12x |
| naive mixed-res | 1468 | 18.85 | 2.13x |

Ours and ToMe-`merge_mlp=true` differ by 0.9% — ToMe is marginally *faster*. This is not what
you would predict from the fact that ToMe keeps a full-length residual stream while ours runs
the DiT short throughout. The reason is the parallel-block structure above: with 48 of 56
blocks merging attention and MLP together, essentially all matmul work already happens at 1468
tokens, and what remains at 4096 (LayerNorm, residual add, merge/unmerge gather) is
memory-bound and cheap. The 1.7s that `merge_mlp=false` costs is precisely the 8 double blocks
running their feed-forward at full length.

**This is the right framing for the paper.** Matched token budget *and* matched wall clock means
the periphery-quality gap is attributable to the reconstruction — learned TokenAE with CRPA
attention vs. mean-pool merge and copy unmerge — with no compute confound in either direction.
Report ToMe at `merge_mlp=true`; `merge_mlp=false` only handicaps the baseline.

Raw results: `results/bench_tome_vs_ours/` (`report.txt`, per-arm JSON and logs).
