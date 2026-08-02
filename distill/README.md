# Foveated DMD distillation

Few-step distillation of the foveated FLUX.2 model, using [NVIDIA FastGen](https://github.com/NVlabs/FastGen)
for the distillation maths and DiffSynth for the foveated model.

Nothing here modifies DiffSynth-Studio or FastGen; both are imported.

## Why this works at all

The foveated DiT is a plain flow-matching velocity field over a *reduced* mixed-resolution
token sequence `[B, L, 128]`, with `L = m + (h·w − m)/f²`. Every operation DMD needs
(`forward_process`, `flow_to_x0`, the VSD loss) is elementwise or reduces over
`range(1, ndim)`, so the objective transfers verbatim — only the network wrapper is new.

Verified time convention:

| | forward process | velocity | x₀ recovery |
|---|---|---|---|
| FastGen `rf` | `(1−t)x₀ + t·ε` | `ε − x₀` | `x_t − t·v` |
| DiffSynth FLUX.2 | `(1−σ)z₀ + σ·ε` | `ε − z₀` | `z_t − σ·v` |

so `t ≡ σ ≡ timestep/1000`. The DiT rescales by 1000 internally.

### The timestep cast chain matters

`Flux2FoveatedNet.forward` deliberately reproduces the reference pipeline's *casts*, not just
its arithmetic:

```python
timestep_dit = (t.to(torch.float32) * 1000).to(self.torch_dtype) / 1000
```

The pipeline builds `timestep_val = (sigma*1000).to(bf16)` — quantising on the bf16 grid at
magnitude ~1000, where the spacing is ~2 — then `model_fn` divides by 1000 and the DiT
multiplies by 1000 again. Casting `t ∈ [0,1]` to bf16 instead quantises on a 0.0039 grid: a
*finer* discretisation of the same schedule, but a different one. Over 50 Euler steps that
compounds, ~3× worse under CFG w=4 since `4·v_pos − 3·v_neg` amplifies any deviation.

`verify_teacher.py` vs the reference pipeline, 50 steps, f=2 (pixel MAE / max abs, out of 255):

| | before | after |
|---|---|---|
| cfg 1 | 0.41 / 56 | **0.22 / 8** |
| cfg 4 | 1.16 / 103 | **0.78 / 81** |

The remaining residual is **not** a logic difference. `scheduler.step` multiplies the bf16
latents by a **CPU** 0-dim sigma, which PyTorch's promotion rules treat as a scalar, so the
reference keeps the Euler accumulation in **bf16**; our CUDA 0-dim sigma promotes to
**float32**. The reference accumulates less precisely than we do. At cfg 4 that residual is
amplified ~3.6× by the guidance combination, which accounts for the whole 0.78. Deliberately
not matched — losing precision to imitate the reference would be the wrong trade.

Two red herrings, recorded so they aren't re-investigated:
* **Not the `t_max` clamp.** `bf16(0.999) == 1.0` (grid spacing 0.0039 near 1.0), so clamping
  `sigmas[0]` was numerically invisible — removing it changed the output not at all.
* **Single-step tensor comparisons can hide this.** `verify_cfg_step.py` handed both paths the
  same float32 timestep and so reported `0.000e+00` on every quantity, including the CFG
  combination, while the 50-step image comparison still diverged. Trust `verify_teacher.py`
  for end-to-end fidelity.

## Files

| file | role |
|---|---|
| `fov_net.py` | `Flux2FoveatedNet(FastGenNetwork)` — the DiT as a flow network over reduced tokens; mask sampling, foveated encode/decode, student & teacher samplers |
| `dmd.py` | `FoveatedDMD` — generator (VSD) and critic (DSM) losses, imported from `fastgen.methods.common_loss` |
| `data.py` | Aesthetic-Train-V2 → `{image, prompt}` |
| `train_dmd.py` | accelerate training loop, wandb logging, validation, `--smoke_test` |
| `verify_teacher.py` | reproduces `Flux2FoveatedImagePipeline.__call__` through our wrapper |

## One DiT, three roles

Student, critic (DMD fake score) and teacher are three views of a **single** 6.1 GB
transformer instead of three copies:

* **teacher** = base weights with the foveated SFT LoRA *fused in* (`GeneralLoRALoader.fuse_lora_to_base_model`), adapters disabled
* **student** = base + PEFT adapter `student`
* **critic** = base + PEFT adapter `critic`

`Flux2FoveatedNet.set_role()` flips `BaseTunerLayer.enable_adapters`/`set_adapter`, which
also isolates `requires_grad` to the active adapter.

⚠️ **The trainable role must stay active through `backward()`.** Activation checkpointing
replays blocks during backward, so recomputing under a different adapter raises
`CheckpointError: A different number of tensors was saved`. Each loss leaves the trainable
role active and `train_dmd.py` asserts it before `backward`.

## Running

```bash
# check occupancy FIRST — this box is shared
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

GPU_IDS=3,4,5,6 bash distill/run_dmd_smoke.sh   # assertions + a few steps + validation
GPU_IDS=3,4,5,6 bash distill/run_dmd.sh         # full run
```

`--gpu_ids` must be passed to `accelerate launch`; setting `CUDA_VISIBLE_DEVICES` does not
work because accelerate overwrites it (and `~/.cache/huggingface/accelerate/default_config.yaml`
pins `gpu_ids: 0,1,2,3`).

Measured: **0.73 steps/s on 4×A6000** at 1024², f=2 (≈1.4 s/step, batch 1/GPU), ~26 GB/GPU.

## Guidance: klein-base is NOT guidance-distilled

The FLUX.2-klein-base-4B transformer checkpoint contains only
`time_guidance_embed.timestep_embedder.*` — **there is no `guidance_embedder` tensor**. So
`guidance_embeds=False` at load, `Flux2TimestepGuidanceEmbeddings.guidance_embedder is None`,
and its forward returns `timesteps_emb` while *discarding* the `guidance` argument
(`flux2_dit.py:851-857`).

Consequences:

* `--embedded_guidance` is **inert** on this model (as are the `--embedded_guidance 1.0` in
  the SFT scripts and the pipeline's `4.0` default). Kept only for models that do have one.
* All prompt adherence comes from **true CFG** — two forwards, cond and uncond. Hence
  `--teacher_cfg_scale 4.0` by default, matching the SFT validation's `--cfg_scale 4.0`.
* Only the **teacher** uses CFG. The student and critic run single-forward, because
  internalising the guidance is the point of the distillation. `teacher_cfg_scale=1.0` would
  distil an *unguided* model and look like "DMD doesn't work".
* CFG in velocity space (what `cfg_guided_model_fn` does) and in x₀ space (what DMD/FastGen
  does) are exactly equivalent here, since `x₀ = x_t − t·v` is affine in `v` at fixed `t`.
* Cost: one extra teacher forward, on generator iterations only (1 in
  `--critic_update_freq`), so ≈ +10 % step time.

## Validation

Every `--validation_steps`, two wandb panels:

**`validation/center_quality`** — centred mask, three ways from *identical* noise:
teacher @ `--teacher_val_steps` (50, the quality ceiling), teacher @ `--student_steps`
(4, the naive few-step baseline DMD must beat), and student @ `--student_steps` (4, distilled).

**`validation/student_masks`** — the student at the four mask positions
(left/right/top/bottom, r=0.25, as in `WandbModelLogger`).

Plus `validation/foveation_masks` once at step 0. Images also land in
`<output>/validation/step-N/`.

## Training dynamics: what to watch, and one confirmed failure mode

The VSD loss is **not** a quality metric. FastGen computes
`loss = 0.5·mse(gen, gen − vsd_grad) = 0.5·mean(vsd_grad²)`, so it is a *gradient magnitude*
proxy: it rises both when the student gets worse and when the critic is still learning. Judge
progress from the images and these diagnostics instead:

| metric | healthy | bad |
|---|---|---|
| `grad_norm/student` | fluctuates around a level | grows monotonically → student walking on noise |
| `diag/critic_teacher_l1` | settles | swings with no trend → critic has not converged to the student's score, so the DMD gradient *is* noise |
| `diag/gen_x0_std` vs `diag/real_x0_std` | gap stays roughly constant (E[x₀\|x_t] is legitimately lower-variance than x₀) | gen collapses toward 0 → mode collapse / blur |

**Confirmed failure (run `uiarkwif`).** With effective batch 4, β₁=0, and 100 warmup steps, the
student was a featureless blur by step 500 — *worse* than step 0, where the zero-init adapter
makes it identical to the teacher. `grad_norm/student` had grown 11× monotonically
(0.0036 → 0.041, never near the 1.0 clip) while `critic_teacher_l1` swung 0.05–0.32 without
converging. Root cause: an undertrained critic makes `(critic − teacher)` mostly noise, and
β₁=0 makes each Adam step ≈ `lr·sign(g)`, so the student took full-size steps along it. The
trigger was a bug — `--gradient_accumulation_steps` was accepted but never applied (the loop
stepped the optimizer every iteration and never called `accelerator.accumulate`), pinning the
effective batch at 4 with one mask and one `t` per sample.

Fix: real accumulation (by hand — the optimizer, trainable set and autograd subgraph all
switch between micro-batch groups), warmup 500, β₁=0.9. Note β₁=0.9 already averages over a
~10-step window and so partly substitutes for accumulation at no wall-clock cost; accumulation
costs a full multiple of wall-clock per generator update, so `accum 2` is usually the better
balance once training is stable.

Because warmup is counted in *optimizer* steps, validations before `--critic_warmup_steps`
show a student still identical to the teacher; they confirm plumbing, not learning.

## Knobs worth sweeping

* `--student_t_mode` (default `uniform` → `[0.999, .749, .499, .250, 0]`). `flux2` reuses the
  native shifted schedule, but FLUX.2's empirical shift is so top-heavy that its 4-step
  schedule is `[1.0, .967, .908, .767]` — the student would never see an input below t=0.77.
* `--critic_update_freq` (5), `--critic_warmup_steps` (100). With a zero-init critic the VSD
  gradient is *exactly* zero, so the warmup is not optional.
* `--student_lr` / `--critic_lr` (5e-6, β₁=0 as in DMD2).
* `--teacher_cfg_scale` (1.0 — klein-base uses embedded guidance, so CFG is off by default
  and the teacher costs one forward).

## Not implemented

The DMD2 **GAN term**. Every head in `fastgen/networks/discriminators.py` is Conv2d/Conv3d
over spatial feature maps and FastGen's Flux `classify_forward` reshapes features with
`spatial_size = int(seq_len**0.5)`; L is not a perfect square and the mixed-resolution
sequence is not a grid. Pure DMD (VSD) is unaffected. For the GAN term,
`_build_dit_attention_discriminator_head` is the sequence-friendly starting point.

## Setup notes

* FastGen on `PYTHONPATH`, **not** `pip install -e .` — its pins (`transformers==4.49.0`,
  `diffusers==0.35.1`, `numpy<2`) would downgrade the env and break the Qwen3 text encoder.
* Only extra dependency: `boto3` (installed; FastGen imports it via `utils/io_utils.py`).
