# ICLR 2027 rebuttal: architecture ablations (handoff)

This doc tells you how to run, monitor, and evaluate the two architecture ablations for the ICLR 2027 rebuttal. Paths and accounts are in [`compute.md`](compute.md). The shared Marlowe procedures are in the global `marlowe` skill.

## 1. What the experiment answers

Reviewers asked two questions (`iclr_2027_rebuttal/reviews.md`):

- 7R3o: "There is no clear ablation of the main components, such as the modified RoPE."
- qyyd: "How does the DiT know which token is high or low resolution? Did you ablate token embeddings?"

We train two LoRAs. Each one changes one thing relative to our foveated LoRA:

| Arm | Change | Answers |
|---|---|---|
| `abl_rope_standard` | No phase-aligned RoPE. One joint attention. Every token keeps its HR-grid RoPE id, and an LR token sits at the top-left `(2i, 2j)` of its 2x2 block. | 7R3o |
| `abl_res_embed` | Phase-aligned RoPE, plus a learned HR/LR embedding: a zero-init `[2, 3072]` table added to each image token after `x_embedder`. | qyyd |
| `ref_rank32fov_10k` | No training. This is our existing LoRA at 10k steps. | the reference row |

Everything else equals the reference run: data, LoRA rank and targets, learning rate, mask sampling, 10k steps.

## 2. The reference run

The reference is wandb `flux2-training/cbx1ng9w` (`rank_32_fov`):

- commit `ce1c906`, host apollo, 6 x A6000 (effective batch 6), 10k steps, 6.1 h, about 19 GB per GPU
- checkpoint: apollo `/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov/step-10000.safetensors`
- flags: `marlowe_scripts/no_tokenAE_lora.sh` plus `--max_training_steps 10000`

Two facts to know before you compare numbers:

1. Table 1 uses `rank_32_fov_more_steps/step-5000`. That run resumed from `rank_32_fov/step-10000`, so Table 1 is a 15k-step model. The ablations compare against the 10k checkpoint, so we evaluate it again (arm `ref_rank32fov_10k`).
2. The paper appendix says "8 H100, effective batch 8". The record shows 6 x A6000 and batch 6. The ablations use 8 processes (batch 8), by decision on 2026-11-06. Set `TRAIN_PROCS=6` to copy the reference exactly.

## 3. Code

| File | Change |
|---|---|
| `DiffSynth-Studio/diffsynth/models/flux2_dit_foveated.py` | `set_rope_mode()`, `enable_resolution_embedding()`, and checkpoint helpers (`foveation_arch_metadata`, `read_foveation_arch`, `configure_foveation_arch_from_checkpoint`) |
| `DiffSynth-Studio/examples/flux2/model_training/train_foveated.py` | flags `--rope_mode {phase_aligned,standard}` and `--resolution_embedding` |
| `DiffSynth-Studio/diffsynth/diffusion/logger.py` | writes the architecture config into the safetensors metadata |
| `eval/inference.py` | reads that metadata after `load_lora`. `--rope_mode` and `--resolution_embedding` apply only to checkpoints without metadata, and a conflict stops the run. |
| `DiffSynth-Studio/diffsynth/pipelines/flux2_image_foveated.py` | skips the unused DeepGaze pass when training uses random or fixed masks. Training is unchanged. |
| `tests/test_foveation_arch_ablation.py` | 14 CPU unit tests |

The checkpoint describes its own architecture. So `eval.sh` needs only `FOV_LORA=<checkpoint>`, and old checkpoints load as the paper model.

To run the tests on apollo:

```bash
cd /home/brianchc/foveated_diffusion-private
CUDA_VISIBLE_DEVICES="" PYTHONPATH=DiffSynth-Studio \
  /home/brianchc/miniconda3/envs/imp_tome/bin/python -m pytest -q tests/test_foveation_arch_ablation.py
```

## 4. Marlowe setup (completed 2026-11-06)

| Item | Location | Verification |
|---|---|---|
| Code | worktree `/users/brianchc/projects/foveated_diffusion_iclr_abl`, detached at the ablation commit | `git log -1` |
| Conda env | `/scratch/m000051/brian/conda_environments/camera_ctrl` | imports torch 2.9.1, peft 0.18, diffsynth from the worktree |
| FLUX.2-klein-base-4B | `/scratch/m000051/brian/models/black-forest-labs/` | all 17 files have the same md5 as apollo `/miele/brian/modelscope` |
| Dataset images | `/scratch/m000051-pm07/brianchc/datasets/Aesthetic-Train-V2/images` (HF rev `3902ab5`) | tar `0000` has the same files and md5 as apollo |
| Split CSVs | same folder, copied from apollo | same md5 (`metadata_train.csv` 95,288 rows, `metadata_test.csv` 10,000 rows) |
| Reference ckpt | `/scratch/m000051-pm07/brianchc/foveated_diffusion/experiments/reference/rank_32_fov/step-10000.safetensors` | same md5 |

Leave the main checkout `/users/brianchc/foveated_diffusion` as it is. It has local commits and uncommitted edits that are not on `origin`.

If you set this up again in a later cycle:

1. On apollo, start the SSH master (see `compute.md`).
2. Download the dataset on Marlowe with `fetch_aesthetic_marlowe.sh`. It needs `tar_sizes.txt` next to it, and it is safe to rerun. Hugging Face gives about 120 MB/s, but apollo to Marlowe over SSH gives only 2–11 MB/s.
3. Copy `metadata_{train,test}.csv` from [`data/aesthetic_train_v2/`](../data/aesthetic_train_v2/) into the dataset root. Do not regenerate them, because the split is ours.
4. Download the FLUX weights with `huggingface_hub.snapshot_download` (`text_encoder`, `transformer`, `vae`, `tokenizer`, `scheduler`, json files), then compare md5 against apollo.
5. Add a git worktree at the ablation commit, and point `REPO` in `ablation_env.sh` to it.

## 5. Launch

Run all commands on the Marlowe login node:

```bash
cd /users/brianchc/projects/foveated_diffusion_iclr_abl/marlowe_scripts
./submit_ablation.sh preflight     # must print "preflight OK" (checks all 106 image tars)
./submit_ablation.sh test-only     # sbatch --test-only for every job shape
./submit_ablation.sh submit-all
```

`submit-all` submits five batch jobs. Each job uses one 8-GPU node and the account `marlowe-m000051-pm07`:

| Job | Starts | Time limit | Output |
|---|---|---|---|
| `fovabl_eval_ref_rank32fov_10k` | when scheduled | 12 h | `results/iclr_abl/ref_rank32fov_10k/` |
| `fovabl_abl_rope_standard` | when scheduled | 12 h | `experiments/abl_rope_standard/step-{1000..10000}.safetensors` |
| `fovabl_abl_res_embed` | when scheduled | 12 h | `experiments/abl_res_embed/step-…` |
| `fovabl_eval_abl_rope_standard` | `afterok` on its training job | 12 h | `results/iclr_abl/abl_rope_standard/` |
| `fovabl_eval_abl_res_embed` | `afterok` on its training job | 12 h | `results/iclr_abl/abl_res_embed/` |

All output paths are under `/scratch/m000051-pm07/brianchc/foveated_diffusion/`. Job IDs are added to `experiments/jobs.tsv`.

A training job exits with success only if `step-10000.safetensors` exists and its metadata shows the right architecture. So `afterok` never evaluates a broken checkpoint. Each training folder also gets `launch_<job>.txt` (commit and full args) and `gpu_<job>.csv` (memory trace). If the compute node has no network, wandb logs offline. Then run `wandb sync <run>/wandb` from the login node.

To run a single job: use `./submit_ablation.sh submit abl_res_embed`, or the 1-GPU preempt `smoke` job (20 steps per arm).

## 6. Monitor

```bash
./submit_ablation.sh status
tail -f /scratch/m000051-pm07/brianchc/foveated_diffusion/logs/fovabl_<run>_<jobid>.out
sreport cluster UserUtilizationByAccount -T cpu,gres/gpu Start=2026-10-30 End=now account=marlowe-m000051-pm07 -t hours
```

Expected time: about 6 h per training job on A6000. H100 is faster, and the validation image every 100 steps adds some time. The eval takes about 3 h per arm. On 2026-11-06 the batch queue had about 200 pending jobs, and Slurm estimated a start on 2026-11-09.

## 7. Evaluation and scoring

The eval jobs generate the Table 1 setting:
- `ours` arm, `circular:0.25`, random centers seed 42 (range ±0.3, clamped), all 10k prompts of `metadata_test.csv`
- base model, 50 steps, CFG 4.0, 1024 px

The prompt order and the centers equal `iclr_2027_base/circular_r0.25_randcenter42`. Each image's prompt and center is in `metadata_*.csv`.

1. Check that the `center_x`/`center_y` columns of `ref_rank32fov_10k` equal Table 1's columns.
2. Score each `results/iclr_abl/<arm>/circular_r0.25_randcenter42/ours` folder with the `/iqa` suite. These are the same metrics as `iclr_2027_base/iqa_comparison.md`. FID uses the test images as reference.
3. Report three rows: `ref_rank32fov_10k`, `abl_rope_standard`, `abl_res_embed`.

## 8. Rebuttal text

`iclr_2027_rebuttal/rebuttal_v5.md` holds the placeholder `[VERIFY: E1 numbers]` (7R3o Q3) and the promise for the architecture ablation (qyyd Q1). Replace them with the three rows when the scores are in.
