# Marlowe compute: foveated diffusion

Shared procedures are in the global `marlowe` skill. This file holds the project values.

## Cycle pm07 (from 2026-10-30 10:00 PT)

| Item | Value |
|---|---|
| Login | `brianchc@cs.marlowe.stanford.edu` |
| SSH master on apollo | socket `~/.ssh/marlowe-tmux`, tmux session `marlowe-connection` |
| Batch account | `marlowe-m000051-pm07`, partition `batch` (whole nodes, at most 48 h) |
| Preempt account | `marlowe-m000051`, partition `preempt` (jobs smaller than a node) |
| Allocation | 23,500 GPU-hours. Usage: `sreport cluster UserUtilizationByAccount -T cpu,gres/gpu Start=2026-10-30 End=now account=marlowe-m000051-pm07 -t hours` |
| Cycle scratch | `/scratch/m000051-pm07/brianchc` (purged after the cycle) |
| Base scratch | `/scratch/m000051/brian` (conda envs, model weights) |
| Repo | worktree `/users/brianchc/projects/foveated_diffusion_iclr_abl` (detached at the ablation commit). The main checkout `/users/brianchc/foveated_diffusion` has local commits and edits, so do not pull there. |
| Slurm | `module load slurm/slurm/25.05.2` |
| Conda env | `/scratch/m000051/brian/conda_environments/camera_ctrl` |
| Models | `DIFFSYNTH_MODEL_BASE_PATH=/scratch/m000051/brian/models` |
| Dataset | `/scratch/m000051-pm07/brianchc/datasets/Aesthetic-Train-V2` (split CSVs copied from apollo `/local/brian/Aesthetic-Train-V2`) |
| Experiments | `/scratch/m000051-pm07/brianchc/foveated_diffusion/experiments/<run>` |

All paths are variables in `ablation_env.sh`. If a path changes, edit it there.

## ICLR 2027 rebuttal ablations

The reference is wandb `flux2-training/cbx1ng9w` (`rank_32_fov`): commit `ce1c906`, apollo, 6 x A6000, effective batch 6, 10k steps, 6.1 h. The checkpoint is `/local/brian/foveated_diffusion/models/eccv2026_final/rank_32_fov/step-10000.safetensors` on apollo.

| Run | Change |
|---|---|
| `abl_rope_standard` | `--rope_mode standard` |
| `abl_res_embed` | `--resolution_embedding` |

```bash
cd /users/brianchc/projects/foveated_diffusion_iclr_abl/marlowe_scripts
./submit_ablation.sh preflight     # paths, env, data, models
./submit_ablation.sh test-only     # sbatch --test-only
./submit_ablation.sh smoke         # 1 GPU, preempt, 20 steps per run
./submit_ablation.sh submit        # one 8-GPU batch node per run, 6 processes
./submit_ablation.sh status
```

The arch config is in the checkpoint's safetensors metadata, and `eval/inference.py` reads it from there. For evaluation, set `FOV_LORA=<run>/step-10000.safetensors` and use the Table 1 setting: `METHODS=ours MASK_CONFIGS=circular:0.25 RANDOM_MASK_CENTER=true`, all 10k prompts from `metadata_test.csv`.
