#!/usr/bin/bash

#SBATCH -N 1
#SBATCH -G 8
#SBATCH --cpus-per-task 64
#SBATCH --job-name=
#SBATCH --mem=1024G
#SBATCH -A marlowe-m000051
#SBATCH -p preempt
#SBATCH --time=12:00:00
#SBATCH --output=cluster_logs/%x-%j/%j.out
#SBATCH --error=cluster_logs/%x-%j/%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=brianchc@stanford.edu

# Load conda environment
source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate /scratch/m000051/brian/conda_environments/camera_ctrl

# move triton cache off NFS system
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

# Set job-specific MPS directories (for preventing MPS bug)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export DIFFSYNTH_MODEL_BASE_PATH="/scratch/m000051-pm05/brianchc_2/models"

bash foveated_diffusion/DiffSynth-Studio/examples/flux2/model_training/flux2-fov-marlowe.sh


#!/usr/bin/bash

#SBATCH -N 1
#SBATCH -G 8
#SBATCH --cpus-per-task 64
#SBATCH --job-name=job_name
#SBATCH --mem=1024G
#SBATCH -A marlowe-m000051-pm05
#SBATCH -p batch
#SBATCH --time=12:00:00
#SBATCH --output=cluster_logs/%x-%j/%j.out
#SBATCH --error=cluster_logs/%x-%j/%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=brianchc@stanford.edu

job_name=$1

# Load conda environment
source /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh
conda activate /scratch/m000051/brian/conda_environments/camera_ctrl

# move triton cache off NFS system
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

# Set job-specific MPS directories (for preventing MPS bug)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export DIFFSYNTH_MODEL_BASE_PATH="/scratch/m000051-pm05/brianchc_2/models"

bash foveated_diffusion/marlowe_scripts/${job_name}.sh
