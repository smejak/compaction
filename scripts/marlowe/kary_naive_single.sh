#!/bin/bash
#SBATCH --job-name=lh_kary5_naive
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --error=logs/kary5_naive_%x_%j.err
#SBATCH --output=logs/kary5_naive_%x_%j.out

# Single k=5 stacking probe over patients 01,03,04,05,06.
# naive variant only (no RoPE correction). See contexts/07042026/K5_PROBE_PLAN.md for the
# full plan and contexts/06042026/KARY_STACKING_DEEP_DIVE.md for the
# analysis that motivated this configuration.

set -euo pipefail

# -------- Environment ------------------------------------------------ #
# Marlowe setup: call the env's python directly. Do NOT load the cudatoolkit
# or cudnn modules — they prepend nvhpc's CUDA 12.5 compat libs to
# LD_LIBRARY_PATH and break torch's bundled cu124 runtime with
# `cudaErrorSystemDriverMismatch` (error 803). Torch's pip-installed
# nvidia-cuda-*-cu12 packages already ship the right runtime.
PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

export HF_HOME="${HF_HOME:-/projects/m000120/jsmekal/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs long-health/kary_experiment

# -------- Job info --------------------------------------------------- #
start_time=$(date +%s)
echo "Start:   $(date -d @"$start_time")"
echo "Node:    $(hostname)"
echo "Job:     $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "CWD:     $PWD"
echo "Python:  $PY"
"$PY" --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -------- Run k=5 probe ---------------------------------------------- #
"$PY" -u scripts/run_kary_experiment.py \
    --patients patient_01,patient_03,patient_04,patient_05,patient_06 \
    --variant naive \
    --results-dir long-health/kary_experiment \
    --caches-dir long-health

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Elapsed: %dh %dm %ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
