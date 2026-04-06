#!/bin/bash
#SBATCH --job-name=lh_pair
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-41%8
#SBATCH --error=logs/pair_%x_%A_%a.err
#SBATCH --output=logs/pair_%x_%A_%a.out

# Evaluate one stacked pair of pre-computed LongHealth KV caches.
#
# Array indices 0..41 map to a deterministic ordered-pair list over the 7
# patients in long-health/ (patient_01, patient_03..patient_08), excluding
# self-pairs. See scripts/run_pair_experiment.py:PAIRS.
#
# VARIANT env var selects the stacking strategy:
#   naive       — concat cache tensors without RoPE correction (phase 1)
#   rope_shift  — shift cache_B's keys by original_seq_len_A before concat
#                 (phase 2, requires ROPE_SHIFT_NOTE.md to exist)
#
# Submit phase 1 (default variant=naive):
#   sbatch scripts/marlowe/pair_experiment.sh
#
# Submit phase 2 (after phase 1 completes and ROPE_SHIFT_NOTE.md is written):
#   VARIANT=rope_shift sbatch --export=ALL,VARIANT scripts/marlowe/pair_experiment.sh
#
# Monitor:  squeue -u $USER ; tail -f logs/pair_*.out

set -euo pipefail

VARIANT="${VARIANT:-naive}"

# -------- Environment ------------------------------------------------ #
# Marlowe setup: call the env's python directly. Do NOT load the cudatoolkit
# or cudnn modules — they prepend nvhpc's CUDA 12.5 compat libs to
# LD_LIBRARY_PATH, which shadow the H100 driver (565.57.01 / CUDA 12.7) and
# break torch's bundled cu124 runtime with `cudaErrorSystemDriverMismatch`
# (error 803). Torch's pip-installed nvidia-cuda-*-cu12 packages already
# ship the right runtime, so the system modules are unnecessary.
PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

export HF_HOME="${HF_HOME:-/projects/m000120/jsmekal/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs long-health/pair_experiment

# -------- Job info --------------------------------------------------- #
start_time=$(date +%s)
echo "Start:   $(date -d @"$start_time")"
echo "Node:    $(hostname)"
echo "Job:     $SLURM_JOB_NAME ($SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID])"
echo "Task:    pair_idx=$SLURM_ARRAY_TASK_ID  variant=$VARIANT"
echo "CWD:     $PWD"
echo "Python:  $PY"
"$PY" --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -------- Run single pair ------------------------------------------- #
"$PY" -u scripts/run_pair_experiment.py \
    --pair-idx "$SLURM_ARRAY_TASK_ID" \
    --variant "$VARIANT" \
    --results-dir long-health/pair_experiment \
    --caches-dir long-health

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Elapsed: %dh %dm %ds\n' $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
