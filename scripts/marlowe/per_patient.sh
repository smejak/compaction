#!/bin/bash
#SBATCH --job-name=lh_per_patient
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=20:00:00
#SBATCH --array=1,8-19%8
#SBATCH --error=logs/per_patient_%x_%A_%a.err
#SBATCH --output=logs/per_patient_%x_%A_%a.out

# Run one LongHealth patient through AM-OMP-fast compaction + eval.
#
# Array indices map to patient_XX via idx = XX - 1:
#   idx 1       → patient_02 (hung on Modal, retrying with 20h budget)
#   idx 8..19   → patient_09..patient_20 (never started on Modal)
# Patients 01 and 03..08 are already in long-health/ and are skipped by
# run_per_patient.py via its cache.pt + results.json existence check.
#
# Submit:   sbatch scripts/marlowe/per_patient.sh
# Monitor:  squeue -u $USER ; tail -f logs/per_patient_*.out

set -euo pipefail

# -------- Environment ------------------------------------------------ #
# Edit VENV_PATH (or export it before sbatch) if your venv lives elsewhere.
VENV_PATH="${VENV_PATH:-$HOME/compaction/.venv}"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs long-health

# -------- Job info --------------------------------------------------- #
start_time=$(date +%s)
echo "Start: $(date -d @"$start_time")"
echo "Node:  $(hostname)"
echo "Job:   $SLURM_JOB_NAME ($SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID])"
echo "Task:  patient_idx=$SLURM_ARRAY_TASK_ID"
echo "CWD:   $PWD"
echo "Venv:  $VENV_PATH"
echo "Python: $(python --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -------- Run single patient ---------------------------------------- #
python -u scripts/run_per_patient.py \
    --patient-idx "$SLURM_ARRAY_TASK_ID" \
    --results-dir long-health

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Elapsed: %dh %dm %ds\n' $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
