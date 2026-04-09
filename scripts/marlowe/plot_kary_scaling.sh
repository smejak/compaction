#!/bin/bash
#SBATCH --job-name=lh_kary_plot
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --error=logs/kary_plot_%x_%j.err
#SBATCH --output=logs/kary_plot_%x_%j.out

# Plot naive vs rope_shift accuracy as a function of k (number of stacked
# patients) on the cumulative-prefix subset patient_01, _03, _04, _05, _06.
# CPU-only — pure-python json + matplotlib over a handful of small files.
# Idempotent: re-run as new k results land to refresh the figure.

set -euo pipefail

# -------- Environment ------------------------------------------------ #
# Direct env python — see scripts/marlowe/per_patient.sh for why the
# cudatoolkit/cudnn modules are deliberately NOT loaded.
PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

mkdir -p logs long-health/kary_experiment/figures

# -------- Job info --------------------------------------------------- #
echo "Start:  $(date)"
echo "Node:   $(hostname)"
echo "Job:    $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Python: $PY"

# -------- Plot ------------------------------------------------------- #
"$PY" -u scripts/plot_kary_scaling.py \
    --root . \
    --out-dir long-health/kary_experiment/figures

echo "End:    $(date)"
