#!/bin/bash
#SBATCH --job-name=lh_kary_per_pt
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --error=logs/kary_per_pt_%x_%j.err
#SBATCH --output=logs/kary_per_pt_%x_%j.out

# Per-patient accuracy breakdown across k=2..5 stacking depths.
# Produces three figures per sbatch:
#   per_patient_accuracy.{pdf,png}           — rope_shift single panel
#   per_patient_accuracy_naive.{pdf,png}     — naive single panel
#   per_patient_accuracy_combined.{pdf,png}  — rope_shift + naive side by side
# CPU-only — pure-python json + matplotlib over a handful of small files.
# Idempotent: re-run as new k results land to refresh all three figures.

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

# -------- Plot: single-panel per variant, then combined ------------- #
"$PY" -u scripts/plot_kary_per_patient.py \
    --root . \
    --out-dir long-health/kary_experiment/figures \
    --variant rope_shift

"$PY" -u scripts/plot_kary_per_patient.py \
    --root . \
    --out-dir long-health/kary_experiment/figures \
    --variant naive

"$PY" -u scripts/plot_kary_per_patient.py \
    --root . \
    --out-dir long-health/kary_experiment/figures \
    --combined

echo "End:    $(date)"
