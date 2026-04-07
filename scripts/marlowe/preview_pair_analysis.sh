#!/bin/bash
#SBATCH --job-name=lh_pair_drill
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --error=logs/pair_drill_%x_%j.err
#SBATCH --output=logs/pair_drill_%x_%j.out

# Drill-down analyses for the pair-stacked KV cache experiment.
# Runs scripts/preview_pair_analysis.py over the canonical phase-2 results.
# CPU-only — pure-python json + numpy + matplotlib over ~760 small files.
#
# Submit: sbatch scripts/marlowe/preview_pair_analysis.sh

set -euo pipefail

# -------- Environment ------------------------------------------------ #
# Direct env python — see scripts/marlowe/per_patient.sh for why the
# cudatoolkit/cudnn modules are deliberately NOT loaded.
PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

mkdir -p logs

# -------- Job info --------------------------------------------------- #
echo "Start:  $(date)"
echo "Node:   $(hostname)"
echo "Job:    $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Python: $PY"

# -------- Run drill-downs ------------------------------------------- #
"$PY" -u scripts/preview_pair_analysis.py \
    --results-dir long-health/pair_experiment \
    --output-dir  long-health/pair_experiment \
    --variants    naive rope_shift

echo "End:    $(date)"
