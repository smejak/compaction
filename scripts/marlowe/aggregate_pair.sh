#!/bin/bash
#SBATCH --job-name=lh_pair_agg
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --error=logs/pair_agg_%x_%j.err
#SBATCH --output=logs/pair_agg_%x_%j.out

# Aggregate pair-stacked eval results into 7x7 heatmaps + summary.json.
# No GPU needed — this is pure pandas/matplotlib over the per-pair JSON files
# written by scripts/run_pair_experiment.py.
#
# Submit:   sbatch scripts/marlowe/aggregate_pair.sh
# With specific variants: VARIANTS="naive" sbatch --export=ALL,VARIANTS scripts/marlowe/aggregate_pair.sh
#
# Defaults to aggregating both naive and rope_shift.

set -euo pipefail

VARIANTS="${VARIANTS:-naive rope_shift}"

# -------- Environment ------------------------------------------------ #
# Call the env's python directly — see scripts/marlowe/per_patient.sh for why
# the cudatoolkit/cudnn modules are deliberately NOT loaded (driver mismatch).
PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

mkdir -p logs

# -------- Job info --------------------------------------------------- #
echo "Start:    $(date)"
echo "Node:     $(hostname)"
echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Variants: $VARIANTS"
echo "Python:   $PY"

# -------- Run aggregation ------------------------------------------- #
"$PY" -u scripts/aggregate_pair_results.py \
    --results-dir long-health/pair_experiment \
    --variants $VARIANTS

echo "End:      $(date)"
