#!/bin/bash
#SBATCH --job-name=lh_pair
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --error=logs/pair_%x_%A_%a.err
#SBATCH --output=logs/pair_%x_%A_%a.out

# Evaluate stacked pairs of pre-computed LongHealth KV caches.
#
# Each array task processes a CHUNK of pair indices (default: 6 pairs/task,
# 7 tasks → all 42 pairs) — chunking exists because the marlowe-m000120-pm05
# account has a per-account submit limit of 32 jobs in the medium QOS, and
# 42 + 42 individual array tasks for both variants would routinely exceed
# that limit when other users have jobs queued. With 6 pairs/task we land
# at 7 tasks per variant (14 total) — comfortable headroom.
#
# Pair indices 0..41 map to a deterministic ordered-pair list over the 7
# patients in long-health/ (patient_01, patient_03..patient_08), excluding
# self-pairs. See scripts/run_pair_experiment.py:PAIRS.
#
# VARIANT env var selects the stacking strategy:
#   naive       — concat cache tensors without RoPE correction (phase 1)
#   rope_shift  — shift cache_B's keys by original_seq_len_A before concat
#                 (phase 2)
#
# PAIRS_PER_TASK env var (default 6) controls the chunk size; if you change
# it, also adjust the #SBATCH --array directive above so the last task lands
# at index ceil(42 / PAIRS_PER_TASK) - 1.
#
# Submit naive (default):
#   sbatch scripts/marlowe/pair_experiment.sh
#
# Submit rope_shift in parallel:
#   VARIANT=rope_shift sbatch --export=ALL,VARIANT scripts/marlowe/pair_experiment.sh
#
# Monitor:  squeue -u $USER ; tail -f logs/pair_*.out

set -uo pipefail   # NB: no `-e` because we want the inner pair loop to keep
                   # going past a single failed pair and report at the end.

VARIANT="${VARIANT:-naive}"
PAIRS_PER_TASK="${PAIRS_PER_TASK:-6}"
TOTAL_PAIRS=42

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

# -------- Compute pair range for this array task -------------------- #
START=$((SLURM_ARRAY_TASK_ID * PAIRS_PER_TASK))
END=$((START + PAIRS_PER_TASK - 1))
if [ $END -ge $TOTAL_PAIRS ]; then
    END=$((TOTAL_PAIRS - 1))
fi

# -------- Job info --------------------------------------------------- #
start_time=$(date +%s)
echo "Start:        $(date -d @"$start_time")"
echo "Node:         $(hostname)"
echo "Job:          $SLURM_JOB_NAME ($SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID])"
echo "Task chunk:   pair_idx=[$START..$END]  variant=$VARIANT"
echo "Pairs/task:   $PAIRS_PER_TASK"
echo "CWD:          $PWD"
echo "Python:       $PY"
"$PY" --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -------- Run all pairs in this chunk ------------------------------- #
failed=()
for pair_idx in $(seq $START $END); do
    echo
    echo "================================================================"
    echo "  Starting pair_idx=$pair_idx (variant=$VARIANT)"
    echo "================================================================"
    pair_start=$(date +%s)
    if "$PY" -u scripts/run_pair_experiment.py \
            --pair-idx "$pair_idx" \
            --variant "$VARIANT" \
            --results-dir long-health/pair_experiment \
            --caches-dir long-health; then
        pair_end=$(date +%s)
        pair_elapsed=$((pair_end - pair_start))
        printf "  pair_idx=%d OK in %dh %dm %ds\n" "$pair_idx" \
            $((pair_elapsed / 3600)) $(((pair_elapsed % 3600) / 60)) $((pair_elapsed % 60))
    else
        rc=$?
        echo "  pair_idx=$pair_idx FAILED with exit $rc"
        failed+=("$pair_idx")
    fi
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf '\nTask total elapsed: %dh %dm %ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED PAIRS in this task: ${failed[*]}"
    exit 1
fi
echo "All pairs in chunk [$START..$END] OK"
