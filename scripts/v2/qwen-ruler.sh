#!/bin/bash
#SBATCH --job-name=q-ruler
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-50
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --requeue

# -------- Environment ------------------------------------------------ #
export HOME=/data/cl/u/adamz/
source ~/.bashrc
cd ~/compaction-release
conda activate compaction

MODEL=Qwen/Qwen3-4B
budget_path="head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

log_dir=logs/qa_evaluation/qwen-ruler

# Array: name dataset n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget
# RULER datasets: ruler_4k, ruler_8k, ruler_16k (from HuggingFace simonjegou/ruler)
#
# Methods:
#   Ours:    highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy  (algo config: best, budget: 1)
#   H2O:     highest_attn_keys_rms_nobeta_direct              (algo config: kvzip-uniform)
#   KVZip:   global_highest_attn_keys_rms_nobeta_direct       (algo config: kvzip)
configs=(
  # ---- 4K context ---- #
  # Baselines
  "ruler4k_orig           ruler_4k   -1  0 0 original,no_context 0.99 repeat summarize"
  # Ours
  "ruler4k_t0.05_ours     ruler_4k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 repeat best 1"
  "ruler4k_t0.1_ours      ruler_4k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1  repeat best 1"
  "ruler4k_t0.2_ours      ruler_4k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2  repeat best 1"
  "ruler4k_t0.5_ours      ruler_4k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.5  repeat best 1"
  # H2O
  "ruler4k_t0.05_h2o      ruler_4k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.05 repeat kvzip-uniform"
  "ruler4k_t0.1_h2o       ruler_4k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.1  repeat kvzip-uniform"
  "ruler4k_t0.2_h2o       ruler_4k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.2  repeat kvzip-uniform"
  "ruler4k_t0.5_h2o       ruler_4k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.5  repeat kvzip-uniform"
  # KVZip
  "ruler4k_t0.05_kvzip    ruler_4k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.05 repeat kvzip"
  "ruler4k_t0.1_kvzip     ruler_4k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.1  repeat kvzip"
  "ruler4k_t0.2_kvzip     ruler_4k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.2  repeat kvzip"
  "ruler4k_t0.5_kvzip     ruler_4k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.5  repeat kvzip"

  # ---- 8K context ---- #
  # Baselines
  "ruler8k_orig           ruler_8k   -1  0 0 original,no_context 0.99 repeat summarize"
  # Ours
  "ruler8k_t0.05_ours     ruler_8k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 repeat best 1"
  "ruler8k_t0.1_ours      ruler_8k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1  repeat best 1"
  "ruler8k_t0.2_ours      ruler_8k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2  repeat best 1"
  "ruler8k_t0.5_ours      ruler_8k   -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.5  repeat best 1"
  # H2O
  "ruler8k_t0.05_h2o      ruler_8k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.05 repeat kvzip-uniform"
  "ruler8k_t0.1_h2o       ruler_8k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.1  repeat kvzip-uniform"
  "ruler8k_t0.2_h2o       ruler_8k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.2  repeat kvzip-uniform"
  "ruler8k_t0.5_h2o       ruler_8k   -1  0 0 highest_attn_keys_rms_nobeta_direct 0.5  repeat kvzip-uniform"
  # KVZip
  "ruler8k_t0.05_kvzip    ruler_8k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.05 repeat kvzip"
  "ruler8k_t0.1_kvzip     ruler_8k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.1  repeat kvzip"
  "ruler8k_t0.2_kvzip     ruler_8k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.2  repeat kvzip"
  "ruler8k_t0.5_kvzip     ruler_8k   -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.5  repeat kvzip"

  # ---- 16K context ---- #
  # Baselines
  "ruler16k_orig          ruler_16k  -1  0 0 original,no_context 0.99 repeat summarize"
  # Ours
  "ruler16k_t0.05_ours    ruler_16k  -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 repeat best 1"
  "ruler16k_t0.1_ours     ruler_16k  -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1  repeat best 1"
  "ruler16k_t0.2_ours     ruler_16k  -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2  repeat best 1"
  "ruler16k_t0.5_ours     ruler_16k  -1  0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.5  repeat best 1"
  # H2O
  "ruler16k_t0.05_h2o     ruler_16k  -1  0 0 highest_attn_keys_rms_nobeta_direct 0.05 repeat kvzip-uniform"
  "ruler16k_t0.1_h2o      ruler_16k  -1  0 0 highest_attn_keys_rms_nobeta_direct 0.1  repeat kvzip-uniform"
  "ruler16k_t0.2_h2o      ruler_16k  -1  0 0 highest_attn_keys_rms_nobeta_direct 0.2  repeat kvzip-uniform"
  "ruler16k_t0.5_h2o      ruler_16k  -1  0 0 highest_attn_keys_rms_nobeta_direct 0.5  repeat kvzip-uniform"
  # KVZip
  "ruler16k_t0.05_kvzip   ruler_16k  -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.05 repeat kvzip"
  "ruler16k_t0.1_kvzip    ruler_16k  -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.1  repeat kvzip"
  "ruler16k_t0.2_kvzip    ruler_16k  -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.2  repeat kvzip"
  "ruler16k_t0.5_kvzip    ruler_16k  -1  0 0 global_highest_attn_keys_rms_nobeta_direct 0.5  repeat kvzip"
)

# Select configuration based on SLURM array task ID
config="${configs[$SLURM_ARRAY_TASK_ID]}"
read -r name dataset n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget <<< "$config"

model_flag=""
if [ -n "$MODEL" ]; then
  model_flag="--model-name $MODEL"
fi

dataset_flag=""
if [ -n "$dataset" ]; then
  dataset_flag="--dataset-name $dataset"
fi

budget_flag=""
if [ -n "$budget_path" ] && [ "$use_budget" = "1" ]; then
  budget_flag="--precomputed-budget-path $budget_path --max-ratio-per-head 0.75"
fi

log_dir_flag=""
if [ -n "$log_dir" ]; then
  log_dir_flag="--log-dir $log_dir"
fi

methods_formatted=$(echo "$methods" | tr ',' ' ')

start_time=$(date +%s)
echo "Start time: $(date -d @$start_time)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -m evaluation.run_qa_evaluation --name "$name" $dataset_flag --n-articles "$n_articles" --start-article "$start_article" --compute-stats "$compute_stats" --methods $methods_formatted --target-size "$target_size" --query-config "$query_config" --algorithm-config "$algorithm_config" $model_flag $budget_flag $log_dir_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
