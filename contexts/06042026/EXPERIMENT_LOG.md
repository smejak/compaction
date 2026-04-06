# Per-Patient KV Cache Compaction — Experiment Log

## What was run

Individual KV cache compaction on each patient in the LongHealth dataset using Qwen3-4B on Modal H100 GPUs.

### Setup

- **Model**: `Qwen/Qwen3-4B` (bfloat16, SDPA attention)
- **Dataset**: LongHealth — 20 patients, each with 2-13 clinical documents (~8k-15k tokens per patient) and 20 multiple-choice questions (5 options, A-E)
- **Compaction ratio**: 0.1 (retain 10% of article tokens)
- **Algorithm**: `AM-OMP-fast`
  - `algorithm=omp`, `k_choice=4`, `nnls_interval=2`
  - `nnls_iters=0`, `nnls_upper_bound=exp(7)`, `drop_key_beta_cutoff=-7`
  - `c2_method=lsq`, `on_policy=True`
  - `precomputed_budget_path=head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json`
- **Query generation**: `repeat` config (self-study, repeats context prefill once to extract query vectors)
- **Infrastructure**: Modal H100 GPUs, max 8 parallel

### What each job does

1. Load Qwen3-4B model
2. Load single-patient article (one patient's clinical documents)
3. Format with chat template and extract full KV cache via `extract_full_kv_cache`
4. Run AM-OMP-fast compaction at 10% ratio with nonuniform per-head budgets
5. Save compacted KV cache as `cache.pt` (torch checkpoint)
6. Evaluate all 20 questions using `generate_with_compacted_cache_batch`
7. Save per-question results as `results.json`

### Script

```bash
modal run modal_per_patient.py
```

The orchestrator (`orchestrate` function) runs on Modal, spawns GPU jobs in batches of 8, and skips already-completed patients. The script is idempotent — re-running picks up where it left off.

## Current status

**7 of 20 patients completed** (from batch 1). Patient_02 hung during on_policy OMP compaction (timed out at 2 hours both attempts). Patients 09-20 were never started because the orchestrator blocked on patient_02.

| Patient | Name | Accuracy | Status |
|---------|------|----------|--------|
| patient_01 | Anna Sample | 80% (16/20) | Done |
| patient_02 | Jane Done | — | Hung (on_policy OMP timeout) |
| patient_03 | Mr. John Williams | 90% (18/20) | Done |
| patient_04 | Jill Anderson | 90% (18/20) | Done |
| patient_05 | John Miller | 85% (17/20) | Done |
| patient_06 | Paul Doe | 80% (16/20) | Done |
| patient_07 | Linda Mayer | 90% (18/20) | Done |
| patient_08 | Laura Miller | 75% (15/20) | Done |
| patient_09–20 | — | — | Not started |

## Where results are stored

### Modal volume

Volume name: `am-experiment-results`

```
/per_patient/
  patient_01/
    cache.pt          # compacted KV cache (torch checkpoint)
    results.json      # evaluation results
  patient_03/
    cache.pt
    results.json
  ... (patient_04 through patient_08)
```

### cache.pt format

```python
import torch
data = torch.load("cache.pt", weights_only=False)

data["cache"]            # list of (C1, beta, C2) tuples, one per layer
                         # C1: (1, num_kv_heads, t, head_dim) — compacted keys
                         # beta: (1, num_kv_heads, t) — attention biases
                         # C2: (1, num_kv_heads, t, head_dim) — compacted values
                         # t varies per layer (nonuniform head budgets)
data["original_seq_len"] # int — original sequence length before compaction
data["patient_id"]       # str — e.g. "patient_01"
data["patient_name"]     # str — e.g. "Anna Sample"
data["model"]            # str — "Qwen/Qwen3-4B"
data["ratio"]            # float — 0.1
data["article_len"]      # int — article token count
data["compaction_time"]  # float — seconds
```

### results.json format

```json
{
  "patient_id": "patient_01",
  "patient_name": "Anna Sample",
  "model": "Qwen/Qwen3-4B",
  "ratio": 0.1,
  "accuracy": 0.8,
  "correct": 16,
  "total": 20,
  "compaction_time": 123.4,
  "max_layer_len": 12345,
  "per_question": [
    {"qid": "patient_01_q0", "correct": true, "pred": 4, "gold": 4, "answer": "..."}
  ]
}
```

## How to download results

```bash
bash scripts/download_per_patient.sh              # downloads to results/per_patient/
bash scripts/download_per_patient.sh /my/path      # custom output directory
```

## How to use a saved compacted cache

```python
import torch
from models.generate import generate_with_compacted_cache_batch
from evaluation.utils import load_model_and_tokenizer, format_question

# Load model
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-4B", "cuda")

# Load saved cache
data = torch.load("results/per_patient/patient_01/cache.pt", weights_only=False)
cache = data["cache"]  # list of (C1, beta, C2) per layer
original_seq_len = data["original_seq_len"]

# Move to GPU
device = next(model.parameters()).device
cache_gpu = tuple((c1.to(device), b.to(device), c2.to(device)) for c1, b, c2 in cache)

# Generate answer
prompt = format_question(tokenizer, "What is the patient's diagnosis?",
                         ["DLBCL", "AML", "CLL", "ALL", "MDS"], "Qwen/Qwen3-4B")
answers = generate_with_compacted_cache_batch(
    model, tokenizer, [prompt], cache_gpu,
    max_new_tokens=2048, original_seq_len=original_seq_len)
print(answers[0])
```

## How to re-run remaining patients

```bash
# Just re-run — skip logic handles completed patients automatically
modal run modal_per_patient.py
```

The orchestrator checks for existing `cache.pt` + `results.json` on the volume and skips patients that are already done. Patient_02 may need investigation (on_policy OMP hangs on its 5-document, 12.5k-token context).

## Previous experiment: Joint vs Independent (2-patient pilot)

An earlier experiment compared joint vs independent compaction on 2 patients using Qwen3-8B at ratio=0.1 (script: `modal_stacked_experiment.py`). Results:

- **Joint**: 55% overall (P1: 55%, P2: 55%)
- **Independent**: 35% overall (P1: 20%, P2: 50%)

Joint compaction outperformed independent by 20pp overall, with P1 suffering most under independent compaction. These results used `AM-HighestAttnKeys` (not OMP) with `on_policy=False` due to OOM issues with Qwen3-8B.

Results stored on volume at `/pilot/J/` and `/pilot/I/`.
