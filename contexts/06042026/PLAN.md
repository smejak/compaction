# Plan: Port per-patient LongHealth compaction to Marlowe SLURM

## Context

The per-patient LongHealth compaction experiment (`modal_per_patient.py`) ran on
Modal H100s and completed 7 of 20 patients (01, 03–08). `patient_02` hung
twice at Modal's 2-hour timeout during on-policy OMP; `patient_09` through
`patient_20` never started because the orchestrator blocked on `patient_02`.

The 7 completed results are already downloaded to
`long-health/patient_{01,03..08}/{cache.pt,results.json}`. The goal is to run
the remaining **13 patients** (`patient_02`, `patient_09`–`patient_20`) on
Marlowe via a SLURM array job that invokes the same compaction/eval pipeline
and lands outputs in the same `long-health/` layout.

The Modal script keeps running. No change to `modal_per_patient.py`; this plan
only adds a non-Modal path alongside it.

## Approach

Two new files:

1. **`scripts/run_per_patient.py`** — single-patient Python CLI (the body of
   `modal_per_patient.py:run_patient()` stripped of Modal bits).
2. **`scripts/marlowe/per_patient.sh`** — Marlowe SLURM array job that invokes
   the CLI once per `$SLURM_ARRAY_TASK_ID`.

### 1. `scripts/run_per_patient.py` (new, ~140 lines)

A one-patient CLI that matches `modal_per_patient.py:run_patient` exactly so
results are comparable across the 7 Modal patients and the 13 Marlowe patients.

**Argparse:**
- `--patient-idx INT` (required) — 0-indexed, 0..19
- `--results-dir PATH` — default `long-health` (repo-relative)
- `--model-name STR` — default `Qwen/Qwen3-4B`
- `--ratio FLOAT` — default `0.1`

**Body — copy from `modal_per_patient.py` lines 53–187 with these edits:**

| Modal line | Replace with |
|---|---|
| `sys.path.insert(0, "/root/compaction")` / `os.chdir` (57–58) | Delete; the SLURM script `cd`s into repo root |
| `@app.function(...)` decorator (47–50) | Delete (plain Python function) |
| `out_dir = f"/results/per_patient/{patient_id}"` (112) | `out_dir = os.path.join(args.results_dir, patient_id)` |
| `vol.commit()` (184) | Delete |
| `return {...}` (187) | Keep but also print a one-line summary for log-scraping |

**Skip logic (new):** Before loading the model, check for existing outputs
and exit cleanly — matches the Modal orchestrator's skip pattern at
`modal_per_patient.py:259-268`:

```python
cache_path = os.path.join(out_dir, "cache.pt")
result_path = os.path.join(out_dir, "results.json")
if os.path.exists(cache_path) and os.path.exists(result_path):
    print(f"skip {patient_id} (already done)")
    return
```

**Config constants** — copy verbatim from `modal_per_patient.py:28-44`:
`MODEL_NAME`, `BUDGET_PATH`, `RATIO`, `AM_OMP_FAST_KWARGS` (with
`algorithm=omp`, `k_choice=4`, `nnls_interval=2`, `nnls_iters=0`,
`nnls_upper_bound=exp(7)`, `drop_key_beta_cutoff=-7`, `c2_method=lsq`,
`on_policy=True`, `precomputed_budget_path=...optimized_agnostic.json`).

**Imports** (all exist; no code changes needed in the library):
- `evaluation.utils`: `load_model_and_tokenizer`, `extract_full_kv_cache`,
  `format_context`, `format_question`, `parse_model_choice`
- `evaluation.datasets`: `load_dataset` (returns 20 LongHealth patients)
- `evaluation.configs.utils`: `load_query_config("repeat")`
- `compaction.compaction_methods.registry`: `get_compaction_method("AM-OMP-fast", ...)`
- `models.generate`: `generate_with_compacted_cache_batch`

### 2. `scripts/marlowe/per_patient.sh` (new, ~55 lines)

SLURM array job using the user-provided Marlowe header, with `--array` added
to run only the 13 remaining patients.

**Patient index mapping**: `patient_XX` uses 0-based index `XX-1`, so:
- `patient_02` → idx `1`
- `patient_09`–`patient_20` → idx `8`–`19`
- Array spec: `--array=1,8-19` (13 tasks)
- Throttle to 8 concurrent to match Modal's `MAX_PARALLEL=8`: `--array=1,8-19%8`

**Script body:**

```bash
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

set -euo pipefail

# -------- Environment ------------------------------------------------ #
# Edit VENV_PATH if your venv lives elsewhere on Marlowe.
VENV_PATH="${VENV_PATH:-$HOME/compaction/.venv}"
source "$VENV_PATH/bin/activate"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs long-health

# -------- Job info --------------------------------------------------- #
start_time=$(date +%s)
echo "Start: $(date -d @$start_time)"
echo "Node:  $(hostname)"
echo "Task:  patient_idx=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -------- Run single patient ---------------------------------------- #
python -u scripts/run_per_patient.py \
    --patient-idx "$SLURM_ARRAY_TASK_ID" \
    --results-dir long-health

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Elapsed: %dh %dm %ds\n' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
```

**Header notes:**
- Matches the user-supplied Marlowe reference header line-for-line except
  `--job-name` is set to `lh_per_patient` and `--array` / log paths are added.
- `--time=20:00:00` is 10× the Modal timeout — generous enough that
  `patient_02` should finish even if its on-policy OMP was near the edge.
- Log path prefix `per_patient_` distinguishes this run from other SLURM jobs
  using the shared `logs/` directory.
- `set -euo pipefail` so a failing Python run surfaces as a SLURM task failure.

### Critical files

| Action | File | Notes |
|---|---|---|
| Read (source) | `modal_per_patient.py` (51–187) | Copy per-patient body |
| Read (source) | `modal_per_patient.py` (28–44) | Copy config constants |
| Create | `scripts/run_per_patient.py` | New CLI wrapper |
| Create | `scripts/marlowe/per_patient.sh` | New SLURM script |
| No change | `evaluation/utils.py`, `evaluation/datasets.py`, `models/generate.py`, `compaction/**` | Library code reused as-is |
| No change | `modal_per_patient.py` | Kept for Modal use |
| No change | `head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json` | Already present, loaded by path |

## Verification

**1. Local smoke test** (optional, on any CUDA machine before submitting):
```bash
# Run an already-done patient and confirm the skip path triggers:
python -u scripts/run_per_patient.py --patient-idx 0 --results-dir long-health
# Expect: "skip patient_01 (already done)" and exit 0
```

**2. Submit to Marlowe:**
```bash
cd ~/compaction
sbatch scripts/marlowe/per_patient.sh
```

**3. Monitor:**
```bash
squeue -u $USER
tail -f logs/per_patient_lh_per_patient_*_1.out   # patient_02
```

**4. Completion check** — after the array finishes, verify all 20 patients
have both files:
```bash
for i in $(seq -w 1 20); do
  d=long-health/patient_$i
  [ -f "$d/cache.pt" ] && [ -f "$d/results.json" ] && echo "$d OK" || echo "$d MISSING"
done
```

**5. Quick accuracy sanity check:**
```bash
python -c "
import json, glob
for p in sorted(glob.glob('long-health/patient_*/results.json')):
    d = json.load(open(p))
    print(f\"{d['patient_id']}: {d['accuracy']:.0%} ({d['correct']}/{d['total']})\")
"
```

## Open items / assumptions

- **VENV_PATH**: defaults to `$HOME/compaction/.venv`. If the venv lives
  somewhere else on Marlowe, edit the one line at the top of
  `scripts/marlowe/per_patient.sh` (or export `VENV_PATH` before `sbatch`).
- **`patient_02` still hangs**: the 20h SLURM budget is ~10× what Modal
  allowed, but this isn't a logic fix. If it hangs again, it's a
  patient-specific debug task (not in scope for this port).
- **Modal cache format compatibility**: `cache.pt` saved by the Marlowe CLI is
  byte-compatible with the 7 files already in `long-health/` — same keys,
  same dtypes, same CPU placement — so downstream consumers won't need
  changes.
