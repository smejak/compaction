# Session Handoff — Marlowe per-patient SLURM run

This is a session handoff so a fresh agent can pick up the in-flight monitoring
work without re-deriving context. Companion to `contexts/06042026/PLAN.md` and
top-level `EXPERIMENT_LOG.md`; do not duplicate them — read them.

## Where we are in the 3-task workflow

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Fix SLURM script env activation | Done | Commit `a99858a` — `module load` + direct env python (`/users/jsmekal/.conda/envs/hard_drive/bin/python`), bypasses `conda activate` because the conda module shadows the env bin in PATH |
| 2 | Submit SLURM array job | Done | Job `233782` submitted from `login-02` on 2026-04-06 ~10:15 local; `mkdir -p logs && sbatch scripts/marlowe/per_patient.sh` succeeded |
| 3 | Monitor logs and fix runtime errors | **In progress — paused before any task started** | Queue state at handoff: `PD`, reason oscillating between `Priority` and `Resources`, estimated start time kept moving. `logs/` directory exists but is empty. Nothing to monitor yet. |

## Job 233782 quick facts

- **Array spec:** `--array=1,8-19%8` → 13 tasks, max 8 concurrent
- **Index → patient mapping** (`run_per_patient.py:61` derives `patient_id = f"patient_{patient_idx+1:02d}"` where `patient_idx = $SLURM_ARRAY_TASK_ID`):
  - idx `1` → `patient_02` (hung on Modal twice at 2h timeout, retrying with 20h budget)
  - idx `8..19` → `patient_09..patient_20` (never started on Modal)
- **GPU/CPU/Mem:** 1 H100, 4 CPUs/GPU, 16G/CPU
- **Wall time:** 20h per task
- **Account:** `marlowe-m000120-pm05`
- **Submit dir:** `/users/jsmekal/compaction`
- **Log path pattern:** `logs/per_patient_lh_per_patient_233782_<task>.{out,err}`
- **Already-done patients (auto-skipped):** `patient_01`, `patient_03..patient_08` — present in `long-health/` from the Modal run. `run_per_patient.py:65-67` short-circuits via `cache.pt + results.json` existence check, so any accidental array indices for these would exit in milliseconds.

## Pre-flight checks already verified — don't repeat

- `sbatch` is on PATH (`/cm/shared/apps/slurm/current/bin/sbatch`); we're on `login-02`.
- `/users/jsmekal/.conda/envs/hard_drive/bin/python` exists → Python 3.11.14.
- `/projects/m000120/jsmekal/.cache/huggingface/hub/models--Qwen--Qwen3-4B` is populated (HF cache hit, no first-run download).
- `batch` partition has live H100 nodes (mix of mixed/allocated/drained).
- `long-health/patient_{01,03..08}` already on disk from the Modal run.
- `logs/` was created by `mkdir -p logs` immediately before `sbatch` and is currently empty.

## Pickup commands for the next agent

1. **Has the job started yet?**
   ```bash
   squeue -u $USER -o '%.14i %.9P %.20j %.2t %.10M %S %R'
   sacct -j 233782 --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
   ```
2. **As soon as `logs/` has files, tail patient_02 first** (most likely to fail/hang based on Modal history):
   ```bash
   ls -la logs/
   tail -f logs/per_patient_lh_per_patient_233782_1.out logs/per_patient_lh_per_patient_233782_1.err
   ```
3. **Watch every running task:**
   ```bash
   tail -f logs/per_patient_lh_per_patient_233782_*.out
   ```
4. **Sanity check after the array finishes** (all 20 should be present):
   ```bash
   for i in $(seq -w 1 20); do
     d=long-health/patient_$i
     [ -f "$d/cache.pt" ] && [ -f "$d/results.json" ] && echo "$d OK" || echo "$d MISSING"
   done
   ```
5. **Re-submit any tasks that failed** by editing the `--array=` line in
   `scripts/marlowe/per_patient.sh` (e.g. `--array=1` to retry only patient_02)
   and resubmitting; the skip logic in `run_per_patient.py:65-67` prevents
   double-work on already-completed patients.

## Known risks to watch for

1. **patient_02 hang** — on Modal it hung twice during on-policy OMP and timed
   out at 2h. SLURM gives 20h, so even a slow run should complete or surface a
   real error this time. If it hangs the full 20h, that's a patient-specific
   debug task — not a port issue (`scripts/marlowe/PLAN.md:199-201`).
2. **First-run import errors** — the env activation chain (modules + direct env
   python) hasn't actually been exercised inside a SLURM allocation yet in this
   session. The very first thing to check in the first running task's `.err` is
   whether `import torch` and the module imports succeeded. If not, the fix is
   in `scripts/marlowe/per_patient.sh:32-36` (module loads) and line 36 (`PY=`).
3. **`mem-per-cpu=16G * cpus-per-gpu=4 = 64G`** — same as Modal's H100 setup,
   but if any patient needs more (longer context), watch for OOM in `.err` and
   bump `--mem-per-cpu` or use `--mem`.
4. **Queue contention** — at handoff time the scheduler estimate was unstable,
   bouncing between "now" and several hours out. Don't be alarmed if start is
   delayed; only intervene if jobs sit `PD` for >12h with reason `(launch failed
   requeued held)` or similar.

## What NOT to do

- Don't push to origin — the user has not asked for that.
- Don't `scancel` 233782 unless something is clearly broken; the array is
  idempotent and all-skip-on-restart, but cancelling loses queue priority.
- Don't edit `scripts/run_per_patient.py` to "fix" the patient_02 hang
  preemptively — wait for an actual error from the SLURM run before changing
  anything. The Modal hang may not reproduce on Marlowe.
- Don't commit `long-health/` — it's untracked on purpose (large binary
  `cache.pt` files, ~hundreds of MB each).

## Pointers

- **Plan this implements:** `scripts/marlowe/PLAN.md`
- **SLURM script:** `scripts/marlowe/per_patient.sh`
- **Standalone CLI it invokes:** `scripts/run_per_patient.py`
- **Modal counterpart for cross-reference:** `modal_per_patient.py`
- **Higher-level experiment context (Modal results, accuracies, formats):** `EXPERIMENT_LOG.md`
