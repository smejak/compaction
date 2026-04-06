# Per-Patient SLURM Run Summary — 2026-04-06

Post-mortem of the Marlowe SLURM port of the per-patient AM-OMP-fast LongHealth
experiment. Companion to `PLAN.md` (the design) and `HANDOFF.md` (the
workflow checklist). This file is the narrative of what actually happened on
2026-04-06, including the two env-related failures and how they were
diagnosed and fixed.

## TL;DR

- **Outcome:** All 13 missing patients (`patient_02`, `patient_09..patient_20`)
  ran end-to-end on Marlowe, in three submission attempts. The 7
  Modal-baseline patients (01, 03..08) were correctly skipped via
  `run_per_patient.py:65-67`'s cache existence check. **All 20 patients now
  have `cache.pt + results.json` on disk.**
- **Mean accuracy across the 13 SLURM patients: 0.850**.
- **patient_02**, which hung twice on Modal at the 2h timeout, finished
  cleanly on Marlowe in **1h 12m at 90% accuracy** — the Modal hang did not
  reproduce.
- Two env-related fixes were required between the failing first/second
  attempts and the successful third attempt:
  1. Pin `torchvision==0.21.0+cu124` to match `torch==2.6.0+cu124` in the
     `hard_drive` conda env.
  2. Strip all `module load` calls from `scripts/marlowe/per_patient.sh` —
     `cudatoolkit/12.5` was poisoning `LD_LIBRARY_PATH`.

## Starting state (handoff context)

The prior session left job **233782** submitted but pending in the queue,
oscillating between `(Priority)` and `(Resources)`, with `logs/` empty. The
handoff explicitly noted that the env activation chain (modules + direct env
python) had **not yet been exercised inside a SLURM allocation** in any
session, so the first run was effectively a smoke test of the environment.

State at the start of this session (2026-04-06 ~10:30):
- Job 233782 had actually run between handoff and pickup.
- All 13 array tasks (1, 8..19) had **FAILED** within ~5-15 seconds each.
- Logs existed for tasks 8..19 but **not** for task 1, despite sacct showing
  it ran on n31 from 10:18:19 → 10:18:30. The log file disappearance was a
  one-time NFS/state oddity that did not recur on the third submission.

## Failure #1 — `RuntimeError: operator torchvision::nms does not exist`

### Symptom

Every task crashed within ~10s, traceback ending in:
```
File ".../torchvision/_meta_registrations.py", line 163, in <module>
    @torch.library.register_fake("torchvision::nms")
  ...
RuntimeError: operator torchvision::nms does not exist
```

The crash happened during `from models.qwen3 import Qwen3ForCausalLM` →
`from transformers.modeling_layers import (...)` →
`transformers/processing_utils.py` → `transformers/image_utils.py:55` →
`from torchvision.transforms import InterpolationMode`. The unconditional
torchvision import inside `image_utils.py` is gated by
`is_torchvision_available()` (`image_utils.py:54`), but
`is_torchvision_available()` only checks `find_spec("torchvision")`, which
returns True even when the module fails to import.

### Diagnosis

```bash
$ /users/jsmekal/.conda/envs/hard_drive/bin/pip show torch
Version: 2.6.0
$ /users/jsmekal/.conda/envs/hard_drive/bin/pip show torchvision
Version: 0.25.0
Required-by: vllm
```

torch 2.6.0 ships with the C++ op `torchvision::nms` registered against
torchvision 0.21.x's ABI. torchvision 0.25.0 expects a much newer torch
(roughly 2.10.x) and tries to register a fake-impl for an op that doesn't
exist in torch 2.6's C registry. The 0.25.0 wheel had been pulled in by
some prior `pip install` (likely a vllm upgrade), creating the version skew.

### Fix

Pin torchvision back to the version matching torch 2.6.0+cu124, taking
care to avoid letting pip touch torch itself:

```bash
/users/jsmekal/.conda/envs/hard_drive/bin/pip install --no-deps \
  "torchvision==0.21.0+cu124" \
  --index-url https://download.pytorch.org/whl/cu124
```

Verification:
```python
import torch                                          # 2.6.0+cu124
import torchvision                                    # 0.21.0+cu124
from torchvision.transforms import InterpolationMode  # OK
from transformers.modeling_layers import GradientCheckpointingLayer  # OK
from models.qwen3 import Qwen3ForCausalLM             # OK
```

vllm pins `torchvision==0.25.0` in this env's metadata but is **not** imported
on the per-patient compaction codepath, so the downgrade is safe for this
experiment. If vllm-using code paths break later, the fix is to upgrade torch
to a 2.10.x release (which the rest of the env may not be ready for) or to
maintain a separate env.

## Failure #2 — `cudaErrorSystemDriverMismatch` (CUDA error 803)

### Symptom

After fix #1, **job 233828** was submitted. All tasks again crashed within
~15s, this time with the model loading further along (data loaded, "Loading
model: Qwen/Qwen3-4B" printed) before:
```
File ".../torch/cuda/__init__.py", line 319, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Unexpected error from cudaGetDeviceCount(). ... Error 803:
system has unsupported display driver / cuda driver combination
```

The crash was reached at `transformers/modeling_utils.py:5432` in
`caching_allocator_warmup`, the very first place that touches CUDA. Notably:
- `nvidia-smi` printed normally from the SLURM script (it talks to the kernel
  driver directly, not to the CUDA runtime).
- Data loading worked.
- Only `torch._C._cuda_init()` failed.

### Diagnosis

I grabbed an interactive allocation on a compute node and bisected the
environment:

```bash
srun -N 1 -G 1 -A marlowe-m000120-pm05 -p batch -t 00:10:00 bash -c '
  /users/jsmekal/.conda/envs/hard_drive/bin/python -c \
    "import torch; print(torch.cuda.is_available())"
'
# → True
```

CUDA worked **without** modules. Then loading the same modules used by the
SLURM script:

```bash
module load cudatoolkit/12.5
module load cudnn/cuda12/9.3.0.75
module load conda/24.3.0-0
/users/jsmekal/.conda/envs/hard_drive/bin/python -c \
  "import torch; print(torch.cuda.is_available())"
# → False, with error 803
```

`echo $LD_LIBRARY_PATH` after the module loads showed:
```
/cm/shared/apps/cudnn/12-9.3.0.75/lib:
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/cuda/12.5/compat:    ← culprit
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/comm_libs/nvshmem/lib:
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/comm_libs/nccl/lib:
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/math_libs/lib64:
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/compilers/lib:
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/compilers/extras/qd/lib:
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/cuda/extras/CUPTI/lib64:
/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/cuda/lib64:
...
```

`nvidia-smi` on the compute node reported:
```
Driver Version: 565.57.01      CUDA Version: 12.7
```

So the host driver (565.57.01) supports CUDA 12.7. Torch is built against
cu124 and ships its own CUDA runtime via the `nvidia-cuda-runtime-cu12` pip
wheel. **The CUDA `compat` directory inside the cudatoolkit/12.5 module is
meant for older drivers** — it provides a forward-compat `libcuda.so` that
allows running CUDA 12.5 runtimes on driver versions older than 555. On
Marlowe's H100 nodes (driver 565+) this compat lib is *older* than the
system driver and overrides it via `LD_LIBRARY_PATH`. Torch's cu124 runtime
then sees a driver/runtime version mismatch and fails initialization.

### Fix

Strip all three `module load` calls from `scripts/marlowe/per_patient.sh`.
Torch's pip-installed `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`,
`nvidia-cublas-cu12`, etc. provide everything torch needs. The conda module
was only there to enable `conda activate`, which the script already bypasses
by calling `/users/jsmekal/.conda/envs/hard_drive/bin/python` directly.

The replaced section in `scripts/marlowe/per_patient.sh:27-34` now reads:

```bash
# -------- Environment ------------------------------------------------ #
# Marlowe setup: call the env's python directly. Do NOT load the cudatoolkit
# or cudnn modules — they prepend nvhpc's CUDA 12.5 compat libs to
# LD_LIBRARY_PATH, which shadow the H100 driver (565.57.01 / CUDA 12.7) and
# break torch's bundled cu124 runtime with `cudaErrorSystemDriverMismatch`
# (error 803). Torch's pip-installed nvidia-cuda-*-cu12 packages already
# ship the right runtime, so the system modules are unnecessary.
PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"
```

## The successful run — job 233882

Submitted 2026-04-06 ~11:27 PDT. Same `--array=1,8-19%8`, same resource
spec (1 H100, 4 CPUs/GPU, 16G/CPU, 20h walltime). The run was monitored by
a small background bash poller (`/tmp/marlowe_monitor.sh`) that scanned
`squeue`, log tails, and `.err` files every 60s.

### Timeline

| time (PDT) | event |
|---|---|
| 11:27:23 | Tasks 1, 8 start on n24 |
| 11:28:55 | Task 9 starts on n25 |
| ~11:34 | Tasks 10, 11 start |
| ~11:35 | Task 12 starts |
| ~11:36 | Tasks 13, 14 start (8 concurrent — `%8` cap reached) |
| 12:11:50 | First completion: patient_09 (task 8) at 95%, 44m 29s |
| ~12:35 | Tasks 15, 16 start as earlier ones drain |
| ~12:48 | Task 17 starts |
| 13:00 | 8 patients complete (02, 09–15) |
| ~13:13 | Tasks 18, 19 start |
| 13:16 | 9 done (patient_16 finishes) |
| 13:27 | 10 done (patient_17) |
| 13:59 | 12 done (patients 18, 20) |
| 14:06 | **All 13 done** (patient_19 finishes after 1h 23m) |

Total wallclock from first task start to last task completion:
**11:27:23 → 14:06:14 ≈ 2h 39m**.

### Per-patient elapsed times

Wallclock from `Start:` line to `Elapsed:` line in each `.out`:

| array idx | patient | elapsed | accuracy | notes |
|---|---|---|---|---|
| 1 | patient_02 | 1h 12m 40s | 90% | previously hung on Modal |
| 8 | patient_09 | 0h 44m 29s | 95% | fastest — uncontended early |
| 9 | patient_10 | 0h 53m 21s | 95% | |
| 10 | patient_11 | 1h 20m 49s | 85% | |
| 11 | patient_12 | 1h 22m 25s | 95% | |
| 12 | patient_13 | 1h 15m 49s | 75% | |
| 13 | patient_14 | 1h 01m 39s | 70% | |
| 14 | patient_15 | 0h 57m 34s | 95% | |
| 15 | patient_16 | 0h 53m 05s | 85% | |
| 16 | patient_17 | 1h 01m 00s | 90% | |
| 17 | patient_18 | 1h 15m 16s | 80% | |
| 18 | patient_19 | 1h 23m 05s | 75% | |
| 19 | patient_20 | 1h 09m 00s | 75% | |
| **mean** |  | **~1h 05m** | **0.850** |  |

The compaction loop ran at ~1.2-1.7 min/layer × 36 layers under 8-way
concurrency on shared H100 nodes (multiple tasks packed onto the same node,
each with its own GPU but shared CPU/memory bandwidth). The on-policy query
extraction at layer 2 (`nnls_interval=2` triggers extraction once after the
warmup layers) is the slow step that accounts for the early-layer
appearance of synchronized progress — the script prints "Compacting layer N"
fast, then sits silent during the long compute step before printing the
next batch of layer numbers.

### Final accuracy across all 20 patients

```
patient_01: 80%   (Modal baseline)
patient_02: 90%   (SLURM, 18/20)
patient_03: 90%   (Modal baseline)
patient_04: 90%   (Modal baseline)
patient_05: 85%   (Modal baseline)
patient_06: 80%   (Modal baseline)
patient_07: 90%   (Modal baseline)
patient_08: 75%   (Modal baseline)
patient_09: 95%   (SLURM, 19/20)
patient_10: 95%   (SLURM, 19/20)
patient_11: 85%   (SLURM, 17/20)
patient_12: 95%   (SLURM, 19/20)
patient_13: 75%   (SLURM, 15/20)
patient_14: 70%   (SLURM, 14/20)
patient_15: 95%   (SLURM, 19/20)
patient_16: 85%   (SLURM, 17/20)
patient_17: 90%   (SLURM, 18/20)
patient_18: 80%   (SLURM, 16/20)
patient_19: 75%   (SLURM, 15/20)
patient_20: 75%   (SLURM, 15/20)
```

Mean across all 20: **(0.80 + 0.90 + 0.90 + 0.90 + 0.85 + 0.80 + 0.90 + 0.75
+ 0.95 + 0.95 + 0.85 + 0.95 + 0.75 + 0.70 + 0.95 + 0.85 + 0.90 + 0.80 + 0.75
+ 0.75) / 20 = 0.8475**.

Mean across the 13 SLURM patients: **0.850**. The SLURM batch is slightly
above the Modal baseline mean (0.85 vs ~0.84 for Modal patients
01, 03..08), well within run-to-run noise.

## Files changed during this session

| file | change |
|---|---|
| `scripts/marlowe/per_patient.sh:27-34` | Removed `module load cudatoolkit/12.5`, `cudnn/cuda12/9.3.0.75`, `conda/24.3.0-0`. Replaced with comment explaining why. |
| `~/.conda/envs/hard_drive/lib/python3.11/site-packages/torchvision/...` | Replaced 0.25.0 with 0.21.0+cu124 via `pip install --no-deps`. |
| `contexts/06042026/HANDOFF.md` | Workflow table extended through final completion; added accuracy table; marked workflow Done. |
| `~/.claude/projects/-users-jsmekal-compaction/memory/feedback_marlowe_env.md` | New memory documenting both env traps. |
| `~/.claude/projects/-users-jsmekal-compaction/memory/MEMORY.md` | Index entry for the new memory. |
| `contexts/06042026/PER_PATIENT_RUN_SUMMARY.md` | This file. |

No changes to `scripts/run_per_patient.py`, `models/`, `compaction/`, or any
other code path. The fixes were purely environmental — the AM-OMP-fast
implementation itself ran cleanly on Marlowe with no patches.

## Things to remember for future Marlowe runs

These are now also captured in
`~/.claude/projects/-users-jsmekal-compaction/memory/feedback_marlowe_env.md`
so they survive into future sessions:

1. **Do not load `cudatoolkit/12.5` for torch + cu124 jobs.** The `compat`
   directory inside it shadows the system H100 driver and breaks torch CUDA
   init. Same goes for the other nvhpc-derived cudatoolkit modules; if you
   ever need them for non-torch code, set them up in a separate shell.
2. **Watch for `torchvision==0.25.0` drift in the `hard_drive` env.** vllm
   pulls it in but it's incompatible with torch 2.6. If a future
   `pip install` reintroduces it and breaks transformers imports, the fix is
   the `--no-deps torchvision==0.21.0+cu124` reinstall.
3. **`is_torchvision_available()` is not a real availability check** — it
   only checks `find_spec`, so a broken torchvision still reports True. If
   you ever need to defensively-skip torchvision, the only reliable signal
   is wrapping the actual import in a try/except.
4. **`nvidia-smi` working ≠ CUDA runtime working.** nvidia-smi talks to the
   kernel driver directly. The first real CUDA-runtime check is
   `torch.cuda.is_available()` or `torch.cuda.init()`.
5. **Job array task logs may be missing on first run for unknown reasons.**
   In the very first failed submission (233782), task index 1's `.out`/`.err`
   files were never created on disk despite sacct showing the task ran on
   n31. This did not reproduce in 233828 or 233882 — likely an NFS metadata
   race tied to the first allocation. If it happens again, sacct +
   `scontrol show job` is the fallback for getting state.

## Pointers

- **Plan this implements:** `contexts/06042026/PLAN.md`
- **Workflow checklist:** `contexts/06042026/HANDOFF.md`
- **SLURM script:** `scripts/marlowe/per_patient.sh`
- **Standalone CLI:** `scripts/run_per_patient.py`
- **Modal counterpart:** `modal_per_patient.py`
- **Higher-level experiment context:** `contexts/06042026/EXPERIMENT_LOG.md`
- **Marlowe usage cheat sheet:** `contexts/MARLOWE.md`
