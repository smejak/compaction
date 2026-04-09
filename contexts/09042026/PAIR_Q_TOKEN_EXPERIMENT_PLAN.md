# Per-Q-token attention experiment for patient_03 / patient_04

## Context

Drill-down 6 in `contexts/09042026/PAIR_EXPERIMENT_RESULTS_REPORT.md` showed
that the per-region per-layer attention mass on the **last question token**
does not distinguish correct from incorrect answers in either variant —
selectivity ≈ 0 under naive concat, and strong but correctness-independent
under rope_shift. The natural follow-up: maybe earlier tokens in the
question (not just the last one) show different dynamics. The model could
be **accumulating information** across the question — early Q tokens
attending to one patient, later Q tokens shifting attention as more context
gets pulled into the residual stream — and the last-token snapshot would
miss that.

To test this, we re-evaluate **one pair** with full per-Q-token attention
recording (instead of last-token only) and look at the per-token
trajectories. We pick the pair that scores high on both axes the user
cares about: large swap asymmetry AND large per-patient degradation vs the
single-patient baseline.

## Pair selection — patient_03 / patient_04 (confirmed)

From joining `summary_extended.json` (asymmetry top-10) with
`individual_vs_paired_summary.json` (per-patient deltas):

| pair | swap asymmetry | gap from individual | min Δ from indiv | combined |
|---|---|---|---|---|
| **patient_03 / patient_04** | 0.275 (rank #2) | **0.325** (tied #1) | −0.161 (P04, top-3 loser) | **0.761** |
| patient_06 / patient_12 | 0.300 (rank #1) | 0.325 (tied #1) | −0.120 (mid-tier) | 0.745 |
| patient_05 / patient_11 | 0.225 | 0.300 | −0.207 (worst) | 0.732 |

patient_03 / patient_04 wins on the combined criterion (chosen by user):
- Both individuals start at 0.90 (high baseline) — clean apples-to-apples
- rope_shift 03→04 = 0.850, **rope_shift 04→03 = 0.575** — 27.5 pp swap gap
- Naive 03→04 = 0.875, naive 04→03 ≈ 0.85 (much smaller swap gap)
- patient_04 is the #3 worst single-patient loser under rope_shift (Δ = −16.1 pp)
- patient_03 also notable (Δ = −11.4 pp under rope_shift)

We re-run **both orderings × both variants = 4 evals**:

| pair-idx | A | B | variant | expected from existing data |
|---|---|---|---|---|
| 40 | patient_03 | patient_04 | naive       | overall ≈ 0.875 |
| 40 | patient_03 | patient_04 | rope_shift  | overall ≈ 0.850 |
| 59 | patient_04 | patient_03 | naive       | overall ≈ 0.85  |
| 59 | patient_04 | patient_03 | rope_shift  | overall ≈ 0.575 (the failure mode) |

## Decisions (from user)

1. **Pair**: patient_03 / patient_04, **both naive AND rope_shift** for both orderings.
2. **Recording mode**: full per-Q-token series (n_real_Q_tokens × 36 layers × 3 regions per question), not just mean/std.

## Scope and out-of-scope

**In scope**:
- Modify `scripts/run_pair_experiment.py` to optionally record per-Q-token attention behind a flag.
- Re-run 4 evals (one pair × 2 orderings × 2 variants) with the flag set.
- Write a focused analysis script that loads the 4 outputs and produces the per-Q-token dynamics figures.

**Out of scope**:
- Re-running any other pairs (only patient_03 / patient_04).
- Modifying the canonical aggregator or existing analysis pipeline. The new outputs land at a separate path so canonical results are untouched.
- Pushing to git (user does it).

## Code change to `scripts/run_pair_experiment.py`

The core modification is in `_run_instrumented_forward_single` (currently
script lines 252–356). It already has the full per-layer attention tensor
in scope; the only change is to extract attention rows for **all real Q
tokens**, not just the last one.

### New flag

Add to `main()`:
```python
parser.add_argument("--record-q-token-trace", action="store_true",
                    help="Also record per-Q-token attention into "
                         "<out_dir>/q_token_trace.json. Off by default.")
parser.add_argument("--pair-a", type=str, default=None,
                    help="Patient A id (alternative to --pair-idx). "
                         "Both --pair-a and --pair-b must be set together.")
parser.add_argument("--pair-b", type=str, default=None,
                    help="Patient B id (alternative to --pair-idx).")
```

If `--pair-a` / `--pair-b` are set, compute `pair_idx = PAIRS.index((pair_a, pair_b))`
and ignore `--pair-idx`. This makes one-off runs ergonomic.

### Modify `_run_instrumented_forward_single`

```python
def _run_instrumented_forward_single(model, tokenizer, prompt, cache_cpu,
                                     stacked_seq_len, t_A_per_layer,
                                     t_B_per_layer, device, dtype,
                                     record_q_token_trace=False):
    # ... existing tokenization, cache build, forward pass ...

    # Existing: last query token attention (always recorded)
    q_last = input_len - 1
    layers_abq = []
    layers_abq_per_q_token = [] if record_q_token_trace else None

    pad_count = int(pad_counts[0])  # left-padded → real Q tokens at [pad_count, input_len)

    for layer_idx, attn in enumerate(attentions):
        t_a = t_A_per_layer[layer_idx]
        t_b = t_B_per_layer[layer_idx]
        # Last-token (existing — unchanged)
        row = attn[0, :, q_last, :].float().mean(dim=0)
        layers_abq.append({
            "layer": layer_idx,
            "A": float(row[:t_a].sum()),
            "B": float(row[t_a:t_a + t_b].sum()),
            "Q": float(row[t_a + t_b:].sum()),
        })

        # NEW: per-Q-token (only when flag is set)
        if record_q_token_trace:
            # All real (non-pad) Q rows. Average over heads only.
            rows_real = attn[0, :, pad_count:input_len, :].float().mean(dim=0)
            # rows_real shape: (n_real_Q, k_len)
            mass_a = rows_real[:, :t_a].sum(dim=-1)            # (n_real_Q,)
            mass_b = rows_real[:, t_a:t_a + t_b].sum(dim=-1)
            mass_q = rows_real[:, t_a + t_b:].sum(dim=-1)
            layers_abq_per_q_token.append({
                "layer": layer_idx,
                "A": mass_a.tolist(),
                "B": mass_b.tolist(),
                "Q": mass_q.tolist(),
            })

    # ... existing cleanup ...

    if record_q_token_trace:
        return layers_abq, layers_abq_per_q_token, int(input_len - pad_count)
    return layers_abq
```

### Modify `run_pair`

- Take a `record_q_token_trace: bool` argument
- Pass it through to the inner forward call
- When set: collect per-question per-Q-token data and write
  `<out_dir>/q_token_trace.json` alongside `results.json`
- The existing `results.json` is **unchanged in schema** — last-token
  attention still goes there. The per-Q-token data lives in a sibling file
  so canonical analysis tools work unchanged.
- The idempotency check (line 389) should now check `q_token_trace.json`
  too if the flag is set, so re-running with the flag recovers from a partial run.

### Output schema for `q_token_trace.json`

```json
{
  "variant": "rope_shift",
  "pair": ["patient_04", "patient_03"],
  "stacked_seq_len": ...,
  "t_A_per_layer": [..36 ints..],
  "t_B_per_layer": [..36 ints..],
  "n_layers": 36,
  "model": "Qwen/Qwen3-4B",
  "per_question": [
    {
      "qid": "patient_04_q0",
      "patient": "patient_04",
      "position": 1,
      "correct": true,
      "n_q_tokens": 487,
      "attn_per_layer_per_q_token": {
        "A": [[..487 floats..], ..36 layer entries..],
        "B": [[..487 floats..], ..],
        "Q": [[..487 floats..], ..]
      }
    },
    ...
  ]
}
```

**Storage estimate**: ≈ 500 Q tokens × 36 layers × 3 regions × 8 bytes × 40
questions ≈ 17 MB per eval, ~70 MB across 4 evals. JSON-encoded with
6-digit floats: maybe 3-5x larger → ~250 MB total. Acceptable but at the
edge — if it gets uncomfortable I'll switch to `.npz` (numpy binary).
**I'll use JSON for inspectability and switch to `.npz` only if a
storage check after the first eval comes back > 100 MB.**

### Output path

Use a separate `--results-dir long-health/pair_experiment_q_token` (CLI flag
already exists). Both `results.json` and `q_token_trace.json` land at
`long-health/pair_experiment_q_token/{naive,rope_shift}/pair_<A>_<B>/`. The
canonical `long-health/pair_experiment/...` is untouched.

## New file: `scripts/marlowe/pair_q_token.sh`

A small SLURM wrapper that submits **one** eval (no array). 4 invocations
total — one per (variant, ordering). Each is an independent sbatch.

Modeled on `scripts/marlowe/pair_experiment.sh` but stripped to a single
job:

```bash
#!/bin/bash
#SBATCH --job-name=lh_qtok
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --error=logs/qtok_%x_%j.err
#SBATCH --output=logs/qtok_%x_%j.out

# Re-run a single pair with per-Q-token attention recording.
#
# Required env vars:
#   PAIR_A     — e.g. patient_03
#   PAIR_B     — e.g. patient_04
#   VARIANT    — naive | rope_shift
#
# Submit:
#   for V in naive rope_shift; do
#     for AB in "patient_03 patient_04" "patient_04 patient_03"; do
#       PA=$(echo $AB | awk '{print $1}'); PB=$(echo $AB | awk '{print $2}')
#       PAIR_A=$PA PAIR_B=$PB VARIANT=$V \
#         sbatch --export=ALL,PAIR_A,PAIR_B,VARIANT scripts/marlowe/pair_q_token.sh
#     done
#   done

set -uo pipefail

PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"
cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

export HF_HOME="${HF_HOME:-/projects/m000120/jsmekal/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs long-health/pair_experiment_q_token

echo "Pair: $PAIR_A -> $PAIR_B  variant=$VARIANT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -u scripts/run_pair_experiment.py \
    --pair-a "$PAIR_A" \
    --pair-b "$PAIR_B" \
    --variant "$VARIANT" \
    --record-q-token-trace \
    --results-dir long-health/pair_experiment_q_token \
    --caches-dir long-health
```

Wall-time: 2 hours per single eval is generous (canonical runs took 20-40
min per pair under the array, and this is the same eval shape just with a
slightly larger output buffer).

## New file: `scripts/analyze_q_token_attention.py`

Standalone CPU-only analysis script that loads the 4 q_token_trace.json
files and produces dynamics figures.

### Top-level shape

```python
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS = [
    ("naive",      "patient_03", "patient_04"),
    ("naive",      "patient_04", "patient_03"),
    ("rope_shift", "patient_03", "patient_04"),
    ("rope_shift", "patient_04", "patient_03"),
]

def load_trace(results_dir, variant, a, b):
    """Load q_token_trace.json for one (variant, ordering)."""
    path = Path(results_dir) / variant / f"pair_{a}_{b}" / "q_token_trace.json"
    with open(path) as f:
        return json.load(f)

def per_q_token_means(trace, position, correct):
    """Within (position, correct) cell, return mean (across questions)
    of attn_right / attn_wrong / attn_q as (n_layers, n_q_tokens) arrays.

    Different questions have different n_q_tokens — pad to max with NaN
    and use nanmean."""
    questions = [q for q in trace["per_question"]
                 if q["position"] == position and q["correct"] == correct]
    if not questions:
        return None
    n_layers = trace["n_layers"]
    max_q = max(q["n_q_tokens"] for q in questions)
    # Stack into (n_questions, n_layers, n_q_tokens) with NaN padding
    right = np.full((len(questions), n_layers, max_q), np.nan)
    wrong = np.full((len(questions), n_layers, max_q), np.nan)
    qreg  = np.full((len(questions), n_layers, max_q), np.nan)
    for qi, q in enumerate(questions):
        nt = q["n_q_tokens"]
        # right_key = A if pos==1 else B
        right_k = "A" if q["position"] == 1 else "B"
        wrong_k = "B" if q["position"] == 1 else "A"
        for li, layer_entry in enumerate(q["attn_per_layer_per_q_token"][right_k]):
            right[qi, li, :nt] = layer_entry
        for li, layer_entry in enumerate(q["attn_per_layer_per_q_token"][wrong_k]):
            wrong[qi, li, :nt] = layer_entry
        for li, layer_entry in enumerate(q["attn_per_layer_per_q_token"]["Q"]):
            qreg[qi, li, :nt] = layer_entry
    return {
        "right": np.nanmean(right, axis=0),  # (n_layers, max_q)
        "wrong": np.nanmean(wrong, axis=0),
        "q":     np.nanmean(qreg,  axis=0),
        "n":     len(questions),
        "max_q": max_q,
    }

# Three figures + one summary JSON
def plot_per_q_token_trajectory(...): ...
def plot_layer_avg_vs_q_token_position(...): ...
def plot_last_vs_mean_comparison(...): ...
def write_summary_json(...): ...
```

### Outputs

| Path | What |
|---|---|
| `long-health/pair_experiment_q_token/figures/q_token_trajectory_per_layer.{png,pdf}` | Heatmap-style: rows = layers, cols = Q token index, color = attn mass on right/wrong cache. One panel per (variant, ordering, position). |
| `long-health/pair_experiment_q_token/figures/q_token_avg_attn.{png,pdf}` | Line plots: x = Q token position (normalized 0..1), y = mean attention mass on right/wrong/q averaged across layers. 4 panels (variant × ordering). Helps spot accumulation patterns. |
| `long-health/pair_experiment_q_token/figures/last_vs_mean_q_token.{png,pdf}` | Comparison: per-layer right vs wrong attention for last-token (from results.json) vs mean-over-Q-tokens (from q_token_trace.json). Side-by-side. Tests whether the drill-down 6 null result holds when averaged over Q tokens. |
| `long-health/pair_experiment_q_token/q_token_attention_summary.json` | Per-cell stats: per-layer mean-over-Q-tokens of right/wrong/q, plus the difference vs last-token. Mirrors `attention_selectivity_summary.json` shape. |

### Figure 1 — per-Q-token trajectory (the centerpiece)

For each of 4 (variant, ordering) cells, a 2-panel figure (asked_patient at
position 1, asked_patient at position 2). Each panel shows attention mass
trajectories as the model walks through the question:

- x-axis: **Q token index** (0 = first real Q token, max = last Q token)
- y-axis: attention mass (0 to ~0.5)
- 3 lines: cache_A (blue), cache_B (orange), question region (green)
- Lines averaged across both layers (taking the mean of all 36 layers) AND
  questions in the cell. Optionally a second figure breaks it out by layer.

**What to look for**: a flat-ish trajectory means the model attends ~uniformly
to both caches throughout the question. A trajectory where cache_A starts
high and cache_B grows toward the end (or vice versa) would indicate the
"accumulation" hypothesis. A spike at the very last token toward the right
patient would indicate the existing last-token snapshot is misleading.

## Execution plan

1. **Modify `scripts/run_pair_experiment.py`** — add `--record-q-token-trace`,
   `--pair-a`, `--pair-b` flags and the per-Q-token recording branch in
   `_run_instrumented_forward_single`. Smoke-test argparse with `--help`
   locally before submitting to SLURM.
2. **Write `scripts/marlowe/pair_q_token.sh`** — single-job SLURM wrapper.
3. **Submit 4 sbatch jobs** to Marlowe (one per variant × ordering). Each
   should take 20-40 min under H100. Total wall-clock ~1h if submitted
   sequentially, or ~30 min if all 4 run in parallel (4 GPUs available
   simultaneously).
4. **Verify after each job completes**: `q_token_trace.json` exists,
   `n_q_tokens > 0`, last-token slice matches the existing
   `attn_per_layer` in `results.json` (sanity check the pad/index math).
5. **Write `scripts/analyze_q_token_attention.py`** and run it.
6. **Eyeball the figures** for the accumulation pattern. Iterate on the
   figure design if needed.
7. **(Optional) write a small companion report** documenting findings, in
   `contexts/09042026/PAIR_EXPERIMENT_Q_TOKEN_FINDINGS.md`. Decision after
   seeing results — if findings are interesting, write it; if not, just
   commit the data + figures and verbally summarize.

## Verification

After all 4 SLURM jobs complete:

1. `find long-health/pair_experiment_q_token -name 'q_token_trace.json' | wc -l` → **4**
2. `find long-health/pair_experiment_q_token -name 'results.json' | wc -l` → **4**
3. For each `q_token_trace.json`: `len(per_question) == 40`
4. **Sanity check**: for question `q`, the **last token** of the per-Q-token
   trace must equal the corresponding entry in the sibling `results.json`'s
   `per_question[*].attn_per_layer`. I.e.:
   ```python
   assert q_trace["attn_per_layer_per_q_token"]["A"][layer][-1] == \
          results_q["attn_per_layer"]["A"][layer]
   ```
   for all layers, all questions. If this fails, the pad/index math in
   `_run_instrumented_forward_single` is wrong.
5. Storage check: total `q_token_trace.json` size across the 4 evals should
   be < 500 MB. If larger, switch to `.npz`.
6. After running the analysis script: 4 figure files + 1 summary JSON
   exist under `long-health/pair_experiment_q_token/figures/`.
7. Open figures visually — confirm the trajectory plot has reasonable
   axes, the lines aren't constant zero, and the per-cell counts are correct.

## Files to be created or modified

| File | Type | Purpose |
|---|---|---|
| `scripts/run_pair_experiment.py` | **modified** | Add `--record-q-token-trace`, `--pair-a`, `--pair-b` flags. ~40 LOC change, gated behind the flag (default off — canonical pipeline unaffected). |
| `scripts/marlowe/pair_q_token.sh` | new | Single-job SLURM wrapper, modeled on `pair_experiment.sh`. |
| `scripts/analyze_q_token_attention.py` | new | Standalone analysis. |
| `long-health/pair_experiment_q_token/{naive,rope_shift}/pair_patient_03_patient_04/{results.json,q_token_trace.json}` | new (run output) | 4 evals × 2 files each = 8 new files. |
| `long-health/pair_experiment_q_token/{naive,rope_shift}/pair_patient_04_patient_03/{results.json,q_token_trace.json}` | new (run output) | (included in the 8 above) |
| `long-health/pair_experiment_q_token/figures/q_token_trajectory_per_layer.{png,pdf}` | new (analysis) | Centerpiece figure |
| `long-health/pair_experiment_q_token/figures/q_token_avg_attn.{png,pdf}` | new (analysis) | Line plots |
| `long-health/pair_experiment_q_token/figures/last_vs_mean_q_token.{png,pdf}` | new (analysis) | Comparison vs last-token |
| `long-health/pair_experiment_q_token/q_token_attention_summary.json` | new (analysis) | Per-cell stats |

## Critical files to read while implementing

- `scripts/run_pair_experiment.py:252-356` — `_run_instrumented_forward_single`.
  Already inspected; the modification site is the layer loop at lines 339-352.
- `scripts/run_pair_experiment.py:381-549` — `run_pair` driver. Will need
  to thread the flag through and write the second JSON.
- `scripts/marlowe/pair_experiment.sh` — template for the new SLURM script.
  Already inspected.
- `scripts/attention_selectivity_analysis.py` — figure style template for
  the new analysis script (mostly the colormap and panel layout
  conventions).
- `long-health/pair_experiment/naive/pair_patient_03_patient_04/results.json`
  — for the verification step (last-token sanity check).

## Risk / things that could go wrong

- **Memory pressure during the instrumented forward.** The current code
  builds the full `output_attentions` tensor `(1, heads, q_len, k_len)` per
  layer. We're not changing this — just slicing more of it before the
  attentions are freed. No additional GPU memory cost. The CPU-side
  storage of the per-Q-token series is allocated lazily and freed per
  question.
- **Long questions exceeding storage estimate.** If LongHealth questions
  end up with `> 1000` real Q tokens, the JSON output gets large. The plan
  has a fallback to `.npz`. Check storage size after the first eval.
- **Pad index off-by-one.** Left padding at the prompt tokenization step
  (`tokenizer.padding_side = "left"`) means real tokens start at
  `pad_count`. With `batch_size=1` there's typically no padding, but
  the assert in step 4 of verification catches any indexing bug.
- **Pair index ergonomics.** Hard-coding `--pair-idx 40` and `--pair-idx 59`
  is error-prone. Adding `--pair-a` / `--pair-b` is the small UX investment
  that makes one-off runs robust.
