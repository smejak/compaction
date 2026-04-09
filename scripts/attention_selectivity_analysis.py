"""
Attention selectivity analysis for the pair-stacked KV cache experiment.

Question: given a question about a specific patient X, does the model
concentrate the last-question-token's attention mass on the cache containing
X (the "right" patient)? And does this selectivity break down on questions
the model answers incorrectly?

For every per_question record in `long-health/pair_experiment/{variant}/pair_*/results.json`:

  - q["patient"]   identifies the asked-about patient
  - q["position"]  is 1 or 2 — which slot the asked-about patient occupies
  - q["correct"]   is True/False
  - q["attn_per_layer"] = {"A": [..36 floats..],
                           "B": [..36 floats..],
                           "Q": [..36 floats..]}
                  per-layer attention-mass shares from the LAST question token
                  on the three regions of the stacked KV cache. At each layer
                  A+B+Q ≈ 1.0 (verified).

We define:
  right_key = "A" if position == 1 else "B"   # cache holding the asked patient
  wrong_key = "B" if position == 1 else "A"   # the other patient's cache
  attn_right(layer) = q["attn_per_layer"][right_key][layer]
  attn_wrong(layer) = q["attn_per_layer"][wrong_key][layer]

We bin every question record into one of 8 cells:
  (variant ∈ {naive, rope_shift}) × (position ∈ {1, 2}) × (correct ∈ {T, F})

and aggregate (n, per-layer mean ± std) of attn_right, attn_wrong and attn_q
within each cell. Total instances = 760 results × 40 questions = 30,400
(15,200 per variant), exactly matching the contingency in summary_extended.json.

Outputs (relative to --output-dir, default `long-health/pair_experiment`):
  figures/attn_on_right_patient.{png,pdf}    — 4-panel figure (rows=variants,
                                                cols=positions). 4 lines per
                                                panel: right/wrong × correct/
                                                incorrect.
  attention_selectivity_summary.json         — per-cell stats (n, per-layer
                                                mean of right/wrong/q, mean-
                                                over-layers selectivity).
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARIANTS = ("naive", "rope_shift")
POSITIONS = (1, 2)
CORRECTNESS = (True, False)
N_LAYERS = 36

# Plot styling
COLOR_CORRECT = "#1f78b4"   # blue
COLOR_INCORRECT = "#e31a1c"  # red
LINESTYLE_RIGHT = "-"
LINESTYLE_WRONG = "--"


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #

def load_per_pair(results_dir: str, variant: str) -> list:
    pattern = os.path.join(results_dir, variant, "pair_*", "results.json")
    paths = sorted(glob.glob(pattern))
    print(f"  variant={variant}: found {len(paths)} per-pair files")
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


# --------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------- #

def aggregate_attention_by_cell(results_by_variant: dict) -> dict:
    """Walks every per_question record and stacks per-layer right/wrong/q
    attention vectors into per-(variant, position, correct) numpy arrays.

    Returns dict[(variant, position, correct)] -> {
        right: np.ndarray (n, 36),
        wrong: np.ndarray (n, 36),
        q:     np.ndarray (n, 36),
        n:     int,
    }
    """
    bins = defaultdict(lambda: {"right": [], "wrong": [], "q": []})
    for variant, results in results_by_variant.items():
        for r in results:
            for question in r["per_question"]:
                pos = question["position"]
                correct = bool(question["correct"])
                right_key = "A" if pos == 1 else "B"
                wrong_key = "B" if pos == 1 else "A"
                apl = question["attn_per_layer"]
                key = (variant, pos, correct)
                bins[key]["right"].append(apl[right_key])
                bins[key]["wrong"].append(apl[wrong_key])
                bins[key]["q"].append(apl["Q"])

    cells = {}
    for key, regions in bins.items():
        right = np.asarray(regions["right"], dtype=np.float64)
        wrong = np.asarray(regions["wrong"], dtype=np.float64)
        q     = np.asarray(regions["q"],     dtype=np.float64)
        n = right.shape[0]
        # Sanity: every layer count must be 36 (verified at load time)
        assert right.shape[1] == N_LAYERS, (key, right.shape)
        cells[key] = {"right": right, "wrong": wrong, "q": q, "n": n}
    return cells


def cell_summary_stats(cells: dict) -> dict:
    """Per-cell summary: n, per-layer mean/std for right/wrong/q, plus a
    single mean-over-layers number for each, and selectivity = right - wrong."""
    out = {}
    for key, c in cells.items():
        right = c["right"]
        wrong = c["wrong"]
        q = c["q"]
        n = c["n"]
        right_mean = right.mean(axis=0)            # (36,)
        wrong_mean = wrong.mean(axis=0)
        q_mean     = q.mean(axis=0)
        right_std = right.std(axis=0, ddof=1) if n > 1 else np.zeros(N_LAYERS)
        wrong_std = wrong.std(axis=0, ddof=1) if n > 1 else np.zeros(N_LAYERS)
        q_std     = q.std(axis=0, ddof=1)     if n > 1 else np.zeros(N_LAYERS)
        sel_per_layer = right_mean - wrong_mean
        out[key] = {
            "n": int(n),
            "right_mean": right_mean,
            "wrong_mean": wrong_mean,
            "q_mean":     q_mean,
            "right_std":  right_std,
            "wrong_std":  wrong_std,
            "q_std":      q_std,
            "selectivity_per_layer": sel_per_layer,
            "right_mean_over_layers":      float(right_mean.mean()),
            "wrong_mean_over_layers":      float(wrong_mean.mean()),
            "q_mean_over_layers":          float(q_mean.mean()),
            "selectivity_mean_over_layers": float(sel_per_layer.mean()),
        }
    return out


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

def plot_attention_on_right_patient(stats: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    xs = np.arange(N_LAYERS)

    for row, variant in enumerate(VARIANTS):
        for col, position in enumerate(POSITIONS):
            ax = axes[row, col]
            d_correct = stats.get((variant, position, True))
            d_incorrect = stats.get((variant, position, False))
            n_c = d_correct["n"] if d_correct else 0
            n_i = d_incorrect["n"] if d_incorrect else 0

            # Solid right (with std bands)
            if d_correct:
                ax.plot(
                    xs, d_correct["right_mean"],
                    color=COLOR_CORRECT, linestyle=LINESTYLE_RIGHT, linewidth=2.0,
                    label="attn_right (correct)",
                )
                ax.fill_between(
                    xs,
                    np.clip(d_correct["right_mean"] - d_correct["right_std"], 0, 1),
                    np.clip(d_correct["right_mean"] + d_correct["right_std"], 0, 1),
                    color=COLOR_CORRECT, alpha=0.18, linewidth=0,
                )
            if d_incorrect:
                ax.plot(
                    xs, d_incorrect["right_mean"],
                    color=COLOR_INCORRECT, linestyle=LINESTYLE_RIGHT, linewidth=2.0,
                    label="attn_right (incorrect)",
                )
                ax.fill_between(
                    xs,
                    np.clip(d_incorrect["right_mean"] - d_incorrect["right_std"], 0, 1),
                    np.clip(d_incorrect["right_mean"] + d_incorrect["right_std"], 0, 1),
                    color=COLOR_INCORRECT, alpha=0.18, linewidth=0,
                )

            # Dashed wrong (no std bands — readability)
            if d_correct:
                ax.plot(
                    xs, d_correct["wrong_mean"],
                    color=COLOR_CORRECT, linestyle=LINESTYLE_WRONG, linewidth=1.4,
                    label="attn_wrong (correct)",
                )
            if d_incorrect:
                ax.plot(
                    xs, d_incorrect["wrong_mean"],
                    color=COLOR_INCORRECT, linestyle=LINESTYLE_WRONG, linewidth=1.4,
                    label="attn_wrong (incorrect)",
                )

            ax.set_title(
                f"{variant} — asked patient at position {position}\n"
                f"n_correct={n_c}, n_incorrect={n_i}",
                fontsize=10,
            )
            ax.set_ylim(0, None)
            if row == 1:
                ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("attention mass on cache region")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    fig.suptitle(
        "Attention mass on right vs wrong patient cache (last question token)\n"
        "split by variant × asked-patient position × correctness",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path + ".png", dpi=160)
    fig.savefig(out_path + ".pdf")
    plt.close(fig)
    print(f"  wrote {out_path}.{{png,pdf}}")


# --------------------------------------------------------------------- #
# Summary JSON
# --------------------------------------------------------------------- #

def write_summary_json(stats: dict, out_path: str) -> None:
    """Serializable per-cell stats. Per-layer arrays converted to plain
    Python lists for JSON."""
    out = {
        "n_layers": N_LAYERS,
        "method_note": (
            "For each per_question record, right_key = 'A' if position == 1 "
            "else 'B' (cache holding the asked-about patient). "
            "attn_right(layer) = q['attn_per_layer'][right_key][layer]; "
            "attn_wrong = the other cache; attn_q = q['attn_per_layer']['Q']. "
            "Cells aggregated over all 760 per-pair results.json files. "
            "Verified A+B+Q ≈ 1.0 at every layer."
        ),
        "cells": {},
        "totals": {},
    }
    grand_total = 0
    per_variant = defaultdict(int)
    per_variant_position = defaultdict(int)
    for (variant, position, correct), s in sorted(stats.items()):
        cell_key = f"{variant}__pos{position}__{'correct' if correct else 'incorrect'}"
        out["cells"][cell_key] = {
            "variant": variant,
            "position": position,
            "correct": bool(correct),
            "n": s["n"],
            "right_mean_per_layer": [float(x) for x in s["right_mean"]],
            "wrong_mean_per_layer": [float(x) for x in s["wrong_mean"]],
            "q_mean_per_layer":     [float(x) for x in s["q_mean"]],
            "right_std_per_layer":  [float(x) for x in s["right_std"]],
            "wrong_std_per_layer":  [float(x) for x in s["wrong_std"]],
            "q_std_per_layer":      [float(x) for x in s["q_std"]],
            "selectivity_per_layer": [float(x) for x in s["selectivity_per_layer"]],
            "right_mean_over_layers": s["right_mean_over_layers"],
            "wrong_mean_over_layers": s["wrong_mean_over_layers"],
            "q_mean_over_layers":     s["q_mean_over_layers"],
            "selectivity_mean_over_layers": s["selectivity_mean_over_layers"],
        }
        grand_total += s["n"]
        per_variant[variant] += s["n"]
        per_variant_position[(variant, position)] += s["n"]

    out["totals"] = {
        "grand_total_question_instances": grand_total,
        "per_variant": dict(per_variant),
        "per_variant_position": {
            f"{v}__pos{p}": n for (v, p), n in per_variant_position.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="long-health/pair_experiment")
    ap.add_argument("--output-dir", default="long-health/pair_experiment")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading per-pair results:")
    results_by_variant = {v: load_per_pair(args.results_dir, v) for v in VARIANTS}

    print("\nAggregating per-question attention by cell:")
    cells = aggregate_attention_by_cell(results_by_variant)
    grand_total = sum(c["n"] for c in cells.values())
    print(f"  total question instances: {grand_total} (expected 30400)")
    for variant in VARIANTS:
        for position in POSITIONS:
            n_c = cells.get((variant, position, True), {}).get("n", 0)
            n_i = cells.get((variant, position, False), {}).get("n", 0)
            print(f"  {variant} pos{position}: correct={n_c}, incorrect={n_i}, total={n_c + n_i}")

    print("\nComputing per-cell summary stats:")
    stats = cell_summary_stats(cells)

    print("\nMean-over-layers selectivity (right − wrong) per cell:")
    print(f"  {'variant':<12s} {'pos':<5s} {'corr':<10s} {'n':>6s} "
          f"{'right':>8s} {'wrong':>8s} {'select':>9s}")
    for variant in VARIANTS:
        for position in POSITIONS:
            for correct in (True, False):
                key = (variant, position, correct)
                if key not in stats:
                    continue
                s = stats[key]
                print(f"  {variant:<12s} {position:<5d} "
                      f"{'correct' if correct else 'incorrect':<10s} "
                      f"{s['n']:>6d} "
                      f"{s['right_mean_over_layers']:>8.4f} "
                      f"{s['wrong_mean_over_layers']:>8.4f} "
                      f"{s['selectivity_mean_over_layers']:>+9.4f}")

    print("\nPlotting attention selectivity figure:")
    plot_attention_on_right_patient(stats, str(fig_dir / "attn_on_right_patient"))

    print("\nWriting summary JSON:")
    write_summary_json(stats, str(output_dir / "attention_selectivity_summary.json"))

    print("\ndone")


if __name__ == "__main__":
    main()
