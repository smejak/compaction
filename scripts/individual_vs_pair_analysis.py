"""
Individual-vs-paired patient accuracy comparison.

For each of the 20 LongHealth patients X, compares five accuracy numbers, all
on the SAME 20 questions about X with the SAME compacted KV cache (ratio=0.1)
and the SAME Qwen3-4B model:

  1. individual    — single-patient eval, from long-health/patient_<XX>/results.json
                     (20 questions, 1 eval).
  2. naive_pos1    — when X is at position 1 in a naive-concat stacked pair,
                     averaged over the 19 pairs (X, Y) for Y != X.
                     n = 19 × 20 = 380 question instances about X.
  3. naive_pos2    — when X is at position 2, averaged over (Y, X) pairs.  n = 380.
  4. rope_pos1     — same as (2) but with rope_shift remapping of cache_B keys.
  5. rope_pos2     — same as (3) but rope_shift.

The per-patient marginals (2)–(5) are computed by walking the `per_question`
records of all 760 per-pair results.json files and counting `(correct, total)`
per (asked-patient, position, variant) cell. We do NOT use summary.json's
`by_first_position` / `by_second_position` because those are row/column means
of the overall pair-accuracy matrix — i.e., they average over BOTH patients'
questions in each pair, not just over questions about the asked-about patient.
That's a different (and less useful for this comparison) quantity.

Outputs (relative to --output-dir, default `long-health/pair_experiment`):
  figures/individual_vs_paired.{png,pdf}   — single-panel grouped bar chart,
                                              100 bars (20 patients × 5 bars).
  individual_vs_paired_summary.json        — per-patient table + summary stats
                                              (largest gainers/losers per variant).

Schema notes (verified by Read):
- per-patient results.json has top-level `accuracy` field — use it directly.
- per-pair results.json has `per_question` (40 records). Each record has
  `patient` (asked-about patient ID), `position` (1 or 2), and `correct` (bool).
- Each (asked-patient, position) cell should have exactly 380 question
  instances per variant (19 pairs × 20 questions). Verified by sum check.
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

PATIENT_IDS = [f"patient_{i:02d}" for i in range(1, 21)]
N = len(PATIENT_IDS)

# 5-bar grouping with sequential color pairs:
#   gray   = individual baseline
#   blues  = naive  (light=pos1, dark=pos2)
#   oranges= rope_shift (light=pos1, dark=pos2)
BAR_SPECS = [
    ("individual",  "#6b6b6b"),
    ("naive_pos1",  "#9ecae1"),
    ("naive_pos2",  "#3182bd"),
    ("rope_pos1",   "#fdae6b"),
    ("rope_pos2",   "#e6550d"),
]
BAR_LABELS = {
    "individual":  "individual (single-patient)",
    "naive_pos1":  "naive — patient at position 1",
    "naive_pos2":  "naive — patient at position 2",
    "rope_pos1":   "rope_shift — patient at position 1",
    "rope_pos2":   "rope_shift — patient at position 2",
}


def pid_to_pkey(pid: str) -> str:
    """`patient_01` -> `P01`"""
    return f"P{int(pid.split('_')[1]):02d}"


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #

def load_individual_baselines(per_patient_dir: Path) -> dict:
    out = {}
    for pid in PATIENT_IDS:
        results_path = per_patient_dir / pid / "results.json"
        with open(results_path) as f:
            d = json.load(f)
        out[pid] = float(d["accuracy"])
    assert len(out) == N, f"expected {N} patients, got {len(out)}"
    return out


def load_per_pair_results(results_dir: str, variant: str) -> list:
    pattern = os.path.join(results_dir, variant, "pair_*", "results.json")
    paths = sorted(glob.glob(pattern))
    print(f"  variant={variant}: found {len(paths)} per-pair files")
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def compute_per_patient_marginals(results_by_variant: dict) -> dict:
    """Walks per_question records to compute, per (variant, patient, position),
    accuracy = correct / total across all pairs where that patient is the
    asked-about patient at that position.

    Returns:
      {variant: {position: {patient_id: {"correct": int, "total": int, "acc": float}}}}
    """
    out = {}
    for variant, results in results_by_variant.items():
        cells = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            for q in r["per_question"]:
                pid = q["patient"]
                pos = q["position"]
                cells[(pid, pos)]["total"] += 1
                if q["correct"]:
                    cells[(pid, pos)]["correct"] += 1
        # Re-shape into nested dict for ease of use
        nested = {1: {}, 2: {}}
        for (pid, pos), c in cells.items():
            acc = c["correct"] / c["total"] if c["total"] > 0 else float("nan")
            nested[pos][pid] = {"correct": c["correct"], "total": c["total"], "acc": acc}
        # Sanity: every patient should have 380 instances per position
        for pos in (1, 2):
            for pid in PATIENT_IDS:
                assert pid in nested[pos], (variant, pos, pid)
                assert nested[pos][pid]["total"] == 380, (
                    variant, pos, pid, nested[pos][pid]["total"]
                )
        out[variant] = nested
    return out


def load_pair_overall_means(pair_summary_path: Path) -> dict:
    """Reads summary.json's overall_mean per variant — used only for the
    horizontal reference line in the plot, not for any per-patient logic."""
    with open(pair_summary_path) as f:
        d = json.load(f)
    return {v: {
        "overall_mean": float(d[v]["overall_mean"]),
        "acc_pos1_mean": float(d[v]["acc_pos1_mean"]),
        "acc_pos2_mean": float(d[v]["acc_pos2_mean"]),
        "recency_bias_estimate": float(d[v]["recency_bias_estimate"]),
    } for v in ("naive", "rope_shift")}


# --------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------- #

def build_comparison_table(individual: dict, marginals: dict) -> list:
    rows = []
    for pid in PATIENT_IDS:
        ind = individual[pid]
        np1 = marginals["naive"][1][pid]["acc"]
        np2 = marginals["naive"][2][pid]["acc"]
        rp1 = marginals["rope_shift"][1][pid]["acc"]
        rp2 = marginals["rope_shift"][2][pid]["acc"]
        rows.append({
            "patient": pid,
            "individual": ind,
            "naive_pos1": np1,
            "naive_pos2": np2,
            "rope_pos1": rp1,
            "rope_pos2": rp2,
            "delta_naive_pos1": np1 - ind,
            "delta_naive_pos2": np2 - ind,
            "delta_rope_pos1":  rp1 - ind,
            "delta_rope_pos2":  rp2 - ind,
            "delta_naive_avg": (np1 + np2) / 2 - ind,
            "delta_rope_avg":  (rp1 + rp2) / 2 - ind,
            # Question counts (verified == 380 each at load time)
            "n_naive_pos1": marginals["naive"][1][pid]["total"],
            "n_naive_pos2": marginals["naive"][2][pid]["total"],
            "n_rope_pos1":  marginals["rope_shift"][1][pid]["total"],
            "n_rope_pos2":  marginals["rope_shift"][2][pid]["total"],
        })
    return rows


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

def plot_individual_vs_paired(rows: list, overall_means: dict, out_path: str) -> None:
    n_patients = len(rows)
    n_bars = len(BAR_SPECS)
    bar_width = 0.16
    inter_group_gap = 0.45  # extra space between patient groups
    group_span = bar_width * n_bars + inter_group_gap

    group_centers = np.arange(n_patients) * group_span
    bar_offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(20, 6))

    for i, (key, color) in enumerate(BAR_SPECS):
        vals = np.array([r[key] for r in rows])
        ax.bar(
            group_centers + bar_offsets[i],
            vals,
            width=bar_width,
            label=BAR_LABELS[key],
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    individual_mean = float(np.mean([r["individual"] for r in rows]))
    naive_overall = overall_means["naive"]["overall_mean"]
    rope_overall = overall_means["rope_shift"]["overall_mean"]

    ax.axhline(
        individual_mean, color="#3a3a3a", linestyle="--", linewidth=1.2, alpha=0.75,
        label=f"individual mean across 20 patients ({individual_mean:.3f})",
    )
    ax.axhline(
        naive_overall, color="#3182bd", linestyle="--", linewidth=1.2, alpha=0.75,
        label=f"naive overall mean ({naive_overall:.3f})",
    )
    ax.axhline(
        rope_overall, color="#e6550d", linestyle="--", linewidth=1.2, alpha=0.75,
        label=f"rope_shift overall mean ({rope_overall:.3f})",
    )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([pid_to_pkey(pid) for pid in PATIENT_IDS], fontsize=9)
    ax.set_xlabel("patient")
    ax.set_ylabel("accuracy on questions about that patient")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_title(
        "Individual baseline vs paired-stacked accuracy per patient  "
        f"(n={n_patients}, ratio=0.1 compression, Qwen3-4B, LongHealth)\n"
        "individual: 20 questions • each paired bar: 19 pairs × 20 questions = 380 question instances"
    )
    ax.legend(loc="lower right", ncol=4, fontsize=7.5, framealpha=0.92)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.margins(x=0.005)

    fig.tight_layout()
    fig.savefig(out_path + ".png", dpi=160)
    fig.savefig(out_path + ".pdf")
    plt.close(fig)
    print(f"  wrote {out_path}.{{png,pdf}}")


# --------------------------------------------------------------------- #
# Summary JSON
# --------------------------------------------------------------------- #

def write_summary_json(rows: list, overall_means: dict, out_path: str) -> None:
    individual = np.array([r["individual"] for r in rows])
    naive_pos1 = np.array([r["naive_pos1"] for r in rows])
    naive_pos2 = np.array([r["naive_pos2"] for r in rows])
    rope_pos1 = np.array([r["rope_pos1"] for r in rows])
    rope_pos2 = np.array([r["rope_pos2"] for r in rows])
    delta_naive_avg = np.array([r["delta_naive_avg"] for r in rows])
    delta_rope_avg = np.array([r["delta_rope_avg"] for r in rows])

    naive_sorted = sorted(rows, key=lambda r: r["delta_naive_avg"])
    rope_sorted = sorted(rows, key=lambda r: r["delta_rope_avg"])

    summary = {
        "n_patients": len(rows),
        "per_patient": rows,
        "summary_means": {
            "individual_mean": float(individual.mean()),
            "naive_pos1_mean": float(naive_pos1.mean()),
            "naive_pos2_mean": float(naive_pos2.mean()),
            "rope_pos1_mean": float(rope_pos1.mean()),
            "rope_pos2_mean": float(rope_pos2.mean()),
            "delta_naive_avg_mean": float(delta_naive_avg.mean()),
            "delta_rope_avg_mean": float(delta_rope_avg.mean()),
        },
        "pair_overall_means_from_summary_json": {
            "naive_overall": overall_means["naive"]["overall_mean"],
            "rope_shift_overall": overall_means["rope_shift"]["overall_mean"],
            "naive_recency_bias": overall_means["naive"]["recency_bias_estimate"],
            "rope_shift_recency_bias": overall_means["rope_shift"]["recency_bias_estimate"],
        },
        "largest_naive_losers_top5": [
            {"patient": r["patient"], "delta_naive_avg": r["delta_naive_avg"]}
            for r in naive_sorted[:5]
        ],
        "largest_naive_gainers_top5": [
            {"patient": r["patient"], "delta_naive_avg": r["delta_naive_avg"]}
            for r in naive_sorted[-5:][::-1]
        ],
        "largest_rope_losers_top5": [
            {"patient": r["patient"], "delta_rope_avg": r["delta_rope_avg"]}
            for r in rope_sorted[:5]
        ],
        "largest_rope_gainers_top5": [
            {"patient": r["patient"], "delta_rope_avg": r["delta_rope_avg"]}
            for r in rope_sorted[-5:][::-1]
        ],
        "method_note": (
            "Per-patient marginals are computed by walking per_question records "
            "in all 760 per-pair results.json files and counting correct/total "
            "per (asked-patient, position, variant). Each (patient, position, "
            "variant) cell contains exactly 380 question instances (19 pairs × "
            "20 questions). NOT the same as summary.json's by_first_position / "
            "by_second_position, which are row/column means of the OVERALL pair "
            "accuracy matrix (those average over both patients' questions in "
            "each pair, not just over questions about the asked-about patient)."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-patient-dir", default="long-health")
    ap.add_argument("--results-dir", default="long-health/pair_experiment")
    ap.add_argument("--output-dir", default="long-health/pair_experiment")
    args = ap.parse_args()

    per_patient_dir = Path(args.per_patient_dir)
    pair_summary_path = Path(args.results_dir) / "summary.json"
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading individual baselines:")
    individual = load_individual_baselines(per_patient_dir)
    print(f"  loaded {len(individual)} patients")

    print("\nLoading per-pair results:")
    results_by_variant = {
        v: load_per_pair_results(args.results_dir, v) for v in ("naive", "rope_shift")
    }

    print("\nComputing per-asked-patient marginals from per_question records:")
    marginals = compute_per_patient_marginals(results_by_variant)
    for v in ("naive", "rope_shift"):
        n_inst = sum(c["total"] for pos in (1, 2) for c in marginals[v][pos].values())
        print(f"  variant={v}: {n_inst} question instances "
              f"(expected 20 patients × 2 positions × 380 = 15200)")

    print("\nLoading overall_mean reference lines from summary.json:")
    overall_means = load_pair_overall_means(pair_summary_path)
    print(f"  naive overall:      {overall_means['naive']['overall_mean']:.4f}")
    print(f"  rope_shift overall: {overall_means['rope_shift']['overall_mean']:.4f}")

    print("\nBuilding comparison table:")
    rows = build_comparison_table(individual, marginals)
    print(f"  built {len(rows)} rows")

    print("\nTop-line statistics across 20 patients:")
    print(f"  mean individual:        {float(np.mean([r['individual']  for r in rows])):.4f}")
    print(f"  mean naive_pos1:        {float(np.mean([r['naive_pos1']  for r in rows])):.4f}")
    print(f"  mean naive_pos2:        {float(np.mean([r['naive_pos2']  for r in rows])):.4f}")
    print(f"  mean rope_pos1:         {float(np.mean([r['rope_pos1']   for r in rows])):.4f}")
    print(f"  mean rope_pos2:         {float(np.mean([r['rope_pos2']   for r in rows])):.4f}")
    print(f"  mean Δ naive (avg pos): {float(np.mean([r['delta_naive_avg'] for r in rows])):+.4f}")
    print(f"  mean Δ rope  (avg pos): {float(np.mean([r['delta_rope_avg']  for r in rows])):+.4f}")

    print("\nPlotting individual_vs_paired figure:")
    plot_individual_vs_paired(
        rows, overall_means, str(fig_dir / "individual_vs_paired")
    )

    print("\nWriting summary JSON:")
    write_summary_json(
        rows, overall_means, str(output_dir / "individual_vs_paired_summary.json")
    )

    print("\ndone")


if __name__ == "__main__":
    main()
