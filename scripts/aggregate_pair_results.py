"""
Aggregate pair-stacked eval results from scripts/run_pair_experiment.py.

Reads all per-pair JSON files from long-health/pair_experiment/{variant}/pair_*
for each requested variant, builds 7x7 accuracy matrices (rows = first-position
patient, cols = second-position patient; diagonal masked), plots heatmaps, and
writes a summary JSON.

Usage:
    python -u scripts/aggregate_pair_results.py
    python -u scripts/aggregate_pair_results.py --variants naive
    python -u scripts/aggregate_pair_results.py --results-dir long-health/pair_experiment
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_style("whitegrid")

PATIENT_IDS = [
    "patient_01",
    "patient_03",
    "patient_04",
    "patient_05",
    "patient_06",
    "patient_07",
    "patient_08",
]
LABELS = [p.replace("patient_", "P") for p in PATIENT_IDS]
PID_TO_IDX = {p: i for i, p in enumerate(PATIENT_IDS)}
N = len(PATIENT_IDS)


def _nan_matrix():
    m = np.full((N, N), np.nan, dtype=float)
    return m


def _load_variant(results_dir, variant):
    pattern = os.path.join(results_dir, variant, "pair_*", "results.json")
    paths = sorted(glob.glob(pattern))
    print(f"  variant={variant}: found {len(paths)} result files")
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


def _build_matrices(results):
    """Return dict of 7x7 matrices keyed by metric name."""
    overall = _nan_matrix()
    pos1 = _nan_matrix()
    pos2 = _nan_matrix()
    attn_before_A = _nan_matrix()  # cache_A share from attn_mass_before, averaged over layers
    attn_after_A = _nan_matrix()   # cache_A share from attn_mass_after, averaged across layers and questions

    for r in results:
        pid_a, pid_b = r["pair"]
        if pid_a not in PID_TO_IDX or pid_b not in PID_TO_IDX:
            continue
        i, j = PID_TO_IDX[pid_a], PID_TO_IDX[pid_b]
        overall[i, j] = r["overall_accuracy"]
        pos1[i, j] = r["acc_pos1"]
        pos2[i, j] = r["acc_pos2"]
        attn_before_A[i, j] = r["attn_mass_before"]["mean_A"]
        # Average attn_mass_after cache_A share across layers and questions.
        a_vals = []
        for q in r["per_question"]:
            for entry in q["attn_mass_after"]:
                a_vals.append(entry["A"])
        if a_vals:
            attn_after_A[i, j] = float(np.mean(a_vals))

    delta = pos2 - pos1
    return {
        "overall": overall,
        "acc_pos1": pos1,
        "acc_pos2": pos2,
        "delta": delta,
        "attn_before_A": attn_before_A,
        "attn_after_A": attn_after_A,
    }


def _plot_variant_grid(mats, variant, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    specs = [
        ("overall",  axes[0, 0], "Overall accuracy", "RdYlGn", 0.0, 1.0, ".0%", False),
        ("acc_pos1", axes[0, 1], "Accuracy: first-position patient", "RdYlGn", 0.0, 1.0, ".0%", False),
        ("acc_pos2", axes[1, 0], "Accuracy: second-position patient", "RdYlGn", 0.0, 1.0, ".0%", False),
        ("delta",    axes[1, 1], "Delta (pos2 - pos1)", "RdBu_r", None, None, "+.0%", True),
    ]
    for key, ax, title, cmap, vmin, vmax, fmt, center in specs:
        m = mats[key]
        kwargs = dict(
            annot=True, fmt=fmt, cmap=cmap,
            xticklabels=LABELS, yticklabels=LABELS,
            cbar_kws={"shrink": 0.8},
            ax=ax, square=True, linewidths=0.5,
        )
        if center:
            span = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1.0
            kwargs.update(vmin=-span, vmax=span, center=0)
        else:
            kwargs.update(vmin=vmin, vmax=vmax)
        sns.heatmap(m, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("Second-position patient (B)")
        ax.set_ylabel("First-position patient (A)")

    fig.suptitle(f"Pair-stacked eval — variant: {variant}", fontsize=14)
    fig.tight_layout()
    png = os.path.join(out_dir, f"pair_accuracy_{variant}.png")
    pdf = os.path.join(out_dir, f"pair_accuracy_{variant}.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")


def _plot_cross_variant_diff(mats_by_variant, out_dir):
    if "naive" not in mats_by_variant or "rope_shift" not in mats_by_variant:
        print("  skipping cross-variant diff (need both naive and rope_shift)")
        return
    naive = mats_by_variant["naive"]
    shift = mats_by_variant["rope_shift"]
    keys = ["overall", "acc_pos1", "acc_pos2", "delta"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, key in zip(axes.flat, keys):
        diff = shift[key] - naive[key]
        span = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
        sns.heatmap(
            diff, annot=True, fmt="+.0%", cmap="RdBu_r",
            xticklabels=LABELS, yticklabels=LABELS,
            vmin=-span, vmax=span, center=0,
            cbar_kws={"shrink": 0.8},
            ax=ax, square=True, linewidths=0.5,
        )
        ax.set_title(f"{key}: rope_shift − naive")
        ax.set_xlabel("Second-position patient (B)")
        ax.set_ylabel("First-position patient (A)")
    fig.suptitle("Cross-variant difference (rope_shift minus naive)", fontsize=14)
    fig.tight_layout()
    png = os.path.join(out_dir, "pair_accuracy_diff.png")
    pdf = os.path.join(out_dir, "pair_accuracy_diff.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")


def _plot_attention_mass(results_by_variant, out_dir):
    """Per-layer line plots of attention mass by region, position, and variant."""
    variants = sorted(results_by_variant.keys())
    if not variants:
        return

    # Gather per-layer averages for each (variant, position) combination.
    # attn_mass_after is stored per question as a list of {layer, A, B, Q} dicts.
    data = {v: {1: {"A": None, "B": None, "Q": None},
                2: {"A": None, "B": None, "Q": None}}
            for v in variants}

    for v in variants:
        results = results_by_variant[v]
        if not results:
            continue
        # Gather per-position lists: sums[pos][region] -> list of per-question
        # per-layer arrays.
        per_pos_A = {1: [], 2: []}
        per_pos_B = {1: [], 2: []}
        per_pos_Q = {1: [], 2: []}
        for r in results:
            for q in r["per_question"]:
                pos = q["position"]
                layers = q["attn_mass_after"]
                per_pos_A[pos].append([l["A"] for l in layers])
                per_pos_B[pos].append([l["B"] for l in layers])
                per_pos_Q[pos].append([l["Q"] for l in layers])
        for pos in (1, 2):
            if per_pos_A[pos]:
                data[v][pos]["A"] = np.mean(np.array(per_pos_A[pos]), axis=0)
                data[v][pos]["B"] = np.mean(np.array(per_pos_B[pos]), axis=0)
                data[v][pos]["Q"] = np.mean(np.array(per_pos_Q[pos]), axis=0)

    fig, axes = plt.subplots(len(variants), 2, figsize=(12, 4 * len(variants)),
                             squeeze=False)
    for row, v in enumerate(variants):
        for col, pos in enumerate((1, 2)):
            ax = axes[row, col]
            d = data[v][pos]
            if d["A"] is None:
                ax.set_title(f"{v} — position {pos} (no data)")
                continue
            layers_idx = np.arange(len(d["A"]))
            ax.plot(layers_idx, d["A"], label="cache_A", color="tab:blue")
            ax.plot(layers_idx, d["B"], label="cache_B", color="tab:orange")
            ax.plot(layers_idx, d["Q"], label="question", color="tab:green")
            ax.set_title(f"{v} — question about pos-{pos} patient")
            ax.set_xlabel("layer index")
            ax.set_ylabel("mean attn mass (last Q token)")
            ax.set_ylim(0, 1)
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("attn_mass_after — per-layer mean over questions", fontsize=13)
    fig.tight_layout()
    png = os.path.join(out_dir, "attn_mass_after_per_layer.png")
    pdf = os.path.join(out_dir, "attn_mass_after_per_layer.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")

    # Also plot attn_mass_before as a 7x7 heatmap per variant.
    fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 5),
                             squeeze=False)
    for col, v in enumerate(variants):
        mats = _build_matrices(results_by_variant[v])
        ax = axes[0, col]
        sns.heatmap(
            mats["attn_before_A"], annot=True, fmt=".2f", cmap="coolwarm",
            vmin=0, vmax=1, center=0.5,
            xticklabels=LABELS, yticklabels=LABELS,
            cbar_kws={"shrink": 0.8}, ax=ax, square=True, linewidths=0.5,
        )
        ax.set_title(f"{v}: cache_A share of attn_mass_before")
        ax.set_xlabel("Second-position patient (B)")
        ax.set_ylabel("First-position patient (A)")
    fig.tight_layout()
    png = os.path.join(out_dir, "attn_mass_before_heatmap.png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"  wrote {png}")


def _marginals(mats):
    """Row means (by first-position patient) and column means (by second-position)."""
    def _safe_nanmean(x):
        if np.isfinite(x).any():
            return float(np.nanmean(x))
        return None

    return {
        "by_first_position": {
            LABELS[i]: _safe_nanmean(mats["overall"][i, :]) for i in range(N)
        },
        "by_second_position": {
            LABELS[j]: _safe_nanmean(mats["overall"][:, j]) for j in range(N)
        },
        "overall_mean": _safe_nanmean(mats["overall"]),
        "acc_pos1_mean": _safe_nanmean(mats["acc_pos1"]),
        "acc_pos2_mean": _safe_nanmean(mats["acc_pos2"]),
        "recency_bias_estimate": _safe_nanmean(mats["delta"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="long-health/pair_experiment")
    parser.add_argument("--variants", nargs="+", default=["naive", "rope_shift"])
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    results_by_variant = {}
    mats_by_variant = {}
    summary = {}
    for variant in args.variants:
        variant_dir = os.path.join(args.results_dir, variant)
        if not os.path.isdir(variant_dir):
            print(f"  variant={variant}: directory not found ({variant_dir}), skipping")
            continue
        results = _load_variant(args.results_dir, variant)
        if not results:
            print(f"  variant={variant}: no results found, skipping")
            continue
        results_by_variant[variant] = results
        mats = _build_matrices(results)
        mats_by_variant[variant] = mats
        _plot_variant_grid(mats, variant, fig_dir)
        summary[variant] = _marginals(mats)

    if len(mats_by_variant) >= 2:
        _plot_cross_variant_diff(mats_by_variant, fig_dir)

    if results_by_variant:
        _plot_attention_mass(results_by_variant, fig_dir)

    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
