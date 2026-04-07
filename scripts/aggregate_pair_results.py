"""
Aggregate pair-stacked eval results from scripts/run_pair_experiment.py.

Reads all per-pair JSON files from long-health/pair_experiment/{variant}/pair_*
for each requested variant, writes a comprehensive per-layer attention-mass
figure (split by variant × position × correctness, with mean ± std bands), and
a summary JSON of accuracy marginals.

Accuracy heatmaps and the attn_mass_before heatmap are deliberately not
generated as figures any more — the underlying numbers live in summary.json
and the per-pair results.json files. See
contexts/06042026/PAIR_EXPERIMENT_REPORT.md for the index.

Usage:
    python -u scripts/aggregate_pair_results.py
    python -u scripts/aggregate_pair_results.py --variants naive
    python -u scripts/aggregate_pair_results.py --results-dir long-health/pair_experiment
"""
import argparse
import glob
import json
import os

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

REGION_COLORS = {
    "A": "tab:blue",
    "B": "tab:orange",
    "Q": "tab:green",
}
REGION_LABELS = {
    "A": "cache_A",
    "B": "cache_B",
    "Q": "question",
}


def _nan_matrix():
    return np.full((N, N), np.nan, dtype=float)


def _load_variant(results_dir, variant):
    pattern = os.path.join(results_dir, variant, "pair_*", "results.json")
    paths = sorted(glob.glob(pattern))
    print(f"  variant={variant}: found {len(paths)} result files")
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


def _accuracy_matrices(results):
    """Per-variant 7x7 accuracy matrices needed for the marginals in
    summary.json. Diagonal cells are NaN (no self-pairs)."""
    overall = _nan_matrix()
    pos1 = _nan_matrix()
    pos2 = _nan_matrix()
    for r in results:
        pid_a, pid_b = r["pair"]
        if pid_a not in PID_TO_IDX or pid_b not in PID_TO_IDX:
            continue
        i, j = PID_TO_IDX[pid_a], PID_TO_IDX[pid_b]
        overall[i, j] = r["overall_accuracy"]
        pos1[i, j] = r["acc_pos1"]
        pos2[i, j] = r["acc_pos2"]
    return {
        "overall": overall,
        "acc_pos1": pos1,
        "acc_pos2": pos2,
        "delta": pos2 - pos1,
    }


def _gather_attn_after(results, position, correctness):
    """For one (position, correctness) cell, gather per-pair (num_layers,)
    arrays of mean and std for each region from each pair's
    attn_mass_after_aggregate. Returns dict with stacked arrays plus the
    total sample count across pairs.

    Each pair contributes one observation per layer per region — the
    per-pair-conditional mean (and std) for that bucket. We then average
    across pairs to get the cross-pair mean and pool the per-pair stds (mean
    of stds, weighted equally) for the band. The pooled std is approximate
    but adequate for an exploratory band.
    """
    means = {region: [] for region in "ABQ"}
    stds = {region: [] for region in "ABQ"}
    n_total = 0
    n_pairs = 0
    for r in results:
        bucket = (
            r.get("attn_mass_after_aggregate", {})
            .get(f"position_{position}", {})
            .get(correctness, {})
        )
        layers = bucket.get("per_layer") if bucket else None
        n = bucket.get("n", 0) if bucket else 0
        if not layers or n == 0:
            continue
        n_total += int(n)
        n_pairs += 1
        for region in "ABQ":
            means[region].append([entry[f"{region}_mean"] for entry in layers])
            stds[region].append([entry[f"{region}_std"] for entry in layers])
    if not n_pairs:
        return None
    out = {"n_total": n_total, "n_pairs": n_pairs}
    for region in "ABQ":
        m = np.asarray(means[region], dtype=np.float64)  # (n_pairs, num_layers)
        s = np.asarray(stds[region], dtype=np.float64)
        out[f"{region}_mean"] = m.mean(axis=0)
        out[f"{region}_std"] = s.mean(axis=0)
    return out


def _plot_attention_mass(results_by_variant, out_dir):
    """Comprehensive per-layer attention-mass figure.

    Layout: 2 rows × (n_variants × 2) cols.
        rows = question position (1, 2)
        cols = (variant, correctness) — for each variant, two columns
               (correct, incorrect) side by side
    Each subplot: 3 lines (cache_A, cache_B, question) with mean ± std bands.

    The std band is the average over pairs of each pair's per-position
    per-correctness per-layer std (a rough pooled estimate, not a confidence
    interval; see contexts/06042026/ATTENTION_MASS_SPEC.md §7).
    """
    # Preserve a deterministic variant order: naive first, then rope_shift,
    # then anything else alphabetically.
    preferred = ["naive", "rope_shift"]
    variants = [v for v in preferred if v in results_by_variant]
    variants += sorted(v for v in results_by_variant if v not in preferred)
    if not variants:
        return

    col_specs = []  # list of (variant, correctness)
    for v in variants:
        for corr in ("correct", "incorrect"):
            col_specs.append((v, corr))
    n_cols = len(col_specs)

    # Pre-gather all (variant, position, correctness) cells.
    cells = {}
    for v in variants:
        for pos in (1, 2):
            for corr in ("correct", "incorrect"):
                cells[(v, pos, corr)] = _gather_attn_after(
                    results_by_variant[v], pos, corr
                )

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(3.6 * n_cols, 7.5),
        squeeze=False,
        sharey=True,
        sharex=True,
    )

    for col, (v, corr) in enumerate(col_specs):
        for row, pos in enumerate((1, 2)):
            ax = axes[row, col]
            d = cells.get((v, pos, corr))
            head = f"{v} — {corr}\nposition {pos}"
            if d is None:
                ax.set_title(f"{head}\n(no data)")
                if row == 1:
                    ax.set_xlabel("layer")
                if col == 0:
                    ax.set_ylabel("attn mass (last Q token)")
                continue
            n_layers = len(d["A_mean"])
            xs = np.arange(n_layers)
            for region in "ABQ":
                mean = d[f"{region}_mean"]
                std = d[f"{region}_std"]
                color = REGION_COLORS[region]
                ax.plot(xs, mean, color=color, label=REGION_LABELS[region], linewidth=1.5)
                ax.fill_between(
                    xs,
                    np.clip(mean - std, 0.0, 1.0),
                    np.clip(mean + std, 0.0, 1.0),
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
            ax.set_title(
                f"{head}\nn={d['n_total']} q over {d['n_pairs']} pair(s)",
                fontsize=10,
            )
            if row == 1:
                ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("attn mass (last Q token)")
            ax.set_ylim(0, 1)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "attn_mass_after — per-layer mean ± std, split by variant × position × correctness",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(out_dir, "attn_mass_after_per_layer.png")
    pdf = os.path.join(out_dir, "attn_mass_after_per_layer.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


def _marginals(mats):
    """Row means (by first-position patient) and column means (by second-position),
    plus overall and per-position scalar means and the recency-bias estimate."""
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
        mats = _accuracy_matrices(results)
        summary[variant] = _marginals(mats)

    if results_by_variant:
        _plot_attention_mass(results_by_variant, fig_dir)

    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
