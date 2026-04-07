"""
Drill-down analyses for the pair-stacked KV cache experiment.

Produces what scripts/aggregate_pair_results.py does NOT:
  1. per-cell 20x20 accuracy heatmaps (one per variant)
  2. pair asymmetry: acc(A->B) vs acc(B->A) scatter + top-10 most-asymmetric pairs
  3. per-question naive vs rope_shift correctness flip contingency (2x2 by position)
  4. per-layer attention mass diff (rope_shift - naive) restricted to flipped
     questions, split by flip category (naive_only / rope_only) x position

Reads per-pair results from `<results-dir>/{variant}/pair_*/results.json`
(default: long-health/pair_experiment) and writes outputs to `<output-dir>`
(default: same — figures land in `<dir>/figures/` alongside the canonical
aggregator's figure, JSONs at `<dir>/`). All new filenames are unique so
nothing collides with the canonical aggregator's outputs.

Per-pair results.json schema (verified by direct read):
  pair: [patient_a, patient_b]
  overall_accuracy, acc_pos1, acc_pos2: float
  per_question: list of 40 dicts (20 questions x 2 positions), each:
    qid: str
    patient: str
    position: int  (1 or 2)
    correct: bool
    pred, gold: int
    attn_per_layer: {A: list[36], B: list[36], Q: list[36]}
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_style("whitegrid")

PATIENT_IDS = [f"patient_{i:02d}" for i in range(1, 21)]
LABELS = [p.replace("patient_", "P") for p in PATIENT_IDS]
PID_TO_IDX = {p: i for i, p in enumerate(PATIENT_IDS)}
N = len(PATIENT_IDS)

REGION_COLORS = {"A": "tab:blue", "B": "tab:orange", "Q": "tab:green"}
REGION_LABELS = {"A": "cache_A", "B": "cache_B", "Q": "question"}


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #

def load_variant(results_dir, variant):
    pattern = os.path.join(results_dir, variant, "pair_*", "results.json")
    paths = sorted(glob.glob(pattern))
    print(f"  variant={variant}: found {len(paths)} result files")
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


# --------------------------------------------------------------------- #
# Drill-down 1: per-cell accuracy heatmaps
# --------------------------------------------------------------------- #

def accuracy_matrix(results, key="overall_accuracy"):
    M = np.full((N, N), np.nan, dtype=float)
    for r in results:
        a, b = r["pair"]
        if a in PID_TO_IDX and b in PID_TO_IDX:
            M[PID_TO_IDX[a], PID_TO_IDX[b]] = r[key]
    return M


def plot_accuracy_heatmap(M, variant, out_path):
    overall = float(np.nanmean(M))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        M,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=overall,
        vmin=0.0,
        vmax=1.0,
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar_kws={"label": "overall_accuracy"},
        ax=ax,
        annot_kws={"fontsize": 7},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_xlabel("patient_B (second position)")
    ax.set_ylabel("patient_A (first position)")
    n_cells = int(np.isfinite(M).sum())
    ax.set_title(
        f"{variant} — per-pair overall_accuracy "
        f"(n={n_cells} pairs, mean={overall:.4f})"
    )
    fig.tight_layout()
    fig.savefig(out_path + ".png", dpi=200)
    fig.savefig(out_path + ".pdf")
    plt.close(fig)
    print(f"  wrote {out_path}.{{png,pdf}}")


# --------------------------------------------------------------------- #
# Drill-down 2: pair asymmetry (acc(A->B) vs acc(B->A))
# --------------------------------------------------------------------- #

def compute_pair_asymmetry(results):
    """For each unordered pair {A,B} present in both orderings, return
    {a, b, acc_ab, acc_ba, delta=|acc_ab - acc_ba|}, sorted by delta desc."""
    by_pair = {(r["pair"][0], r["pair"][1]): r["overall_accuracy"] for r in results}
    rows, seen = [], set()
    for (a, b), acc_ab in by_pair.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        if (b, a) not in by_pair:
            continue
        acc_ba = by_pair[(b, a)]
        rows.append(
            {
                "a": a,
                "b": b,
                "acc_ab": float(acc_ab),
                "acc_ba": float(acc_ba),
                "delta": float(abs(acc_ab - acc_ba)),
            }
        )
        seen.add(key)
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows


def plot_pair_asymmetry(asym_by_variant, out_path):
    n_v = len(asym_by_variant)
    fig, axes = plt.subplots(
        1, n_v, figsize=(5.5 * n_v, 5.5), sharex=True, sharey=True
    )
    if n_v == 1:
        axes = [axes]
    for ax, (v, rows) in zip(axes, asym_by_variant.items()):
        ab = np.array([r["acc_ab"] for r in rows])
        ba = np.array([r["acc_ba"] for r in rows])
        ax.scatter(ab, ba, alpha=0.5, s=22, edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.0)
        ax.set_xlabel("acc(A→B)")
        ax.set_ylabel("acc(B→A)")
        ax.set_title(f"{v} — pair asymmetry (n={len(rows)} unordered pairs)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path + ".png", dpi=200)
    fig.savefig(out_path + ".pdf")
    plt.close(fig)
    print(f"  wrote {out_path}.{{png,pdf}}")


# --------------------------------------------------------------------- #
# Drill-downs 3 + 4: correctness flips and per-layer attn diffs
# (single pass over per-question records produces both)
# --------------------------------------------------------------------- #

def joined_per_question(results_by_variant):
    """dict[(pair_a, pair_b, qid, position)] -> {variant: per_question_record}."""
    pq = defaultdict(dict)
    for v, results in results_by_variant.items():
        for r in results:
            a, b = r["pair"]
            for q in r["per_question"]:
                pq[(a, b, q["qid"], q["position"])][v] = q
    return pq


def correctness_flips_and_attn_diffs(joined):
    """Single pass: count contingency cells, dump flip records, gather
    per-layer attn diffs (rope_shift - naive) for flipped questions only.
    Returns (contingency, flip_records, attn_diff_aggregated)."""
    contingency = {
        1: {"both_c": 0, "both_w": 0, "n_only": 0, "r_only": 0},
        2: {"both_c": 0, "both_w": 0, "n_only": 0, "r_only": 0},
    }
    flip_records = []
    diffs = defaultdict(lambda: {"A": [], "B": [], "Q": []})

    for (a, b, qid, pos), d in joined.items():
        if "naive" not in d or "rope_shift" not in d:
            continue
        n_q = d["naive"]
        r_q = d["rope_shift"]
        nc = bool(n_q["correct"])
        rc = bool(r_q["correct"])
        cell = contingency[pos]

        if nc and rc:
            cell["both_c"] += 1
            continue
        if not nc and not rc:
            cell["both_w"] += 1
            continue

        # Flipped question — exactly one of (nc, rc) is True.
        if nc:
            cell["n_only"] += 1
            flip_type = "naive_only"
        else:
            cell["r_only"] += 1
            flip_type = "rope_only"
        flip_records.append(
            {
                "pair": [a, b],
                "qid": qid,
                "position": pos,
                "flip": flip_type,
            }
        )
        cat = (flip_type, pos)
        for region in "ABQ":
            n_attn = n_q["attn_per_layer"][region]
            r_attn = r_q["attn_per_layer"][region]
            diffs[cat][region].append([rv - nv for rv, nv in zip(r_attn, n_attn)])

    aggregated = {}
    for (flip_type, position), regions in diffs.items():
        n_records = len(regions["A"])
        if n_records == 0:
            continue
        agg = {"n": n_records}
        for region in "ABQ":
            arr = np.asarray(regions[region], dtype=np.float64)  # (n, 36)
            agg[f"{region}_mean"] = arr.mean(axis=0)
            if n_records > 1:
                agg[f"{region}_std"] = arr.std(axis=0, ddof=1)
            else:
                agg[f"{region}_std"] = np.zeros(arr.shape[1])
        aggregated[(flip_type, position)] = agg
    return contingency, flip_records, aggregated


def plot_attn_diff_flipped(aggregated, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    flip_types = ["naive_only", "rope_only"]
    flip_titles = {
        "naive_only": "naive correct, rope_shift wrong",
        "rope_only": "rope_shift correct, naive wrong",
    }
    for row, position in enumerate([1, 2]):
        for col, flip_type in enumerate(flip_types):
            ax = axes[row, col]
            d = aggregated.get((flip_type, position))
            head = f"{flip_titles[flip_type]}\nposition {position}"
            if d is None or d["n"] == 0:
                ax.set_title(f"{head}\n(no flipped questions)")
                if row == 1:
                    ax.set_xlabel("layer")
                if col == 0:
                    ax.set_ylabel("attn mass diff (rope_shift − naive)")
                continue
            n_layers = len(d["A_mean"])
            xs = np.arange(n_layers)
            for region in "ABQ":
                mean = d[f"{region}_mean"]
                std = d[f"{region}_std"]
                color = REGION_COLORS[region]
                ax.plot(
                    xs, mean, color=color, label=REGION_LABELS[region], linewidth=1.5
                )
                ax.fill_between(
                    xs, mean - std, mean + std, color=color, alpha=0.18, linewidth=0
                )
            ax.axhline(0, color="black", linestyle=":", alpha=0.6, linewidth=1.0)
            ax.set_title(f"{head}\nn={d['n']} flipped questions")
            if row == 1:
                ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("attn mass diff (rope_shift − naive)")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.suptitle(
        "Per-layer attention mass diff (rope_shift − naive) — flipped questions only",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path + ".png", dpi=200)
    fig.savefig(out_path + ".pdf")
    plt.close(fig)
    print(f"  wrote {out_path}.{{png,pdf}}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="long-health/pair_experiment")
    parser.add_argument("--output-dir", default="long-health/pair_experiment")
    parser.add_argument("--variants", nargs="+", default=["naive", "rope_shift"])
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Loading per-pair results:")
    results_by_variant = {}
    for v in args.variants:
        results_by_variant[v] = load_variant(args.results_dir, v)

    print("\n[1/4] Per-cell accuracy heatmaps:")
    for v, results in results_by_variant.items():
        if not results:
            print(f"  variant={v}: no data, skipping heatmap")
            continue
        M = accuracy_matrix(results)
        plot_accuracy_heatmap(M, v, os.path.join(fig_dir, f"accuracy_heatmap_{v}"))

    print("\n[2/4] Pair asymmetry:")
    asym = {v: compute_pair_asymmetry(rs) for v, rs in results_by_variant.items()}
    for v, rows in asym.items():
        print(f"  variant={v}: {len(rows)} unordered pairs")
    plot_pair_asymmetry(asym, os.path.join(fig_dir, "pair_asymmetry"))

    print("\n[3/4 + 4/4] Correctness flips and per-layer attn diffs (flipped only):")
    joined = joined_per_question(results_by_variant)
    print(f"  joined per-question records: {len(joined)}")
    contingency, flip_records, attn_diff_agg = correctness_flips_and_attn_diffs(joined)
    for pos in (1, 2):
        c = contingency[pos]
        total = c["both_c"] + c["both_w"] + c["n_only"] + c["r_only"]
        print(
            f"  position {pos}: both_c={c['both_c']} both_w={c['both_w']} "
            f"n_only={c['n_only']} r_only={c['r_only']} (total {total})"
        )
    plot_attn_diff_flipped(attn_diff_agg, os.path.join(fig_dir, "attn_diff_flipped"))

    extended = {
        "n_pairs_per_variant": {v: len(r) for v, r in results_by_variant.items()},
        "pair_asymmetry_top10": {v: rows[:10] for v, rows in asym.items()},
        "correctness_flip_contingency_by_position": {
            str(pos): cell for pos, cell in contingency.items()
        },
        "n_flipped_questions_by_cell": {
            f"{ft}_pos{pos}": int(d["n"])
            for (ft, pos), d in attn_diff_agg.items()
        },
    }
    summary_path = os.path.join(args.output_dir, "summary_extended.json")
    with open(summary_path, "w") as f:
        json.dump(extended, f, indent=2)
    print(f"\nwrote {summary_path}")

    flips_path = os.path.join(args.output_dir, "correctness_flips.json")
    with open(flips_path, "w") as f:
        json.dump(flip_records, f, indent=2)
    print(f"wrote {flips_path} ({len(flip_records)} records)")

    print("\ndone")


if __name__ == "__main__":
    main()
