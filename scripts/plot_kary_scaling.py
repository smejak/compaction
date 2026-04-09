"""
scripts/plot_kary_scaling.py — single-panel plot of overall accuracy
vs k (number of stacked patients) for naive vs rope_shift variants
on the cumulative-prefix subset (patient_01, _03, _04, _05, _06).

Data sources:
  k=1: long-health/patient_{01,03,04,05,06}/results.json (mean of 5
       per-patient evals — there is no variant at k=1, no stacking)
  k=2: long-health/pair_experiment/{variant}/pair_patient_01_patient_03/results.json
       (the (01,03) anchor pair only — chosen so the k=2 point is the exact
       2-prefix of the k=5 stack, not the 380-pair population mean)
  k>=3: long-health/kary_experiment/{variant}/k{N}_{cumulative_subset}/results.json

Output: long-health/kary_experiment/figures/naive_vs_rope_scaling.{pdf,png}

Idempotent: missing files are skipped silently. Re-run after k=3/k=4 (or
any other k) land to refresh the figure.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_style("whitegrid")

PATIENTS = ["patient_01", "patient_03", "patient_04", "patient_05", "patient_06"]
VARIANTS = ["naive", "rope_shift"]
COLORS = {"naive": "#d62728", "rope_shift": "#1f77b4"}  # red, blue
LABELS = {"naive": "naive", "rope_shift": "rope_shift"}


def wilson_ci(p, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_k1(root):
    accs, ns = [], []
    for p in PATIENTS:
        path = root / "long-health" / p / "results.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        accs.append(r["accuracy"])
        ns.append(r.get("total", 20))
    if not accs:
        return None
    return {"acc": sum(accs) / len(accs), "n": sum(ns), "per_patient": accs}


def load_k2(root, variant):
    path = (
        root
        / "long-health"
        / "pair_experiment"
        / variant
        / "pair_patient_01_patient_03"
        / "results.json"
    )
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    return {"acc": r["overall_accuracy"], "n": r["total"]}


def load_kk(root, variant, k):
    """Load k>=3 result with cumulative-prefix subset."""
    subset = "_".join(p.replace("patient_", "") for p in PATIENTS[:k])
    path = (
        root
        / "long-health"
        / "kary_experiment"
        / variant
        / f"k{k}_{subset}"
        / "results.json"
    )
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    return {"acc": r["overall_accuracy"], "n": r["total"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="long-health/kary_experiment/figures")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all available data ----
    k1 = load_k1(root)
    by_variant = {v: {} for v in VARIANTS}
    by_variant["naive"][2] = load_k2(root, "naive")
    by_variant["rope_shift"][2] = load_k2(root, "rope_shift")
    for k in (3, 4, 5):
        for v in VARIANTS:
            by_variant[v][k] = load_kk(root, v, k)

    # ---- Print summary ----
    print("=== Data ===")
    if k1:
        print(
            f"k=1 baseline: {k1['acc']:.3f} "
            f"(n={k1['n']} across {len(k1['per_patient'])} patients)"
        )
    for v in VARIANTS:
        for k in sorted(by_variant[v]):
            d = by_variant[v][k]
            if d is None:
                print(f"k={k} {v}: (missing)")
            else:
                lo, hi = wilson_ci(d["acc"], d["n"])
                print(
                    f"k={k} {v}: {d['acc']:.3f} "
                    f"(n={d['n']}, 95% CI [{lo:.3f}, {hi:.3f}])"
                )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # k=1 baseline as a horizontal dashed line
    if k1:
        ax.axhline(
            k1["acc"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            zorder=1,
            label=f"k=1 baseline (mean of 5) = {k1['acc']:.0%}",
        )

    # Per-variant lines + binomial error bars + value annotations
    for v in VARIANTS:
        ks, accs, los, his = [], [], [], []
        for k in sorted(by_variant[v]):
            d = by_variant[v][k]
            if d is None:
                continue
            lo, hi = wilson_ci(d["acc"], d["n"])
            ks.append(k)
            accs.append(d["acc"])
            los.append(d["acc"] - lo)
            his.append(hi - d["acc"])
        if not ks:
            continue
        ax.errorbar(
            ks,
            accs,
            yerr=[los, his],
            color=COLORS[v],
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=4,
            capthick=1.3,
            zorder=3,
            label=LABELS[v],
        )
        # value annotations: alternate sides to avoid overlap with the
        # other variant's marker
        offset = (10, 8) if v == "rope_shift" else (10, -14)
        for k, a in zip(ks, accs):
            ax.annotate(
                f"{a:.0%}",
                (k, a),
                textcoords="offset points",
                xytext=offset,
                fontsize=10,
                color=COLORS[v],
                fontweight="bold",
            )

    ax.set_xlabel("k (number of stacked patients)", fontsize=12)
    ax.set_ylabel("overall accuracy", fontsize=12)
    ax.set_title(
        "LongHealth accuracy vs stacking depth\n"
        "cumulative prefix of patient_01, _03, _04, _05, _06",
        fontsize=13,
    )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    pdf_path = out_dir / "naive_vs_rope_scaling.pdf"
    png_path = out_dir / "naive_vs_rope_scaling.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
