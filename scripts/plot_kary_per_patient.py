"""
scripts/plot_kary_per_patient.py — per-patient accuracy breakdown across
stacking depths. One line per k value (k=2, 3, 4, 5) showing how each
patient in the cumulative-prefix stack scored on its own 20 questions
when evaluated inside a k-patient stacked cache.

Two modes:
  --variant <naive|rope_shift>  single-panel figure for one variant.
                                Output: per_patient_accuracy[_naive].{pdf,png}
  --combined                    side-by-side two-panel figure showing
                                BOTH variants. Output:
                                per_patient_accuracy_combined.{pdf,png}
                                When --combined is passed, --variant is ignored.

Reading the figure: follow a single color (=fixed k) to see how the k
patients in that stack compare to each other. Compare across x values
for a fixed color to see how the SAME patient's accuracy changes as
you embed them in deeper stacks.

x-axis: patient index (P01, P03, P04, P05, P06 — patient_02 is absent
        from the canonical ordering by design)
y-axis: accuracy
lines : one per k in {2, 3, 4, 5}

Data sources:
  k=2:  long-health/pair_experiment/<variant>/pair_patient_01_patient_03/results.json
        (uses acc_pos1 for patient_01, acc_pos2 for patient_03)
  k>=3: long-health/kary_experiment/<variant>/k{N}_<cumulative_subset>/results.json
        (uses acc_per_position array, index i = patient at stack position i+1)

Idempotent — missing k values are skipped silently.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_style("whitegrid")

# Cumulative-prefix stack ordering (matches run_kary_experiment.py layout
# and scripts/marlowe/kary{3,4,5}_single.sh).
PATIENTS = ["patient_01", "patient_03", "patient_04", "patient_05", "patient_06"]
# Actual numeric indices for the x-axis (patient_02 is missing by design —
# the anchor pair at k=2 is (01, 03), and the cumulative prefix follows
# from there). Using the real indices preserves the gap at 2 and makes the
# non-contiguous patient ordering visible.
PATIENT_INDEX = [1, 3, 4, 5, 6]

# One distinct color per k. ColorBrewer Dark2 palette — qualitative, prints OK.
K_COLORS = {
    2: "#1b9e77",  # teal
    3: "#d95f02",  # orange
    4: "#7570b3",  # purple
    5: "#e7298a",  # magenta
}


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
    # pos1 = patient_01, pos2 = patient_03
    return [r["acc_pos1"], r["acc_pos2"]]


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
    return r["acc_per_position"]  # list of length k


def load_all(root, variant):
    """Return {k: [per-patient accuracy list]} for k in {2, 3, 4, 5}."""
    return {
        2: load_k2(root, variant),
        3: load_kk(root, variant, 3),
        4: load_kk(root, variant, 4),
        5: load_kk(root, variant, 5),
    }


def print_summary(variant, by_k):
    print(f"=== Per-patient accuracy ({variant}) ===")
    for k in sorted(by_k):
        d = by_k[k]
        if d is None:
            print(f"k={k}: (missing)")
            continue
        pairs = ", ".join(
            f"{PATIENTS[i].replace('patient_', 'P')}={a:.0%}"
            for i, a in enumerate(d)
        )
        print(f"k={k}: {pairs}")


def plot_variant_on_axes(ax, by_k):
    """Draw the per-patient lines (one per k) for one variant onto `ax`."""
    # Vertical stagger for value annotations to reduce overlap between
    # overlapping k lines at the same patient index.
    y_offset_by_k = {2: 10, 3: -16, 4: 10, 5: -16}
    for k in sorted(by_k):
        d = by_k[k]
        if d is None:
            continue
        xs = PATIENT_INDEX[:k]
        ys = d
        ax.plot(
            xs,
            ys,
            color=K_COLORS[k],
            marker="o",
            markersize=8,
            linewidth=2,
            label=f"k={k}",
        )
        y_off = y_offset_by_k[k]
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.0%}",
                (x, y),
                textcoords="offset points",
                xytext=(8, y_off),
                fontsize=9,
                color=K_COLORS[k],
                fontweight="bold",
            )
    ax.set_xticks(PATIENT_INDEX)
    ax.set_xticklabels([f"P{i:02d}" for i in PATIENT_INDEX])
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])


def save_figure(fig, out_dir, stem):
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="long-health/kary_experiment/figures")
    ap.add_argument(
        "--variant",
        default="rope_shift",
        choices=["naive", "rope_shift"],
        help="Variant for single-panel mode. Ignored with --combined.",
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help="Produce a side-by-side two-panel figure with both variants.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.combined:
        data = {v: load_all(root, v) for v in ("rope_shift", "naive")}
        for variant in ("rope_shift", "naive"):
            print_summary(variant, data[variant])
            print()

        fig, (ax_rope, ax_naive) = plt.subplots(
            1, 2, figsize=(14, 5), sharey=True
        )

        plot_variant_on_axes(ax_rope, data["rope_shift"])
        ax_rope.set_title("rope_shift", fontsize=14)
        ax_rope.set_xlabel("patient index", fontsize=12)
        ax_rope.set_ylabel("accuracy", fontsize=12)
        ax_rope.legend(
            loc="lower left",
            fontsize=10,
            framealpha=0.95,
            title="stack depth",
            title_fontsize=10,
        )

        plot_variant_on_axes(ax_naive, data["naive"])
        ax_naive.set_title("naive", fontsize=14)
        ax_naive.set_xlabel("patient index", fontsize=12)

        fig.suptitle(
            "Per-patient accuracy across stacking depths\n"
            "cumulative prefix of patient_01, _03, _04, _05, _06",
            fontsize=14,
        )
        fig.tight_layout()
        save_figure(fig, out_dir, "per_patient_accuracy_combined")
    else:
        by_k = load_all(root, args.variant)
        print_summary(args.variant, by_k)

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        plot_variant_on_axes(ax, by_k)

        ax.set_xlabel("patient index", fontsize=12)
        ax.set_ylabel("accuracy", fontsize=12)
        ax.set_title(
            f"Per-patient accuracy across stacking depths ({args.variant})\n"
            f"cumulative prefix of patient_01, _03, _04, _05, _06",
            fontsize=13,
        )
        ax.legend(
            loc="lower left",
            fontsize=10,
            framealpha=0.95,
            title="stack depth",
            title_fontsize=10,
        )

        fig.tight_layout()
        suffix = "_naive" if args.variant == "naive" else ""
        save_figure(fig, out_dir, f"per_patient_accuracy{suffix}")


if __name__ == "__main__":
    main()
