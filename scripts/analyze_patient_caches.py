"""
Analyze per-patient compacted KV caches in long-health/patient_XX/cache.pt.

Reports both *memory* compression (governed by mean_l(t_l), i.e. the per-layer
padded K-tensor lengths summed over layers) and *attention* compression
(governed by the dense-equivalent count of real, non-padded K/V slots). These
two numbers can differ by ~7x under the optimized_agnostic.json budget profile
because that profile concentrates most of the per-head budget into a small
number of "greedy" heads which then dominate their layer's padded length.

This script is read-only — it does NOT modify any cache.pt or results.json
files. It only loads CPU tensors and inspects shapes + beta-mask finiteness.

Usage:
    python scripts/analyze_patient_caches.py
    python scripts/analyze_patient_caches.py --patients 1-5,9,12-14
    python scripts/analyze_patient_caches.py --json long-health/cache_compression_summary.json
    python scripts/analyze_patient_caches.py --per-layer patient_09

The padding sentinel is -inf beta, written by per_layer_head_on_policy.py:486
and 587. Real positions have finite beta (including those at
drop_key_beta_cutoff = -7, which are real but heavily down-weighted).
"""
import argparse
import json
import os
import sys
from statistics import mean, median

import torch


def parse_patients(spec: str) -> list:
    """Parse '1-5,9,12-14' → [1,2,3,4,5,9,12,13,14]."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def analyze_patient(cache_path: str, results_path: str) -> dict:
    """Load one cache.pt + results.json and return a stats dict."""
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = blob["cache"]                          # list of (C1, beta, C2)
    original_seq_len = int(blob["original_seq_len"])
    article_len = int(blob["article_len"])
    ratio = float(blob["ratio"])
    non_article = original_seq_len - article_len

    # Pull accuracy from results.json (does not contain seq info; just acc).
    accuracy = None
    if os.path.exists(results_path):
        with open(results_path) as f:
            r = json.load(f)
        accuracy = float(r.get("accuracy", 0.0))

    n_layers = len(cache)
    per_layer_t = []                # t_l = c1.shape[2] for each layer
    per_layer_real_total = []       # sum over heads of real (non-inf) slots
    per_layer_per_head_real = []    # list[list[int]] (n_layers, kv_heads)
    kv_heads = None

    for (C1, beta, C2) in cache:
        # C1: (1, KV, t, D); beta: (1, KV, t)
        if kv_heads is None:
            kv_heads = int(beta.shape[1])
        t_l = int(C1.shape[2])
        per_layer_t.append(t_l)
        real_mask = torch.isfinite(beta[0])                  # (KV, t)
        per_head_real = real_mask.sum(dim=-1).tolist()       # length KV
        per_layer_per_head_real.append(per_head_real)
        per_layer_real_total.append(int(sum(per_head_real)))

    total_real_slots = sum(per_layer_real_total)
    total_padded_slots = sum(t_l * kv_heads for t_l in per_layer_t)
    dense_equivalent = total_real_slots / (n_layers * kv_heads)

    max_l = max(per_layer_t)
    min_l = min(per_layer_t)
    mean_l = mean(per_layer_t)
    p50_l = median(per_layer_t)

    # Compaction target reconstructed from cache metadata.
    article_target = max(1, int(article_len * ratio))
    target = article_target + non_article

    # Per-head extremes across (l, h).
    flat_real = [r for layer in per_layer_per_head_real for r in layer]
    best_head = max(flat_real)
    worst_head = min(flat_real)

    return {
        "n_layers": n_layers,
        "kv_heads": kv_heads,
        "original_seq_len": original_seq_len,
        "article_len": article_len,
        "non_article_len": non_article,
        "ratio": ratio,
        "target": target,
        "max_layer_len": max_l,
        "min_layer_len": min_l,
        "mean_layer_len": mean_l,
        "p50_layer_len": p50_l,
        "per_layer_t": per_layer_t,
        "per_layer_real_total": per_layer_real_total,
        "per_layer_per_head_real": per_layer_per_head_real,
        "total_real_slots": total_real_slots,
        "total_padded_slots": total_padded_slots,
        "dense_equivalent": dense_equivalent,
        "memory_compression": original_seq_len / mean_l,
        "attention_compression": original_seq_len / dense_equivalent,
        "padding_waste_frac": 1.0 - (total_real_slots / total_padded_slots),
        "best_head_real": best_head,
        "worst_head_real": worst_head,
        "realized_vs_target": dense_equivalent / target,
        "accuracy": accuracy,
    }


def fmt_int(x):
    return f"{int(round(x)):>5d}"


def print_summary_table(rows: list):
    """Table 1: one row per patient, plus a final mean row."""
    header = (
        f"{'patient':<11} {'orig':>5} {'article':>7} {'tgt':>5} "
        f"{'max_l':>5} {'mean_l':>6} {'min_l':>5} {'p50_l':>5} "
        f"{'dense':>5} {'mem_x':>6} {'att_x':>6} {'pad%':>5} {'acc':>5}"
    )
    print(header)
    print("-" * len(header))

    def row_str(name, s):
        acc = f"{s['accuracy']:.2f}" if s["accuracy"] is not None else "  -"
        return (
            f"{name:<11} "
            f"{fmt_int(s['original_seq_len'])} "
            f"{fmt_int(s['article_len']):>7} "
            f"{fmt_int(s['target']):>5} "
            f"{fmt_int(s['max_layer_len']):>5} "
            f"{fmt_int(s['mean_layer_len']):>6} "
            f"{fmt_int(s['min_layer_len']):>5} "
            f"{fmt_int(s['p50_layer_len']):>5} "
            f"{fmt_int(s['dense_equivalent']):>5} "
            f"{s['memory_compression']:>5.2f}x "
            f"{s['attention_compression']:>5.2f}x "
            f"{100*s['padding_waste_frac']:>4.1f}% "
            f"{acc:>5}"
        )

    for name, s in rows:
        print(row_str(name, s))

    # Mean row across all patients.
    if not rows:
        return
    keys_num = ["original_seq_len", "article_len", "target", "max_layer_len",
                "mean_layer_len", "min_layer_len", "p50_layer_len",
                "dense_equivalent", "memory_compression",
                "attention_compression", "padding_waste_frac"]
    mean_stats = {k: mean(s[k] for _, s in rows) for k in keys_num}
    accs = [s["accuracy"] for _, s in rows if s["accuracy"] is not None]
    mean_stats["accuracy"] = mean(accs) if accs else None
    print("-" * len(header))
    print(row_str("mean", mean_stats))


def print_per_layer_table(name: str, s: dict):
    """Table 2: per-layer t_l profile for a single patient."""
    print(f"\nPer-layer profile: {name}")
    print(f"  n_layers={s['n_layers']}  kv_heads={s['kv_heads']}  "
          f"orig={s['original_seq_len']}  article={s['article_len']}  "
          f"target={s['target']}")
    header = (f"  {'l':>3} {'t_l':>6} {'real_sum':>8} "
              f"{'h_min':>5} {'h_p50':>5} {'h_max':>5}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for li, (t_l, real_total, per_head) in enumerate(zip(
            s["per_layer_t"], s["per_layer_real_total"],
            s["per_layer_per_head_real"])):
        print(f"  {li:>3d} {t_l:>6d} {real_total:>8d} "
              f"{min(per_head):>5d} {int(median(per_head)):>5d} "
              f"{max(per_head):>5d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="long-health",
                    help="Directory containing patient_XX/cache.pt (default: long-health)")
    ap.add_argument("--patients", default="1-20",
                    help="Patient indices, e.g. '1-20' or '1-5,9,12-14' (default: 1-20)")
    ap.add_argument("--json", default=None,
                    help="Optional path to dump full per-patient stats as JSON")
    ap.add_argument("--per-layer", default=None,
                    help="Patient id (e.g. patient_09) to print per-layer profile for")
    args = ap.parse_args()

    indices = parse_patients(args.patients)
    rows = []  # list of (patient_id, stats_dict)
    for idx in indices:
        pid = f"patient_{idx:02d}"
        cache_path = os.path.join(args.results_dir, pid, "cache.pt")
        results_path = os.path.join(args.results_dir, pid, "results.json")
        if not os.path.exists(cache_path):
            print(f"skip {pid}: no cache.pt at {cache_path}", file=sys.stderr)
            continue
        try:
            stats = analyze_patient(cache_path, results_path)
        except Exception as e:
            print(f"error {pid}: {e}", file=sys.stderr)
            continue
        rows.append((pid, stats))

    if not rows:
        print("no patients analyzed", file=sys.stderr)
        sys.exit(1)

    print_summary_table(rows)

    if args.per_layer:
        match = next((s for n, s in rows if n == args.per_layer), None)
        if match is None:
            print(f"\n--per-layer: {args.per_layer} not in analyzed set",
                  file=sys.stderr)
        else:
            print_per_layer_table(args.per_layer, match)

    if args.json:
        # JSON-serializable copy (lists, no tensors).
        dump = {name: stats for name, stats in rows}
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dump, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
