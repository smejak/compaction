"""
k-ary stacked KV cache eval on LongHealth with Qwen3-4B.

Loads k pre-computed compacted KV caches from disk, stacks them along the
sequence dimension (variant=naive or rope_shift), and evaluates each
patient's questions against the stacked cache. Logs accuracy + per-layer
attention mass over each cache region + question region, both before the
question is processed (from the compacted beta values) and after
(aggregated from instrumented forward passes with output_attentions=True).

This is the k-ary generalization of run_pair_experiment.py — see
contexts/07042026/K5_PROBE_PLAN.md for the design rationale and
contexts/06042026/KARY_STACKING_DEEP_DIVE.md for the analysis that
motivated the probe.

Usage:
    python -u scripts/run_kary_experiment.py \\
        --patients patient_01,patient_03,patient_04,patient_05,patient_06 \\
        --variant rope_shift

Outputs land at:

    <results-dir>/<variant>/k<k>_<subset_id>/results.json

where subset_id is the patient indices joined by underscores, e.g.
"01_03_04_05_06" for the canonical k=5 probe.

Designed to be invoked by scripts/marlowe/kary_single.sh.
"""
import argparse
import gc
import json
import os
import sys
import time

# Make sibling packages importable when invoked as `python scripts/run_kary_experiment.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B"


def _load_cache(path):
    """Load a compacted cache from disk and normalize tensor shapes.

    Cribbed verbatim from scripts/run_pair_experiment.py:_load_cache.
    """
    import torch
    d = torch.load(path, weights_only=False, map_location="cpu")
    # Normalize shapes: (B=1, KV, t, D) for C1/C2, (B=1, KV, t) for beta.
    norm = []
    for c1, beta, c2 in d["cache"]:
        if c1.dim() == 3:
            c1 = c1.unsqueeze(0)
        if c2.dim() == 3:
            c2 = c2.unsqueeze(0)
        if beta.dim() == 2:
            beta = beta.unsqueeze(0)
        norm.append((c1, beta, c2))
    d["cache"] = norm
    return d


def _stack_caches_kary(cache_list, variant, model, device, dtype):
    """Concatenate k compacted caches along the sequence dimension.

    Generalizes scripts/run_pair_experiment.py:_stack_caches from k=2 to
    arbitrary k. For variant=rope_shift, each cache i > 0 has its keys
    rotated by the cumulative offset of all preceding caches' original
    seq_lens (i.e., cache i shifts by sum(seq_lens[0..i-1])).

    Returns
    -------
    cache_cpu : list of (C1, beta, C2) tuples, one per layer, on CPU.
    stacked_seq_len : int, sum of all caches' original_seq_len.
    t_per_cache_per_layer : list[list[int]], shape (k, num_layers).
    """
    import torch
    from compaction.compaction_methods.chunked import (
        compute_rope_correction,
        apply_rotary_pos_emb_to_cache,
    )

    k = len(cache_list)
    num_layers = len(cache_list[0]["cache"])
    for i, c in enumerate(cache_list):
        assert len(c["cache"]) == num_layers, (
            f"layer count mismatch at cache {i}: {len(c['cache'])} vs {num_layers}"
        )
    seq_lens = [int(c["original_seq_len"]) for c in cache_list]

    # Cumulative offsets: cache i starts at sum of seq_lens[0..i-1].
    offsets = [0]
    for i in range(k - 1):
        offsets.append(offsets[-1] + seq_lens[i])

    # For rope_shift: precompute one (cos_diff, sin_diff) per cache i > 0.
    # cache 0 stays at position 0 — its slot is None.
    rope_corrections = [None]
    if variant == "rope_shift":
        for i in range(1, k):
            cos_diff, sin_diff = compute_rope_correction(
                model,
                current_positions=torch.tensor([0], device=device),
                target_positions=torch.tensor([offsets[i]], device=device),
                device=device,
                dtype=dtype,
            )
            rope_corrections.append((cos_diff, sin_diff))

    # Per-cache, per-layer compacted lengths.
    t_per_cache_per_layer = [[] for _ in range(k)]
    cache_cpu = []
    for layer_idx in range(num_layers):
        per_cache_C1 = []
        per_cache_beta = []
        per_cache_C2 = []
        for i in range(k):
            C1, beta, C2 = cache_list[i]["cache"][layer_idx]
            t_per_cache_per_layer[i].append(int(C1.shape[-2]))
            if variant == "rope_shift" and i > 0:
                cos_diff, sin_diff = rope_corrections[i]
                C1_gpu = C1.to(device=device, dtype=dtype)
                C1_shifted = apply_rotary_pos_emb_to_cache(C1_gpu, cos_diff, sin_diff)
                C1 = C1_shifted.to(device="cpu", dtype=cache_list[0]["cache"][layer_idx][0].dtype)
                del C1_gpu, C1_shifted
            per_cache_C1.append(C1)
            per_cache_beta.append(beta)
            per_cache_C2.append(C2)
        # Concat along seq dim (-2). beta and C2 are position-independent.
        C1_cat = torch.cat(per_cache_C1, dim=-2)
        beta_cat = torch.cat(per_cache_beta, dim=-1)
        C2_cat = torch.cat(per_cache_C2, dim=-2)
        cache_cpu.append((C1_cat.contiguous(), beta_cat.contiguous(), C2_cat.contiguous()))

    stacked_seq_len = sum(seq_lens)
    return cache_cpu, stacked_seq_len, t_per_cache_per_layer


def _attn_mass_before_kary(cache_cpu, t_per_cache_per_layer):
    """Per-layer per-cache share of attention mass derived from compaction.

    Generalizes scripts/run_pair_experiment.py:_attn_mass_before from a 2-way
    [A, B] split to a k-way [c_0, c_1, ..., c_{k-1}] split. Uses the beta
    values stored in the stacked cache: beta is log-space attention bias
    accumulated during compaction, so we exponentiate, sum over heads/batch,
    and split by the per-layer cumulative cache boundaries.
    """
    import torch
    k = len(t_per_cache_per_layer)
    per_layer = []
    for layer_idx, (_, beta, _) in enumerate(cache_cpu):
        w = torch.exp(beta.float())  # (1, KV, t_total)
        boundaries = [0]
        for i in range(k):
            boundaries.append(boundaries[-1] + t_per_cache_per_layer[i][layer_idx])
        layer_masses = [
            float(w[..., boundaries[i]:boundaries[i + 1]].sum())
            for i in range(k)
        ]
        total = sum(layer_masses)
        if total > 0:
            per_layer.append([m / total for m in layer_masses])
        else:
            per_layer.append([1.0 / k] * k)
    means = [sum(p[i] for p in per_layer) / len(per_layer) for i in range(k)]
    return {"per_layer": per_layer, "means": means}


def _init_agg_kary(num_layers, k):
    """Running sums for on-the-fly aggregation of attn_mass_after, bucketed
    by source-patient stack position (1..k). One bucket per position.

    Each bucket holds:
        n               : int sample count
        cache_sums[i]   : np.ndarray(num_layers,) running sum for cache i
        cache_sq_sums[i]: same for squares
        Q_sum / Q_sq_sum: same for the question region
    """
    import numpy as np
    return {
        pos: {
            "n": 0,
            "cache_sums": [np.zeros(num_layers, dtype=np.float64) for _ in range(k)],
            "cache_sq_sums": [np.zeros(num_layers, dtype=np.float64) for _ in range(k)],
            "Q_sum": np.zeros(num_layers, dtype=np.float64),
            "Q_sq_sum": np.zeros(num_layers, dtype=np.float64),
        }
        for pos in range(1, k + 1)
    }


def _accumulate_agg_kary(agg, position, layers_kbq):
    """Update the aggregator with one sample's per-layer (cache_masses, Q)
    triples, routed to the bucket for the source patient's position."""
    import numpy as np
    k = len(layers_kbq[0]["cache_masses"])
    bucket = agg[position]
    Q_arr = np.asarray([l["Q"] for l in layers_kbq], dtype=np.float64)
    bucket["Q_sum"] += Q_arr
    bucket["Q_sq_sum"] += Q_arr ** 2
    for i in range(k):
        cache_arr = np.asarray(
            [l["cache_masses"][i] for l in layers_kbq], dtype=np.float64
        )
        bucket["cache_sums"][i] += cache_arr
        bucket["cache_sq_sums"][i] += cache_arr ** 2
    bucket["n"] += 1


def _finalize_agg_kary(agg, k):
    """Convert running sums into the per-layer per-position mean+std schema.

    Returns dict {f"position_<p>": {n, per_layer: [...]}}.
    """
    import numpy as np
    out = {}
    for pos in range(1, k + 1):
        bucket = agg[pos]
        n = bucket["n"]
        if n == 0:
            out[f"position_{pos}"] = {"n": 0, "per_layer": []}
            continue
        Q_mean = bucket["Q_sum"] / n
        Q_var = np.maximum(bucket["Q_sq_sum"] / n - Q_mean ** 2, 0.0)
        Q_std = np.sqrt(Q_var)
        cache_means = [bucket["cache_sums"][i] / n for i in range(k)]
        cache_stds = [
            np.sqrt(np.maximum(
                bucket["cache_sq_sums"][i] / n - cache_means[i] ** 2, 0.0
            ))
            for i in range(k)
        ]
        num_layers = len(Q_mean)
        per_layer = []
        for li in range(num_layers):
            per_layer.append({
                "layer": int(li),
                "cache_means": [float(cache_means[i][li]) for i in range(k)],
                "cache_stds": [float(cache_stds[i][li]) for i in range(k)],
                "Q_mean": float(Q_mean[li]),
                "Q_std": float(Q_std[li]),
            })
        out[f"position_{pos}"] = {"n": int(n), "per_layer": per_layer}
    return out


def _build_cache_gpu(cache_cpu, device, dtype):
    """Move per-layer (C1, beta, C2) tuples to GPU.

    Cribbed verbatim from scripts/run_pair_experiment.py:_build_cache_gpu.
    """
    return tuple(
        (c1.to(device=device, dtype=dtype),
         b.to(device=device, dtype=dtype),
         c2.to(device=device, dtype=dtype))
        for c1, b, c2 in cache_cpu
    )


def _run_instrumented_forward_single_kary(model, tokenizer, prompt, cache_cpu,
                                          stacked_seq_len, t_per_cache_per_layer,
                                          device, dtype):
    """Run one forward pass over (stacked_cache + a single prompt) and return
    per-layer attention mass over {cache_0, cache_1, ..., cache_{k-1}, question}
    for the last query token of the single sample.

    batch_size is hardwired to 1 to keep peak GPU memory bounded — eager
    attention with output_attentions=True materializes a per-layer
    (B, H, q_len, k_len) tensor for every layer in out.attentions, so the
    peak scales linearly with B. See contexts/06042026/ATTENTION_MASS_SPEC.md
    §5.

    Cribbed almost verbatim from
    scripts/run_pair_experiment.py:_run_instrumented_forward_single — only
    the per-layer bucketing loop at the end is generalized from 2 cache
    regions to k.

    Returns a list of length num_layers, each entry a dict
    {"layer": int, "cache_masses": [k floats], "Q": float}.
    """
    import torch
    from models.cache import CompactedPrefixCache

    k = len(t_per_cache_per_layer)

    # Tokenize a single prompt with left-padding (matches the convention
    # used by generate_with_compacted_cache_batch so the last query token
    # sits at input_len - 1 even though there's no padding to apply at B=1).
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    enc = tokenizer([prompt], return_tensors="pt", padding=True,
                    truncation=False, add_special_tokens=False).to(device)
    tokenizer.padding_side = original_padding_side

    input_ids = enc["input_ids"]                # (1, input_len)
    input_attn_mask = enc["attention_mask"]
    input_len = int(input_ids.shape[1])
    pad_counts = (input_attn_mask == 0).sum(dim=1)

    # Build cache_gpu (B=1, no expansion).
    expanded = []
    for (C1, beta, C2) in cache_cpu:
        c1 = C1.to(device=device, dtype=dtype)
        c2 = C2.to(device=device, dtype=dtype)
        bb = beta.to(device=device, dtype=dtype)
        expanded.append((c1, bb, c2))

    cache = CompactedPrefixCache(
        tuple(expanded),
        original_seq_len=stacked_seq_len,
        pad_counts=pad_counts,
        sliding_layer_indices=None,
        sliding_window=None,
    )
    past_seen_tokens = cache.get_seq_length()

    if past_seen_tokens > 0:
        prefix_mask = torch.ones(
            (1, past_seen_tokens),
            device=device, dtype=input_attn_mask.dtype,
        )
        attention_mask = torch.cat([prefix_mask, input_attn_mask], dim=1)
    else:
        attention_mask = input_attn_mask

    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + input_len,
        device=device, dtype=torch.long,
    )

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

    attentions = out.attentions  # tuple of (1, heads, q_len, layer_kv_len) per layer
    if attentions is None:
        raise RuntimeError(
            "model returned no attentions — ensure attn_implementation='eager'"
        )

    # Last query token (left-padded → always at index input_len - 1).
    q_last = input_len - 1

    layers_kbq = []
    for layer_idx, attn in enumerate(attentions):
        # attn: (1, heads, q_len, k_len)
        boundaries = [0]
        for i in range(k):
            boundaries.append(boundaries[-1] + t_per_cache_per_layer[i][layer_idx])
        row = attn[0, :, q_last, :].float().mean(dim=0)  # (k_len,)
        cache_masses = [
            float(row[boundaries[i]:boundaries[i + 1]].sum()) for i in range(k)
        ]
        mass_q = float(row[boundaries[k]:].sum())
        layers_kbq.append({
            "layer": layer_idx,
            "cache_masses": cache_masses,
            "Q": mass_q,
        })

    del cache, expanded, out, attentions
    torch.cuda.empty_cache()
    return layers_kbq


def _load_model_eager(model_name, device):
    """Load Qwen3 model with attn_implementation='eager' so output_attentions=True works.

    Cribbed verbatim from scripts/run_pair_experiment.py:_load_model_eager.
    """
    import torch
    from transformers import AutoTokenizer
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    print(f"Loading model (eager attention): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        device_map=device if device == "cuda" else None,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def run_kary(patients, variant, results_dir, caches_dir, model_name):
    if len(patients) < 2:
        raise ValueError(
            f"--patients must list at least 2 patient IDs; got {len(patients)}"
        )
    if variant not in ("naive", "rope_shift"):
        raise ValueError(f"--variant must be naive|rope_shift; got {variant}")

    # Validate cache files exist before loading the model.
    for pid in patients:
        cache_path = os.path.join(caches_dir, pid, "cache.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"missing cache file: {cache_path}")

    k = len(patients)
    subset_id = "_".join(p.replace("patient_", "") for p in patients)
    out_dir = os.path.join(results_dir, variant, f"k{k}_{subset_id}")
    result_path = os.path.join(out_dir, "results.json")
    if os.path.exists(result_path):
        print(f"skip k={k} {variant} {subset_id} (already done)")
        return {"patients": patients, "variant": variant, "skipped": True}

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from evaluation.utils import format_question, parse_model_choice
    from evaluation.datasets import load_dataset
    from models.generate import generate_with_compacted_cache_batch

    print(f"\n{'='*60}")
    print(f"k-ary experiment: k={k}  variant={variant}")
    print(f"  patients={patients}")
    print(f"{'='*60}")

    # Load caches in stack order.
    caches = []
    for i, pid in enumerate(patients):
        cache_path = os.path.join(caches_dir, pid, "cache.pt")
        c = _load_cache(cache_path)
        seq_len = int(c["original_seq_len"])
        print(f"  cache[{i}]: {pid}  seq_len={seq_len}  layers={len(c['cache'])}")
        caches.append(c)
    seq_lens = [int(c["original_seq_len"]) for c in caches]

    # Load model BEFORE stacking — rope_shift needs model.rotary_emb.
    model, tokenizer = _load_model_eager(model_name, "cuda")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Stack.
    t0 = time.time()
    cache_cpu, stacked_seq_len, t_per_cache_per_layer = _stack_caches_kary(
        caches, variant, model, device, dtype
    )
    stack_time = time.time() - t0
    max_layer_len = max(c1.shape[-2] for c1, _, _ in cache_cpu)
    print(f"  Stacked: seq_len={stacked_seq_len}  max_layer_len={max_layer_len}  "
          f"stack_time={stack_time:.2f}s")

    # attn_mass_before (k-region, derived from beta).
    amb = _attn_mass_before_kary(cache_cpu, t_per_cache_per_layer)
    print("  attn_mass_before means: " +
          ", ".join(f"c{i}={m:.3f}" for i, m in enumerate(amb["means"])))

    # Free raw cache dicts.
    del caches
    gc.collect()
    torch.cuda.empty_cache()

    # Load questions for all k patients, annotated with stack position (1-indexed).
    data = load_dataset("longhealth")
    by_pid = {}
    for art in data:
        pid = art["article_id"].replace("longhealth_", "")
        by_pid[pid] = art
    missing = [pid for pid in patients if pid not in by_pid]
    if missing:
        raise RuntimeError(f"Patient(s) not found in dataset: {missing}")

    questions = []
    for pos, pid in enumerate(patients, start=1):
        for q in by_pid[pid]["questions"]:
            questions.append(dict(q, patient=pid, position=pos))
    print(f"  questions: {len(questions)} total "
          f"({len(by_pid[patients[0]]['questions'])}/patient × {k} patients)")

    # Batch size heuristic — same as the pair experiment, clamps to 1 at large k.
    batch_size = max(1, min(20, int(25000 / max_layer_len)))
    print(f"  batch_size={batch_size}")

    num_layers = len(cache_cpu)
    agg = _init_agg_kary(num_layers, k=k)
    results = []
    for bs in range(0, len(questions), batch_size):
        be = min(bs + batch_size, len(questions))
        batch = questions[bs:be]
        prompts = [
            format_question(tokenizer, q["question"], q.get("options"), model_name)
            for q in batch
        ]

        # (1) Instrumented forwards: ONE prompt at a time at batch_size=1.
        # The instrumented pass mutates CompactedPrefixCache layers via
        # layer.update(), so we discard it and rebuild cache_gpu for the
        # real batched generation below.
        batch_layers_kbq = []
        for q, prompt in zip(batch, prompts):
            layers_kbq = _run_instrumented_forward_single_kary(
                model, tokenizer, prompt, cache_cpu, stacked_seq_len,
                t_per_cache_per_layer, device, dtype,
            )
            batch_layers_kbq.append(layers_kbq)

        # (2) Real batched generation with a fresh cache_gpu.
        cache_gpu = _build_cache_gpu(cache_cpu, device, dtype)
        answers = generate_with_compacted_cache_batch(
            model, tokenizer, prompts, cache_gpu,
            max_new_tokens=2048, original_seq_len=stacked_seq_len,
        )
        del cache_gpu
        torch.cuda.empty_cache()

        # (3) Parse, record correctness, accumulate the captured attention
        # masses into the per-position bucket. per_question records do NOT
        # store raw triples — that's the "aggregate-only" telemetry mode.
        for i, (q, ans) in enumerate(zip(batch, answers)):
            mc = parse_model_choice(ans, max_options=len(q.get("options", [])))
            gold = q.get("gold_label")
            ok = (mc == gold) if mc and gold else False
            _accumulate_agg_kary(agg, q["position"], batch_layers_kbq[i])
            results.append({
                "qid": q["question_unique_id"],
                "patient": q["patient"],
                "position": q["position"],
                "correct": ok,
                "pred": mc,
                "gold": gold,
            })
            print(f"    Q{bs+i+1}: {'ok' if ok else 'x'}  pred={mc} gold={gold}  "
                  f"[{q['patient']} pos{q['position']}]")

    correct = sum(r["correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total else 0.0
    acc_per_position = []
    for pos in range(1, k + 1):
        rs = [r for r in results if r["position"] == pos]
        acc_per_position.append(sum(r["correct"] for r in rs) / len(rs) if rs else 0.0)

    result_json = {
        "variant": variant,
        "k": k,
        "patients": patients,
        "seq_lens": seq_lens,
        "stacked_original_seq_len": stacked_seq_len,
        "max_layer_len": max_layer_len,
        "t_per_cache_per_layer": t_per_cache_per_layer,
        "attn_mass_before": amb,
        "attn_mass_after_aggregate": _finalize_agg_kary(agg, k),
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "acc_per_position": acc_per_position,
        "model": model_name,
        "per_question": results,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)

    pos_summary = "  ".join(
        f"pos{i+1}={a:.0%}" for i, a in enumerate(acc_per_position)
    )
    print(f"\n  k{k} {','.join(patients)} ({variant}):")
    print(f"  overall={accuracy:.0%}  {pos_summary}")
    print(f"  saved {result_path}")
    return result_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patients", type=str, required=True,
        help="Comma-separated list of patient IDs (e.g. patient_01,patient_03,patient_04)",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=["naive", "rope_shift"],
    )
    parser.add_argument("--results-dir", default="long-health/kary_experiment")
    parser.add_argument("--caches-dir", default="long-health")
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    run_kary(patients, args.variant, args.results_dir,
             args.caches_dir, args.model_name)


if __name__ == "__main__":
    main()
