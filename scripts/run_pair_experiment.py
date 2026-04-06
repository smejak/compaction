"""
Pair-stacked KV cache eval on LongHealth with Qwen3-4B.

Loads two pre-computed compacted KV caches from disk, stacks them along the
sequence dimension (variant=naive or rope_shift), and evaluates both patients'
questions against the stacked cache. Logs accuracy + per-layer attention mass
over cache_A / cache_B / question regions, both before the question is
processed (from the compacted beta values) and after (from an instrumented
forward pass with output_attentions=True).

Usage:
    python -u scripts/run_pair_experiment.py --pair-idx 0 --variant naive
    python -u scripts/run_pair_experiment.py --pair-idx 0 --variant rope_shift

Pair indices 0..41 index into a deterministic list of all ordered pairs of the
7 already-compacted patients (excluding self-pairs). Outputs land at:

    <results-dir>/<variant>/pair_<A>_<B>/results.json

Designed to be invoked by a SLURM array job (scripts/marlowe/pair_experiment.sh).
"""
import argparse
import gc
import json
import os
import sys
import time

# Make sibling packages importable when invoked as `python scripts/run_pair_experiment.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B"

# Patients whose compacted caches are already on disk (from the per-patient run).
PATIENT_IDS = [
    "patient_01",
    "patient_03",
    "patient_04",
    "patient_05",
    "patient_06",
    "patient_07",
    "patient_08",
]

# Deterministic ordered pair list (excludes self-pairs). len == 42.
PAIRS = [(a, b) for a in PATIENT_IDS for b in PATIENT_IDS if a != b]


def _load_cache(path):
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


def _stack_caches(cache_a, cache_b, variant, model, device, dtype):
    """Concatenate two compacted caches along the sequence dimension.

    Returns
    -------
    cache_cpu : list of (C1, beta, C2) tuples, one per layer, on CPU.
    stacked_seq_len : int
    t_A_per_layer : list[int]
    t_B_per_layer : list[int]
    """
    import torch
    from compaction.compaction_methods.chunked import (
        compute_rope_correction,
        apply_rotary_pos_emb_to_cache,
    )

    assert len(cache_a["cache"]) == len(cache_b["cache"]), (
        f"layer count mismatch: {len(cache_a['cache'])} vs {len(cache_b['cache'])}"
    )
    num_layers = len(cache_a["cache"])
    seq_len_a = int(cache_a["original_seq_len"])
    seq_len_b = int(cache_b["original_seq_len"])

    # For rope_shift: precompute the cos/sin correction once (same offset for
    # every layer and every token — all of cache_B shifts by +seq_len_a).
    cos_diff = sin_diff = None
    if variant == "rope_shift":
        cos_diff, sin_diff = compute_rope_correction(
            model,
            current_positions=torch.tensor([0], device=device),
            target_positions=torch.tensor([seq_len_a], device=device),
            device=device,
            dtype=dtype,
        )

    t_A_per_layer = []
    t_B_per_layer = []
    cache_cpu = []
    for layer_idx in range(num_layers):
        C1_a, beta_a, C2_a = cache_a["cache"][layer_idx]
        C1_b, beta_b, C2_b = cache_b["cache"][layer_idx]

        t_A_per_layer.append(int(C1_a.shape[-2]))
        t_B_per_layer.append(int(C1_b.shape[-2]))

        if variant == "rope_shift":
            # Move cache_B keys to GPU for RoPE correction, then back to CPU
            C1_b_gpu = C1_b.to(device=device, dtype=dtype)
            C1_b_shifted = apply_rotary_pos_emb_to_cache(C1_b_gpu, cos_diff, sin_diff)
            C1_b = C1_b_shifted.to(device="cpu", dtype=C1_a.dtype)
            del C1_b_gpu, C1_b_shifted

        # Concat along sequence dim (-2). beta and C2 are position-independent.
        C1 = torch.cat([C1_a, C1_b], dim=-2)
        beta = torch.cat([beta_a, beta_b], dim=-1)
        C2 = torch.cat([C2_a, C2_b], dim=-2)
        cache_cpu.append((C1.contiguous(), beta.contiguous(), C2.contiguous()))

    stacked_seq_len = seq_len_a + seq_len_b
    return cache_cpu, stacked_seq_len, t_A_per_layer, t_B_per_layer


def _attn_mass_before(cache_cpu, t_A_per_layer):
    """Per-layer cache_A vs cache_B share of attention mass from compaction.

    Uses the beta values stored in the stacked cache. beta is already attention
    *mass*-like (log-space bias accumulated during compaction), so we exponentiate
    per-layer, sum over heads and batch, and split by the layer-specific split
    point between cache_A and cache_B.
    """
    import torch
    per_layer = []
    for (_, beta, _), t_a in zip(cache_cpu, t_A_per_layer):
        # beta: (1, KV, t) — these are log-space biases added into the attention
        # logits during forward. Exponentiate to get attention-mass weights, sum.
        w = torch.exp(beta.float())  # (1, KV, t)
        mass_a = float(w[..., :t_a].sum())
        mass_b = float(w[..., t_a:].sum())
        total = mass_a + mass_b
        if total > 0:
            per_layer.append([mass_a / total, mass_b / total])
        else:
            per_layer.append([0.5, 0.5])
    mean_a = sum(p[0] for p in per_layer) / len(per_layer)
    mean_b = sum(p[1] for p in per_layer) / len(per_layer)
    return {"per_layer": per_layer, "mean_A": mean_a, "mean_B": mean_b}


def _build_cache_gpu(cache_cpu, device, dtype):
    return tuple(
        (c1.to(device=device, dtype=dtype),
         b.to(device=device, dtype=dtype),
         c2.to(device=device, dtype=dtype))
        for c1, b, c2 in cache_cpu
    )


def _run_instrumented_forward(model, tokenizer, prompts, cache_cpu, stacked_seq_len,
                              t_A_per_layer, t_B_per_layer, device, dtype):
    """Run one forward pass over (stacked_cache + prompts) and return per-layer
    attention mass (averaged over heads and batch) over the three regions
    {cache_A, cache_B, question} for the last question token of each item.

    Returns a list (one entry per prompt in the batch) of lists (per layer) of
    dicts {"layer": int, "A": float, "B": float, "Q": float}.
    """
    import torch
    from models.cache import CompactedPrefixCache

    # Tokenize with left-padding to mirror generate_with_compacted_cache_batch.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=False, add_special_tokens=False).to(device)
    tokenizer.padding_side = original_padding_side

    input_ids = enc["input_ids"]
    input_attn_mask = enc["attention_mask"]
    batch_size, input_len = input_ids.shape
    pad_counts = (input_attn_mask == 0).sum(dim=1)

    # Build cache_gpu and expand to batch size.
    expanded = []
    for (C1, beta, C2) in cache_cpu:
        c1 = C1.to(device=device, dtype=dtype)
        c2 = C2.to(device=device, dtype=dtype)
        bb = beta.to(device=device, dtype=dtype)
        if c1.shape[0] == 1 and batch_size > 1:
            c1 = c1.expand(batch_size, *c1.shape[1:]).contiguous()
            c2 = c2.expand(batch_size, *c2.shape[1:]).contiguous()
            bb = bb.expand(batch_size, *bb.shape[1:]).contiguous()
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
            (batch_size, past_seen_tokens),
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

    attentions = out.attentions  # tuple of (B, heads, q_len, layer_kv_len) per layer
    if attentions is None:
        raise RuntimeError(
            "model returned no attentions — ensure attn_implementation='eager'"
        )

    # The last question token's index within input_ids (no padding stripping
    # needed per-sample since padding is on the LEFT, so the last token is
    # always at input_len - 1).
    q_last = input_len - 1

    # Bucket per (batch, layer). Each layer's attention tensor has
    # k_len = t_A_layer + t_B_layer + input_len (prompt goes after the cache).
    per_item = [[] for _ in range(batch_size)]
    for layer_idx, attn in enumerate(attentions):
        # attn: (B, heads, q_len, k_len)
        t_a = t_A_per_layer[layer_idx]
        t_b = t_B_per_layer[layer_idx]
        # Average over heads first (keep batch and k_len).
        row = attn[:, :, q_last, :].float().mean(dim=1)  # (B, k_len)
        # Bucket the k axis.
        mass_a = row[:, :t_a].sum(dim=-1)                       # (B,)
        mass_b = row[:, t_a:t_a + t_b].sum(dim=-1)              # (B,)
        mass_q = row[:, t_a + t_b:].sum(dim=-1)                 # (B,)
        for b in range(batch_size):
            per_item[b].append({
                "layer": layer_idx,
                "A": float(mass_a[b]),
                "B": float(mass_b[b]),
                "Q": float(mass_q[b]),
            })

    del cache, expanded, out, attentions
    torch.cuda.empty_cache()
    return per_item


def _load_model_eager(model_name, device):
    """Load Qwen3 model with attn_implementation='eager' so output_attentions=True works."""
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


def run_pair(pair_idx: int, variant: str, results_dir: str, caches_dir: str,
             model_name: str):
    if not (0 <= pair_idx < len(PAIRS)):
        raise ValueError(f"--pair-idx must be in [0, {len(PAIRS)}); got {pair_idx}")
    if variant not in ("naive", "rope_shift"):
        raise ValueError(f"--variant must be naive|rope_shift; got {variant}")

    pid_a, pid_b = PAIRS[pair_idx]
    out_dir = os.path.join(results_dir, variant, f"pair_{pid_a}_{pid_b}")
    result_path = os.path.join(out_dir, "results.json")
    if os.path.exists(result_path):
        print(f"skip pair_idx={pair_idx} {variant} {pid_a}->{pid_b} (already done)")
        return {"pair": [pid_a, pid_b], "variant": variant, "skipped": True}

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from evaluation.utils import format_question, parse_model_choice
    from evaluation.datasets import load_dataset
    from models.generate import generate_with_compacted_cache_batch

    print(f"\n{'='*60}")
    print(f"Pair {pair_idx}/{len(PAIRS)}: {pid_a} -> {pid_b}  (variant={variant})")
    print(f"{'='*60}")

    # Load caches.
    cache_a = _load_cache(os.path.join(caches_dir, pid_a, "cache.pt"))
    cache_b = _load_cache(os.path.join(caches_dir, pid_b, "cache.pt"))
    seq_len_a = int(cache_a["original_seq_len"])
    seq_len_b = int(cache_b["original_seq_len"])
    print(f"  cache_A: {pid_a}  seq_len={seq_len_a}  layers={len(cache_a['cache'])}")
    print(f"  cache_B: {pid_b}  seq_len={seq_len_b}  layers={len(cache_b['cache'])}")

    # Load model (eager) BEFORE stacking, because rope_shift needs model.rotary_emb.
    model, tokenizer = _load_model_eager(model_name, "cuda")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Stack.
    t0 = time.time()
    cache_cpu, stacked_seq_len, t_A_per_layer, t_B_per_layer = _stack_caches(
        cache_a, cache_b, variant, model, device, dtype
    )
    stack_time = time.time() - t0
    max_layer_len = max(c1.shape[-2] for c1, _, _ in cache_cpu)
    print(f"  Stacked: seq_len={stacked_seq_len}  max_layer_len={max_layer_len}  "
          f"stack_time={stack_time:.2f}s")

    # attn_mass_before (pair-level, derived from compaction).
    amb = _attn_mass_before(cache_cpu, t_A_per_layer)
    print(f"  attn_mass_before: meanA={amb['mean_A']:.3f}  meanB={amb['mean_B']:.3f}")

    # Free raw cache dicts.
    del cache_a, cache_b
    gc.collect()
    torch.cuda.empty_cache()

    # Load questions for both patients.
    data = load_dataset("longhealth")
    by_pid = {}
    for art in data:
        # article_id is e.g. "longhealth_patient_01"
        pid = art["article_id"].replace("longhealth_", "")
        by_pid[pid] = art
    if pid_a not in by_pid or pid_b not in by_pid:
        raise RuntimeError(f"Patient(s) not found in dataset: {pid_a}, {pid_b}")

    questions_a = [dict(q, patient=pid_a, position=1) for q in by_pid[pid_a]["questions"]]
    questions_b = [dict(q, patient=pid_b, position=2) for q in by_pid[pid_b]["questions"]]
    questions = questions_a + questions_b
    print(f"  questions: {len(questions_a)} (A) + {len(questions_b)} (B) = {len(questions)}")

    # Batch size heuristic (same as run_per_patient.py, but accounting for 2x
    # longer stacked context — 25000 budget divides more aggressively).
    batch_size = max(1, min(20, int(25000 / max_layer_len)))
    print(f"  batch_size={batch_size}")

    results = []
    for bs in range(0, len(questions), batch_size):
        be = min(bs + batch_size, len(questions))
        batch = questions[bs:be]
        prompts = [format_question(tokenizer, q["question"], q.get("options"), model_name)
                   for q in batch]

        # (1) Instrumented forward pass for attn_mass_after. This mutates the
        # cache inside CompactedPrefixCache via layer.update(), so we discard it
        # and rebuild cache_gpu for the actual generation below.
        attn_after_batch = _run_instrumented_forward(
            model, tokenizer, prompts, cache_cpu, stacked_seq_len,
            t_A_per_layer, t_B_per_layer, device, dtype,
        )

        # (2) Real generation with a fresh cache_gpu.
        cache_gpu = _build_cache_gpu(cache_cpu, device, dtype)
        answers = generate_with_compacted_cache_batch(
            model, tokenizer, prompts, cache_gpu,
            max_new_tokens=2048, original_seq_len=stacked_seq_len,
        )
        del cache_gpu
        torch.cuda.empty_cache()

        for i, (q, ans) in enumerate(zip(batch, answers)):
            mc = parse_model_choice(ans, max_options=len(q.get("options", [])))
            gold = q.get("gold_label")
            ok = (mc == gold) if mc and gold else False
            results.append({
                "qid": q["question_unique_id"],
                "patient": q["patient"],
                "position": q["position"],
                "correct": ok,
                "pred": mc,
                "gold": gold,
                "attn_mass_after": attn_after_batch[i],
            })
            print(f"    Q{bs+i+1}: {'ok' if ok else 'x'}  pred={mc} gold={gold}  "
                  f"[{q['patient']} pos{q['position']}]")

    correct = sum(r["correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total else 0.0
    pos1 = [r for r in results if r["position"] == 1]
    pos2 = [r for r in results if r["position"] == 2]
    acc_pos1 = sum(r["correct"] for r in pos1) / len(pos1) if pos1 else 0.0
    acc_pos2 = sum(r["correct"] for r in pos2) / len(pos2) if pos2 else 0.0

    result_json = {
        "variant": variant,
        "pair": [pid_a, pid_b],
        "seq_len_A": seq_len_a,
        "seq_len_B": seq_len_b,
        "stacked_original_seq_len": stacked_seq_len,
        "max_layer_len": max_layer_len,
        "t_A_per_layer": t_A_per_layer,
        "t_B_per_layer": t_B_per_layer,
        "attn_mass_before": amb,
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "acc_pos1": acc_pos1,
        "acc_pos2": acc_pos2,
        "model": model_name,
        "per_question": results,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)

    print(f"\n  pair {pid_a}->{pid_b} ({variant}): "
          f"overall={accuracy:.0%}  pos1={acc_pos1:.0%}  pos2={acc_pos2:.0%}")
    print(f"  saved {result_path}")
    return result_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-idx", type=int, required=True,
                        help=f"0-indexed pair (0..{len(PAIRS)-1})")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["naive", "rope_shift"])
    parser.add_argument("--results-dir", default="long-health/pair_experiment")
    parser.add_argument("--caches-dir", default="long-health")
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    run_pair(args.pair_idx, args.variant, args.results_dir,
             args.caches_dir, args.model_name)


if __name__ == "__main__":
    main()
