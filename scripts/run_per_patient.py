"""
Standalone per-patient KV cache compaction on LongHealth with Qwen3-4B.

This is the non-Modal counterpart to modal_per_patient.py:run_patient().
Runs one patient end-to-end (compact + evaluate) and writes outputs in the
same layout as the Modal volume:

    <results_dir>/<patient_id>/cache.pt
    <results_dir>/<patient_id>/results.json

Usage:
    python -u scripts/run_per_patient.py --patient-idx 1 --results-dir long-health

Intended to be invoked by a SLURM array job (see scripts/marlowe/per_patient.sh).
Matches modal_per_patient.py's AM-OMP-fast configuration exactly so results are
directly comparable across Modal and HPC runs.
"""
import argparse
import gc
import json
import math
import os
import sys
import time

# Make sibling packages (evaluation/, models/, compaction/) importable when
# this file is invoked directly as `python scripts/run_per_patient.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Config — kept in lockstep with modal_per_patient.py:28-44
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B"
BUDGET_PATH = "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"
RATIO = 0.1
N_PATIENTS = 20

AM_OMP_FAST_KWARGS = {
    "algorithm": "omp",
    "k_choice": 4,
    "nnls_interval": 2,
    "nnls_iters": 0,
    "nnls_upper_bound": math.exp(7),
    "drop_key_beta_cutoff": -7,
    "c2_method": "lsq",
    "on_policy": True,
    "precomputed_budget_path": BUDGET_PATH,
}


def run_patient(patient_idx: int, results_dir: str, model_name: str, ratio: float):
    """Compact and evaluate one LongHealth patient.

    Body mirrors modal_per_patient.py:51-187 with Modal-specific lines removed.
    """
    # Skip if already done — do this BEFORE heavy imports so already-done
    # patients exit in milliseconds and the smoke test works without torch.
    # patient_id derivation matches modal_per_patient.py:262.
    patient_id = f"patient_{patient_idx+1:02d}"
    out_dir = os.path.join(results_dir, patient_id)
    cache_path = os.path.join(out_dir, "cache.pt")
    result_path = os.path.join(out_dir, "results.json")
    if os.path.exists(cache_path) and os.path.exists(result_path):
        print(f"skip {patient_id} (already done)")
        return {"patient_id": patient_id, "skipped": True}

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from evaluation.utils import (
        load_model_and_tokenizer, extract_full_kv_cache,
        format_context, format_question, parse_model_choice,
    )
    from evaluation.datasets import load_dataset
    from evaluation.configs.utils import load_query_config
    from compaction.compaction_methods.registry import get_compaction_method
    from models.generate import generate_with_compacted_cache_batch

    # Load data (patients_per_article=1 → 20 single-patient articles)
    data = load_dataset("longhealth")
    article = data[patient_idx]
    # Sanity: the derived patient_id should match the dataset's article_id.
    assert article["article_id"].replace("longhealth_", "") == patient_id, (
        f"patient_id mismatch: idx={patient_idx} derived={patient_id} "
        f"article_id={article['article_id']}")
    questions = article["questions"]

    print(f"\n{'='*60}")
    print(f"Patient {patient_idx+1}/{N_PATIENTS}: {patient_id} ({article['title']})")
    print(f"  {len(questions)} questions")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, "cuda")

    # Prefill
    _ = format_context(tokenizer, article["article"], model_name=model_name)
    seq_len, past_kv, article_indices, formatted_ctx, _ = extract_full_kv_cache(
        model, tokenizer, article["article"], "cuda", model_name=model_name)
    article_len = len(article_indices)
    print(f"  Context: {seq_len} tokens (article: {article_len})")

    # Compact
    method = get_compaction_method("AM-OMP-fast", dict(AM_OMP_FAST_KWARGS))
    query_config = load_query_config("repeat")

    non_article = seq_len - article_len
    article_target = max(1, int(article_len * ratio))
    target = article_target + non_article
    print(f"  Compacting: {seq_len} → {target} (article {article_len} → {article_target})")

    t0 = time.time()
    compacted, stats = method.compact_kv_cache(
        past_key_values=past_kv, target_size=target, indices=article_indices,
        query_config=query_config, model=model, tokenizer=tokenizer,
        formatted_context=formatted_ctx, sliding_layer_indices=set())
    comp_time = time.time() - t0
    print(f"  Compaction done in {comp_time:.1f}s")

    # Free compaction intermediates
    del past_kv, method, query_config
    gc.collect()
    torch.cuda.empty_cache()

    # Save compacted cache
    os.makedirs(out_dir, exist_ok=True)
    cache_cpu = [(c1.cpu(), beta.cpu(), c2.cpu()) for c1, beta, c2 in compacted]
    torch.save({
        "cache": cache_cpu,
        "original_seq_len": seq_len,
        "patient_id": patient_id,
        "patient_name": article["title"],
        "model": model_name,
        "ratio": ratio,
        "article_len": article_len,
        "compaction_time": comp_time,
    }, cache_path)
    print(f"  Saved cache → {cache_path}")

    # Clean up stats (may contain tensor refs)
    if stats:
        stats.pop("_original_chunk_caches", None)
        stats.pop("_compacted_chunk_caches", None)
        stats.pop("per_layer_head_metrics", None)
    del compacted
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate
    max_layer = max(c1.shape[2] for c1, _, _ in cache_cpu)
    batch_size = max(1, min(20, int(25000 / max_layer)))
    print(f"  Evaluating: max_layer={max_layer}, batch_size={batch_size}")

    results = []
    for bs in range(0, len(questions), batch_size):
        be = min(bs + batch_size, len(questions))
        batch = questions[bs:be]
        prompts = [format_question(tokenizer, q["question"], q.get("options"), model_name)
                   for q in batch]
        device = next(model.parameters()).device
        cache_gpu = tuple((c1.to(device), b.to(device), c2.to(device))
                          for c1, b, c2 in cache_cpu)
        answers = generate_with_compacted_cache_batch(
            model, tokenizer, prompts, cache_gpu,
            max_new_tokens=2048, original_seq_len=seq_len)
        del cache_gpu
        torch.cuda.empty_cache()

        for i, (q, ans) in enumerate(zip(batch, answers)):
            mc = parse_model_choice(ans, max_options=len(q.get("options", [])))
            gold = q.get("gold_label")
            ok = (mc == gold) if mc and gold else False
            results.append({"qid": q["question_unique_id"], "correct": ok,
                            "pred": mc, "gold": gold, "answer": ans[:300]})
            print(f"    Q{bs+i+1}: {'ok' if ok else 'x'}  pred={mc} gold={gold}")

    correct = sum(r["correct"] for r in results)
    accuracy = correct / len(results) if results else 0.0

    result_json = {
        "patient_id": patient_id,
        "patient_name": article["title"],
        "model": model_name,
        "ratio": ratio,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "compaction_time": comp_time,
        "max_layer_len": max_layer,
        "per_question": results,
    }

    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    print(f"  Saved results → {result_path}")

    print(f"\n  {patient_id}: {accuracy:.0%} ({correct}/{len(results)})")
    return {"patient_id": patient_id, "accuracy": accuracy,
            "correct": correct, "total": len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-idx", type=int, required=True,
                        help="0-indexed patient (0..19)")
    parser.add_argument("--results-dir", default="long-health",
                        help="Output directory (default: long-health)")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--ratio", type=float, default=RATIO)
    args = parser.parse_args()

    if not (0 <= args.patient_idx < N_PATIENTS):
        print(f"--patient-idx must be in [0, {N_PATIENTS}); got {args.patient_idx}",
              file=sys.stderr)
        sys.exit(2)

    run_patient(args.patient_idx, args.results_dir, args.model_name, args.ratio)


if __name__ == "__main__":
    main()
