"""
Per-patient KV cache compaction on LongHealth with Qwen3-4B.

Trains one AM-OMP-fast compaction per patient (20 total), saves each
compacted KV cache to disk, and evaluates per-patient QA accuracy.

Usage:
    modal run modal_per_patient.py
"""
import modal, math

app = modal.App("am-per-patient")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "transformers==4.57.1", "vllm==0.11.0",
        "accelerate==1.12.0", "datasets==4.4.1",
        "scipy", "matplotlib", "seaborn", "pandas", "huggingface_hub",
    )
    .run_commands("apt-get update", "apt-get install -y git")
    .add_local_dir(".", remote_path="/root/compaction")
)

vol = modal.Volume.from_name("am-experiment-results", create_if_missing=True)
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen3-4B"
BUDGET_PATH = "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"
RATIO = 0.1
MAX_PARALLEL = 8
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


@app.function(
    gpu="H100", timeout=7200, image=image,
    volumes={"/results": vol, "/root/.cache/huggingface": model_vol},
)
def run_patient(patient_idx: int):
    """Compact and evaluate one patient."""
    import sys, os, json, time
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch, gc

    sys.path.insert(0, "/root/compaction")
    os.chdir("/root/compaction")

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
    patient_id = article["article_id"].replace("longhealth_", "")  # e.g. "patient_01"
    questions = article["questions"]

    print(f"\n{'='*60}")
    print(f"Patient {patient_idx+1}/{N_PATIENTS}: {patient_id} ({article['title']})")
    print(f"  {len(questions)} questions")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, "cuda")

    # Prefill
    formatted_context = format_context(tokenizer, article["article"], model_name=MODEL_NAME)
    seq_len, past_kv, article_indices, formatted_ctx, _ = extract_full_kv_cache(
        model, tokenizer, article["article"], "cuda", model_name=MODEL_NAME)
    article_len = len(article_indices)
    print(f"  Context: {seq_len} tokens (article: {article_len})")

    # Compact
    method = get_compaction_method("AM-OMP-fast", dict(AM_OMP_FAST_KWARGS))
    query_config = load_query_config("repeat")

    non_article = seq_len - article_len
    article_target = max(1, int(article_len * RATIO))
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
    gc.collect(); torch.cuda.empty_cache()

    # Save compacted cache
    out_dir = f"/results/per_patient/{patient_id}"
    os.makedirs(out_dir, exist_ok=True)

    cache_cpu = [(c1.cpu(), beta.cpu(), c2.cpu()) for c1, beta, c2 in compacted]
    cache_path = f"{out_dir}/cache.pt"
    torch.save({
        "cache": cache_cpu,
        "original_seq_len": seq_len,
        "patient_id": patient_id,
        "patient_name": article["title"],
        "model": MODEL_NAME,
        "ratio": RATIO,
        "article_len": article_len,
        "compaction_time": comp_time,
    }, cache_path)
    print(f"  Saved cache → {cache_path}")

    # Clean up stats (may contain tensor refs)
    if stats:
        stats.pop("_original_chunk_caches", None)
        stats.pop("_compacted_chunk_caches", None)
        stats.pop("per_layer_head_metrics", None)
    del compacted; gc.collect(); torch.cuda.empty_cache()

    # Evaluate
    max_layer = max(c1.shape[2] for c1, _, _ in cache_cpu)
    batch_size = max(1, min(20, int(25000 / max_layer)))
    print(f"  Evaluating: max_layer={max_layer}, batch_size={batch_size}")

    results = []
    for bs in range(0, len(questions), batch_size):
        be = min(bs + batch_size, len(questions))
        batch = questions[bs:be]
        prompts = [format_question(tokenizer, q["question"], q.get("options"), MODEL_NAME)
                   for q in batch]
        device = next(model.parameters()).device
        cache_gpu = tuple((c1.to(device), b.to(device), c2.to(device))
                          for c1, b, c2 in cache_cpu)
        answers = generate_with_compacted_cache_batch(
            model, tokenizer, prompts, cache_gpu,
            max_new_tokens=2048, original_seq_len=seq_len)
        del cache_gpu; torch.cuda.empty_cache()

        for i, (q, ans) in enumerate(zip(batch, answers)):
            mc = parse_model_choice(ans, max_options=len(q.get("options", [])))
            gold = q.get("gold_label")
            ok = (mc == gold) if mc and gold else False
            results.append({"qid": q["question_unique_id"], "correct": ok,
                            "pred": mc, "gold": gold, "answer": ans[:300]})
            print(f"    Q{bs+i+1}: {'✓' if ok else '✗'}  pred={mc} gold={gold}")

    correct = sum(r["correct"] for r in results)
    accuracy = correct / len(results) if results else 0.0

    result_json = {
        "patient_id": patient_id,
        "patient_name": article["title"],
        "model": MODEL_NAME,
        "ratio": RATIO,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "compaction_time": comp_time,
        "max_layer_len": max_layer,
        "per_question": results,
    }

    result_path = f"{out_dir}/results.json"
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    print(f"  Saved results → {result_path}")

    vol.commit()

    print(f"\n  {patient_id}: {accuracy:.0%} ({correct}/{len(results)})")
    return {"patient_id": patient_id, "accuracy": accuracy, "correct": correct, "total": len(results)}


@app.function(image=image, timeout=600, volumes={"/results": vol})
def aggregate():
    """Load all per-patient results and generate summary + plot."""
    import json, os
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    vol.reload()
    base = "/results/per_patient"
    patients = sorted([d for d in os.listdir(base)
                       if os.path.isdir(f"{base}/{d}") and d.startswith("patient_")])

    rows = []
    for pid in patients:
        rpath = f"{base}/{pid}/results.json"
        if os.path.exists(rpath):
            with open(rpath) as f:
                rows.append(json.load(f))

    # Summary JSON
    summary = {
        "model": MODEL_NAME, "ratio": RATIO,
        "n_patients": len(rows),
        "overall_accuracy": sum(r["correct"] for r in rows) / sum(r["total"] for r in rows) if rows else 0,
        "patients": {r["patient_id"]: {"accuracy": r["accuracy"], "correct": r["correct"],
                                        "total": r["total"], "name": r["patient_name"]}
                     for r in rows},
    }
    with open(f"{base}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig_dir = f"{base}/figures"
    os.makedirs(fig_dir, exist_ok=True)

    labels = [r["patient_id"].replace("patient_", "P") for r in rows]
    accs = [r["accuracy"] for r in rows]
    colors = ["steelblue" if a >= 0.5 else "darkorange" for a in accs]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=colors, width=0.7)
    ax.axhline(y=summary["overall_accuracy"], color="black", linestyle="--",
               label=f'Mean: {summary["overall_accuracy"]:.0%}')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Patient AM-OMP-fast Compaction — {MODEL_NAME}, ratio={RATIO}")
    ax.legend()
    for xi, v in enumerate(accs):
        ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/per_patient_accuracy.png", dpi=300)
    fig.savefig(f"{fig_dir}/per_patient_accuracy.pdf")
    plt.close()

    vol.commit()
    print(f"Summary: {len(rows)} patients, overall={summary['overall_accuracy']:.0%}")
    for r in rows:
        print(f"  {r['patient_id']}: {r['accuracy']:.0%} ({r['correct']}/{r['total']})")
    return summary


@app.function(image=image, timeout=14400, volumes={"/results": vol})
def orchestrate():
    """Run all 20 patients in batches of 8, then aggregate."""
    import os
    vol.reload()

    # Find which patients still need to run
    remaining = []
    for i in range(N_PATIENTS):
        pid = f"patient_{i+1:02d}"
        cache_path = f"/results/per_patient/{pid}/cache.pt"
        result_path = f"/results/per_patient/{pid}/results.json"
        if os.path.exists(cache_path) and os.path.exists(result_path):
            print(f"  skip {pid} (already done)")
        else:
            remaining.append(i)

    print(f"\n{len(remaining)} patients to process ({N_PATIENTS - len(remaining)} already done)")

    # Process in batches of MAX_PARALLEL
    for batch_start in range(0, len(remaining), MAX_PARALLEL):
        batch = remaining[batch_start:batch_start + MAX_PARALLEL]
        batch_num = batch_start // MAX_PARALLEL + 1
        total_batches = (len(remaining) + MAX_PARALLEL - 1) // MAX_PARALLEL
        print(f"\n--- Batch {batch_num}/{total_batches}: patients {[f'{b+1:02d}' for b in batch]} ---")

        futures = [(idx, run_patient.spawn(idx)) for idx in batch]
        for idx, f in futures:
            pid = f"patient_{idx+1:02d}"
            try:
                r = f.get(timeout=5400)  # 90 min per-job timeout
                print(f"  ✓ {r['patient_id']}: {r['accuracy']:.0%}")
            except Exception as e:
                print(f"  ✗ {pid} failed: {type(e).__name__}: {str(e)[:200]}")

    # Aggregate whatever completed
    print("\n--- Aggregating ---")
    summary = aggregate.local()
    return summary


@app.local_entrypoint()
def main():
    print(f"Per-patient compaction: {N_PATIENTS} patients, {MODEL_NAME}, ratio={RATIO}")
    print(f"Algorithm: AM-OMP-fast (k_choice=4, nnls_interval=2, on_policy=True)")
    summary = orchestrate.remote()
    print(f"\nDone. Overall: {summary['overall_accuracy']:.0%}")
    print("Results on volume: /results/per_patient/")
