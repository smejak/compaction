"""
Stacked AM compaction: Joint vs Independent on LongHealth (2 patients).

Joint  (J): prefill [P1,P2] together → compact once
Indep  (I): prefill P1 → compact, prefill P2 → compact, concatenate caches

Uses text-based chunked compaction for I, which:
  1. Splits article on <text_0> tags (one chunk per patient)
  2. Prefills each chunk independently
  3. Compacts each chunk
  4. RoPE-shifts and concatenates the compacted caches

Usage:
    modal run modal_stacked_experiment.py
"""
import modal, math

app = modal.App("am-stacked-compaction")

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

MODEL_NAME = "Qwen/Qwen3-8B"
BUDGET_PATH = "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"
RATIO = 0.1

AM_KWARGS = {
    "algorithm": "highest_attention_keys",
    "score_method": "rms",
    "nnls_iters": 2,
    "nnls_lower_bound": math.exp(-3),
    "nnls_upper_bound": math.exp(3),
    "c2_method": "lsq",
    "on_policy": False,
    "precomputed_budget_path": BUDGET_PATH,
}


@app.function(
    gpu="H100", timeout=7200, image=image,
    volumes={"/results": vol, "/root/.cache/huggingface": model_vol},
)
def run_condition(condition: str):
    """Run J (joint) or I (independent) compaction at RATIO."""
    import sys, os, json, time
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch, gc

    sys.path.insert(0, "/root/compaction")
    os.chdir("/root/compaction")

    from evaluation.utils import (
        load_model_and_tokenizer, extract_full_kv_cache,
        format_context, format_question, parse_model_choice,
        compute_article_indices,
    )
    from evaluation.datasets import load_dataset
    from evaluation.configs.utils import load_query_config
    from compaction.compaction_methods.registry import get_compaction_method
    from models.generate import generate_with_compacted_cache_batch

    print(f"\n{'='*60}")
    print(f"Condition {condition} | ratio={RATIO} | model={MODEL_NAME}")
    print(f"{'='*60}")

    # Load model + data
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, "cuda")
    data = load_dataset("longhealth2")
    article = data[0]
    questions = article["questions"]
    article_text = article["article"]

    patient_ids = sorted(set(
        q["question_unique_id"].rsplit("_q", 1)[0] for q in questions
    ))
    print(f"{len(questions)} questions, patients: {patient_ids}")

    formatted_context = format_context(tokenizer, article_text, model_name=MODEL_NAME)
    ctx_len = tokenizer(formatted_context, return_tensors="pt",
                        add_special_tokens=False)["input_ids"].shape[1]
    print(f"Context: {ctx_len} tokens")

    # --- Compaction ---
    query_config = load_query_config("repeat")
    query_config.max_query_vectors_per_kv_head = 10000

    if condition == "J":
        # Joint: single prefill → single compaction
        seq_len, past_kv, art_idx, fmt_ctx, _ = extract_full_kv_cache(
            model, tokenizer, article_text, "cuda", model_name=MODEL_NAME)
        method = get_compaction_method("AM-HighestAttnKeys", dict(AM_KWARGS))
        art_len = len(art_idx)
        target = max(1, int(art_len * RATIO)) + (seq_len - art_len)
        print(f"Joint compaction: {seq_len} → {target}")

        t0 = time.time()
        compacted, stats = method.compact_kv_cache(
            past_key_values=past_kv, target_size=target, indices=art_idx,
            query_config=query_config, model=model, tokenizer=tokenizer,
            formatted_context=fmt_ctx, sliding_layer_indices=set())
        comp_time = time.time() - t0
        del past_kv, method; gc.collect(); torch.cuda.empty_cache()

    else:  # I
        # Independent: text-based chunked compaction (one chunk per patient)
        art_idx = compute_article_indices(tokenizer, formatted_context, article_text)
        seq_len = ctx_len
        art_len = len(art_idx)
        target = max(1, int(art_len * RATIO)) + (seq_len - art_len)

        kwargs = dict(AM_KWARGS)
        kwargs["chunking"] = "longhealth"      # splits on <text_0> → per patient
        kwargs["use_kv_based"] = False          # independent prefill per chunk
        method = get_compaction_method("AM-HighestAttnKeys", kwargs)
        print(f"Independent compaction: {seq_len} → {target}")

        t0 = time.time()
        compacted, stats = method.compact_kv_cache(
            past_key_values=None, target_size=target, indices=art_idx,
            query_config=query_config, model=model, tokenizer=tokenizer,
            formatted_context=formatted_context,
            article_text=article_text, article_name=article["title"],
            sliding_layer_indices=set())
        comp_time = time.time() - t0

        # Verify the chunked output
        n_chunks = stats.get("num_chunks", "?")
        print(f"  Chunks processed: {n_chunks} (expect 2 for 2 patients)")
        del method; gc.collect(); torch.cuda.empty_cache()

    del query_config

    # --- Move cache to CPU, free GPU ---
    if stats:
        stats.pop("_original_chunk_caches", None)
        stats.pop("_compacted_chunk_caches", None)
        stats.pop("per_layer_head_metrics", None)
    cache_cpu = [(c1.cpu(), b.cpu(), c2.cpu()) for c1, b, c2 in compacted]
    del compacted; gc.collect(); torch.cuda.empty_cache()
    print(f"GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Batch size: scale inversely with max layer length (SDPA naive mode)
    max_layer = max(c1.shape[2] for c1, _, _ in cache_cpu)
    batch_size = max(1, min(20, int(25000 / max_layer)))
    print(f"Generation: max_layer={max_layer}, batch_size={batch_size}")

    # --- Evaluate questions ---
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
            pid = q["question_unique_id"].rsplit("_q", 1)[0]
            results.append({"qid": q["question_unique_id"], "patient": pid,
                            "correct": ok, "pred": mc, "gold": gold})
            print(f"  Q{bs+i+1}: {'✓' if ok else '✗'}  pred={mc} gold={gold}  [{pid}]")

    # --- Per-patient accuracy ---
    per_patient = {}
    for pid in patient_ids:
        pq = [r for r in results if r["patient"] == pid]
        c = sum(r["correct"] for r in pq)
        label = f"P{int(pid.split('_')[1])}"
        per_patient[label] = {"n": len(pq), "correct": c,
                              "accuracy": c / len(pq) if pq else 0.0}

    overall = sum(r["correct"] for r in results) / len(results)

    out = {
        "condition": condition, "ratio": RATIO, "model": MODEL_NAME,
        "overall_accuracy": overall, "per_patient": per_patient,
        "compaction_time": comp_time, "per_question": results,
        "num_chunks": stats.get("num_chunks"),
    }

    path = f"/results/pilot/{condition}/results.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    vol.commit()

    print(f"\n{'='*60}")
    print(f"{condition} | overall={overall:.0%}")
    for p, v in per_patient.items():
        print(f"  {p}: {v['accuracy']:.0%} ({v['correct']}/{v['n']})")
    print(f"Saved → {path}")
    return out


@app.function(image=image, timeout=600, volumes={"/results": vol})
def make_plots():
    """Generate comparison plot."""
    import json, os
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    vol.reload()
    with open("/results/pilot/J/results.json") as f: j = json.load(f)
    with open("/results/pilot/I/results.json") as f: i = json.load(f)

    patients = sorted(j["per_patient"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: overall
    ax = axes[0]
    ax.bar(["Joint", "Independent"], [j["overall_accuracy"], i["overall_accuracy"]],
           color=["steelblue", "darkorange"], width=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Overall (ratio={j['ratio']})")
    for idx, v in enumerate([j["overall_accuracy"], i["overall_accuracy"]]):
        ax.text(idx, v + 0.02, f"{v:.0%}", ha="center", fontsize=12)

    # Right: per-patient
    ax = axes[1]
    x = np.arange(len(patients))
    w = 0.3
    j_acc = [j["per_patient"][p]["accuracy"] for p in patients]
    i_acc = [i["per_patient"][p]["accuracy"] for p in patients]
    ax.bar(x - w/2, j_acc, w, label="Joint", color="steelblue")
    ax.bar(x + w/2, i_acc, w, label="Independent", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(patients)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Per-Patient"); ax.legend()
    for xi, (jv, iv) in enumerate(zip(j_acc, i_acc)):
        ax.text(xi - w/2, jv + 0.02, f"{jv:.0%}", ha="center", fontsize=9)
        ax.text(xi + w/2, iv + 0.02, f"{iv:.0%}", ha="center", fontsize=9)

    fig.suptitle(f"Joint vs Independent AM Compaction — {j['model']}, ratio={j['ratio']}", fontsize=13)
    plt.tight_layout()
    os.makedirs("/results/pilot/figures", exist_ok=True)
    fig.savefig("/results/pilot/figures/joint_vs_indep.png", dpi=300)
    fig.savefig("/results/pilot/figures/joint_vs_indep.pdf")
    plt.close()

    vol.commit()
    print("Plots saved to /results/pilot/figures/")
    return {"J": j["overall_accuracy"], "I": i["overall_accuracy"],
            "J_per_patient": {p: j["per_patient"][p]["accuracy"] for p in patients},
            "I_per_patient": {p: i["per_patient"][p]["accuracy"] for p in patients}}


@app.function(image=image, timeout=14400, volumes={"/results": vol})
def orchestrate():
    """Run J and I in parallel, then plot."""
    import os
    vol.reload()

    futures = []
    for c in ["J", "I"]:
        p = f"/results/pilot/{c}/results.json"
        if os.path.exists(p):
            print(f"  {c} already done, skipping")
        else:
            futures.append((c, run_condition.spawn(c)))

    for label, f in futures:
        r = f.get()
        print(f"  {label}: {r['overall_accuracy']:.0%}")

    summary = make_plots.local()

    print(f"\n{'='*60}")
    print(f"RESULTS (ratio={RATIO})")
    print(f"  Joint:       {summary['J']:.0%}")
    print(f"  Independent: {summary['I']:.0%}")
    for p in sorted(summary["J_per_patient"]):
        print(f"  {p}:  J={summary['J_per_patient'][p]:.0%}  I={summary['I_per_patient'][p]:.0%}")
    print(f"{'='*60}")
    return summary


@app.local_entrypoint()
def main():
    print(f"Launching J vs I at ratio={RATIO} on 2 H100s...")
    summary = orchestrate.remote()
    print("Done. Figures on volume /results/pilot/figures/")
