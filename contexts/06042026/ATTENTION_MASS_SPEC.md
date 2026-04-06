# Attention-mass telemetry spec — pair-stacked KV cache eval

This document is the contract for what attention-mass statistics
`scripts/run_pair_experiment.py` produces and what
`scripts/aggregate_pair_results.py` consumes. It supersedes the inline
description in `contexts/06042026/PAIR_EXPERIMENT_HANDOFF.md` and the
"attention-mass telemetry" section of the master plan
(`~/.claude/plans/adaptive-stargazing-curry.md`).

## 1. What we're measuring and why

The pair experiment stacks two pre-compacted KV caches `[cache_A, cache_B]`
and asks the model questions about both patients. The recency-bias hypothesis
is: when a question is about patient B (the second-position patient), is the
model's last-query-token attention biased toward `cache_B` keys *more* than it
is biased toward `cache_A` keys when the question is about patient A?

Per-layer attention mass over the three k-axis regions
`{cache_A, cache_B, question}` for the **last query token** is the right
proxy because:

- the last query token is the position from which the first answer token is
  generated, so its attention distribution is the most direct predictor of
  retrieval behavior;
- per-layer breakdown lets us see whether any bias is uniform across the
  network or concentrated in a few layers;
- bucketing by region (rather than per-token) keeps the storage manageable
  and matches the hypothesis we're testing.

## 2. What changed from the original design

The original design (committed in `scripts/run_pair_experiment.py` as written)
captures one `[A, B, Q]` triple **per layer per question**, batched. This had
two problems:

1. **OOM risk.** Eager attention with `output_attentions=True` materializes a
   `(batch, num_query_heads, q_len, k_len)` tensor for **every** layer; with
   36 layers, batch ≥ 4, num_heads = 32, and stacked context up to ~21k, the
   peak GPU footprint runs into tens of GB on top of the model and KV cache.
2. **JSON bloat.** 36 layers × 3 floats × 40 questions ≈ 4.3 K floats per pair.
   At 84 pairs (42 × 2 variants) the per-question detail adds up to several
   MB of essentially unused information — the analyses we plan to run all
   want position-conditional averages, not per-question variance.

The replacement design drops the per-question detail and instead computes
**per-position-per-layer running aggregates** on-the-fly while running the
instrumented forward at **batch size 1**.

## 3. Telemetry produced

For each pair, `results.json` carries two attention-mass quantities.

### 3.1 `attn_mass_before` (unchanged)

Pair-level, per-layer share of compaction-time attention mass that the
compactor allocated to the cache_A vs cache_B token ranges. Derived from
`exp(beta).sum()` over `[0, t_A_layer)` and `[t_A_layer, t_A_layer + t_B_layer)`
respectively, with no forward pass needed.

```json
"attn_mass_before": {
  "per_layer": [[0.54, 0.46], [0.50, 0.50], …],
  "mean_A": 0.51,
  "mean_B": 0.49
}
```

`per_layer` has one entry per layer (36 for Qwen3-4B). Each entry is a
two-element list `[A_share, B_share]` summing to 1.

### 3.2 `attn_mass_after_aggregate` (new)

Pair-level. Two sub-objects, one per question position
(`position_1` = questions about cache_A's patient,
`position_2` = questions about cache_B's patient). Each holds a per-layer
list of `{mean, std}` across the 20 questions in that position, for each of
the three regions:

```json
"attn_mass_after_aggregate": {
  "position_1": {
    "n": 20,
    "per_layer": [
      {
        "layer": 0,
        "A_mean": 0.31, "A_std": 0.05,
        "B_mean": 0.18, "B_std": 0.04,
        "Q_mean": 0.51, "Q_std": 0.06
      },
      …
    ]
  },
  "position_2": {
    "n": 20,
    "per_layer": [ … ]
  }
}
```

Invariant: `A_mean + B_mean + Q_mean ≈ 1.0` per layer (last-query-token
softmax row sums to 1, averaged over heads — mean of 1's is still 1; tiny
numerical drift is allowed).

### 3.3 `per_question` (slimmed)

Each question record keeps **only** the fields needed for accuracy analysis:

```json
{
  "qid": "patient_01_q0",
  "patient": "patient_01",
  "position": 1,
  "correct": true,
  "pred": 4,
  "gold": 4
}
```

The previous `attn_mass_after` field is removed. There is no per-question
attention information in the JSON anymore.

## 4. How the aggregates are computed

In the eval loop, `run_pair_experiment.py` maintains a per-position
accumulator initialised once per pair:

```python
agg = {
    1: {"n": 0,
        "A_sum": np.zeros(num_layers),
        "B_sum": np.zeros(num_layers),
        "Q_sum": np.zeros(num_layers),
        "A_sq_sum": np.zeros(num_layers),
        "B_sq_sum": np.zeros(num_layers),
        "Q_sq_sum": np.zeros(num_layers)},
    2: { … },  # same shape
}
```

For each question, the instrumented forward runs at `batch_size=1`. It
returns a per-layer triple `(A, B, Q)` for the single sample's last query
token. The accumulator updates:

```python
agg[pos]["A_sum"]    += A_arr
agg[pos]["A_sq_sum"] += A_arr ** 2
agg[pos]["B_sum"]    += B_arr
agg[pos]["B_sq_sum"] += B_arr ** 2
agg[pos]["Q_sum"]    += Q_arr
agg[pos]["Q_sq_sum"] += Q_arr ** 2
agg[pos]["n"]        += 1
```

After all 40 questions, the accumulator is finalised:

```python
mean = sum / n
var  = sum_sq / n - mean ** 2
std  = sqrt(max(var, 0))   # clamp for numerical safety
```

The std is descriptive only — the 20 questions per position are not iid
samples (same patient, same compacted cache, related question content), so
this should not be interpreted as a confidence interval.

## 5. Why batch size 1 for the instrumented forward

The OOM risk in the original design comes from `output_attentions=True`
materializing one `(B, H, q_len, k_len)` tensor per layer and accumulating
all 36 of them in `out.attentions` at the end of the forward. At
`batch_size=1` the peak is 36× one sample's attention tensor, which on
Qwen3-4B with q_len ≈ 200 prompt tokens and k_len ≈ 21 K stacked context
amounts to ~36 × 32 × 200 × 21000 × 2 bytes ≈ 9 GB — comfortably inside the
80 GB H100 budget on top of the model (~8 GB) and KV cache (~6–8 GB).

The cost is running 40 instrumented forwards per pair instead of ~10 batched
ones. Each forward at `batch_size=1` is cheap relative to generation, so the
wall-clock impact is small — we still expect each pair to finish well inside
the 8 h SLURM budget.

The real generation pass is **still batched** at the original heuristic
(`max(1, min(20, int(25000 / max_layer_len)))`) — only the instrumented
forward is single-sample. Generation has no `output_attentions`, so it
doesn't carry the same OOM risk.

If OOM still occurs at `batch_size=1`, the next escalation is to register a
forward hook on each `Qwen3Attention.forward` that captures the local
`attn_weights` tensor and replaces it with `None` in the layer's return
tuple, so the model never builds up a 36-layer stack. **Don't do this
preemptively** — only if `batch_size=1` is insufficient.

## 6. How the aggregator consumes the new schema

`scripts/aggregate_pair_results.py`:

- **`_build_matrices`** reads `attn_mass_before.mean_A` (unchanged) and a
  pair-level cache-A share derived from the new aggregate as
  `mean(position_1.A_mean) + mean(position_2.A_mean) / 2` averaged across
  layers — i.e., the average of all four per-layer means and treating both
  positions equally. This is the entry that populates the
  `attn_after_A` 7×7 heatmap.
- **`_plot_attention_mass`** loops over pairs and reads
  `r["attn_mass_after_aggregate"]["position_<n>"]["per_layer"][i]["A_mean"]`
  (and `B_mean`, `Q_mean`) directly — one number per (pair, position, layer,
  region). The cross-pair average for the line plot is the
  np.mean of the per-pair means along the pair axis. Each pair contributes
  one observation per layer per region per position, weighted equally.
- **`_plot_variant_grid`** is unchanged — it operates on accuracy matrices.
- **`_plot_cross_variant_diff`** is unchanged for the same reason.

## 7. Caveats

- **Std is descriptive, not inferential.** The 20 questions per position are
  drawn from the same patient (related content, shared cache, multiple choice
  with overlapping option sets). Treat the per-position std as a rough scale
  parameter, not a confidence interval.
- **`A_mean + B_mean + Q_mean` may be slightly off 1.0** due to per-head
  averaging artifacts (if some heads are heavily biased to one region while
  others are not, the head-mean is well-defined but the per-head softmax sums
  averaged after head-averaging are still 1 by linearity — so this should be
  exactly 1 modulo float drift).
- **The instrumented forward path mutates `CompactedPrefixCache` layers via
  `layer.update()`** — the implementation must rebuild `cache_gpu` from
  `cache_cpu` for the real generation call, the same way the original
  implementation did. (Unchanged.)
- **`attn_mass_before` is independent of question content** and therefore
  identical across all questions in a pair. It's stored once at pair level.
- **Per-question correctness is fully preserved** — only the per-question
  attention detail is dropped. Accuracy analyses are unaffected.
