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
  and matches the hypothesis we're testing;
- splitting by question-correctness lets us see whether retrieval *patterns*
  differ between cases the model gets right and cases it gets wrong (a much
  more useful contrast than a population mean alone).

## 2. What changed from the original design

Two iterations of refinement:

**Round 1** (committed in `5cccec6`) replaced the original per-question
batched instrumented forward with a `batch_size=1` instrumented forward + an
on-the-fly per-position aggregator. This addressed the OOM risk from eager
attention's `(B, H, q_len, k_len)` tensors materializing for every layer at
once with `output_attentions=True`.

**Round 2** (this version) keeps the `batch_size=1` instrumented path but:

- Captures per-question per-layer A/B/Q triples in memory **and** writes them
  back into the per_question entry in `results.json`. The OOM concern is
  about GPU memory during the forward pass, not about JSON size — at
  ~43 KB per pair × 760 pairs the storage cost is negligible (~33 MB total),
  and per-question raw data preserves flexibility for any re-analysis we
  haven't thought of yet.
- **Splits the aggregate by `(position, correctness)`**, not just position.
  This is the contrast the user actually cares about: how does the
  attention pattern of correct vs incorrect answers compare?
- Drops the accuracy heatmaps and the `attn_mass_before` heatmap as
  rendered figures. The data is still in `results.json` and `summary.json`;
  see `contexts/06042026/PAIR_EXPERIMENT_REPORT.md` for the index.

## 3. Telemetry produced

For each pair, `results.json` carries:

### 3.1 `attn_mass_before` (unchanged)

Pair-level, per-layer share of compaction-time attention mass that the
compactor allocated to the cache_A vs cache_B token ranges. Derived from
`exp(beta).sum()` over `[0, t_A_layer)` and `[t_A_layer, t_A_layer + t_B_layer)`,
no forward pass needed.

```json
"attn_mass_before": {
  "per_layer": [[0.54, 0.46], [0.50, 0.50], …],
  "mean_A": 0.51,
  "mean_B": 0.49
}
```

`per_layer` has one entry per layer (36 for Qwen3-4B). Each entry is a
two-element list `[A_share, B_share]` summing to 1.

### 3.2 `attn_mass_after_aggregate` (correctness-split)

Pair-level. Two sub-objects, one per question position
(`position_1` = questions about cache_A's patient,
`position_2` = questions about cache_B's patient). Each position holds two
sub-buckets, `correct` and `incorrect`. Each bucket holds a per-layer list of
mean+std across the questions in that bucket, for each of the three regions:

```json
"attn_mass_after_aggregate": {
  "position_1": {
    "correct": {
      "n": 14,
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
    "incorrect": {
      "n": 6,
      "per_layer": [ … ]
    }
  },
  "position_2": {
    "correct":   { "n": …, "per_layer": [ … ] },
    "incorrect": { "n": …, "per_layer": [ … ] }
  }
}
```

Note `n_correct + n_incorrect = 20` per position (the pair's 20 questions in
each position), barring parser failures (`pred=None`) which are counted as
incorrect.

Invariant per layer per bucket: `A_mean + B_mean + Q_mean ≈ 1.0` (last-query
softmax row sums to 1, averaged over heads — mean of 1's is still 1; tiny
numerical drift is allowed).

### 3.3 `per_question` (raw per-question data, with attention)

Each entry has the metadata fields used for accuracy analysis **plus** the
per-question per-layer A/B/Q triples that fed the aggregate above:

```json
{
  "qid": "patient_01_q0",
  "patient": "patient_01",
  "position": 1,
  "correct": true,
  "pred": 4,
  "gold": 4,
  "attn_per_layer": {
    "A": [0.31, 0.32, 0.30, …],   // 36 floats
    "B": [0.18, 0.19, 0.17, …],
    "Q": [0.51, 0.49, 0.53, …]
  }
}
```

This is the raw data — any future re-analysis (e.g. per-layer
correlation with question difficulty, head-of-distribution effects, alternate
correctness criteria) can be derived from this without re-running the eval.

## 4. How the aggregate is computed

In the eval loop, `run_pair_experiment.py` maintains a four-bucket
accumulator initialised once per pair, indexed by `(position, is_correct)`:

```python
agg = {
    (1, True ): {n, A_sum, B_sum, Q_sum, A_sq_sum, B_sq_sum, Q_sq_sum},
    (1, False): { … },
    (2, True ): { … },
    (2, False): { … },
}
```

Per question:

1. Run `_run_instrumented_forward_single(prompt, …)` → per-layer
   `[{"layer", "A", "B", "Q"}]`. Captured into a per-batch list.
2. Run batched generation, parse the answers, compute correctness `ok`.
3. For each question in the batch, accumulate its captured `layers_abq` into
   the bucket `(q["position"], ok)` and append the per-question entry
   (including `attn_per_layer`) to `per_question`.

After all questions are processed, `_finalize_agg(agg)` produces the nested
`attn_mass_after_aggregate` schema in §3.2 by calling `_finalize_bucket` on
each `(pos, correct)` bucket:

```python
mean = sum / n
var  = sum_sq / n - mean ** 2
std  = sqrt(max(var, 0))   # clamp for numerical safety
```

## 5. Why batch size 1 for the instrumented forward

Eager attention with `output_attentions=True` materializes a
`(B, H, q_len, k_len)` tensor per layer and stores all 36 of them in
`out.attentions` after the forward — peak GPU footprint scales linearly with
B. At `batch_size=1` on Qwen3-4B with q_len ≈ 200 prompt tokens and
k_len ≈ 21 K stacked context, that pile is ~36 × 32 × 200 × 21000 × 2 bytes
≈ 9 GB, comfortably inside the 80 GB H100 budget on top of the model
(~8 GB) and KV cache (~6–8 GB).

The cost is running 40 instrumented forwards per pair instead of ~10 batched
ones. Each forward at `batch_size=1` is cheap relative to generation, so the
wall-clock impact is small — empirically each pair finishes in ~15 minutes.

Real generation **stays batched** at the original heuristic
(`max(1, min(20, int(25000 / max_layer_len)))`) — only the instrumented
forward is single-sample. Generation has no `output_attentions`, so it
doesn't carry the same OOM risk.

If OOM still occurs at `batch_size=1`, the next escalation is to register a
forward hook on each `Qwen3Attention.forward` that captures the local
`attn_weights` tensor and replaces it with `None` in the layer's return
tuple, so the model never builds up a 36-layer stack. **Don't do this
preemptively** — only if `batch_size=1` is insufficient.

## 6. How the aggregator consumes the new schema

`scripts/aggregate_pair_results.py` is now intentionally lean:

- **`_accuracy_matrices`** builds 20×20 accuracy matrices (overall, pos1, pos2,
  delta) from each pair's `overall_accuracy`/`acc_pos1`/`acc_pos2`. These
  feed `_marginals` → `summary.json`. They are **not** plotted as heatmaps
  any more.
- **`_gather_attn_after(results, position, correctness)`** loads each pair's
  bucket at `attn_mass_after_aggregate.position_<n>.<correct|incorrect>`,
  stacks per-pair `(num_layers,)` mean and std arrays for each region, and
  averages across pairs. The pooled std is the mean of per-pair stds — not a
  rigorous confidence interval, just a band that shows variability.
- **`_plot_attention_mass`** produces a single comprehensive figure
  `attn_mass_after_per_layer.{png,pdf}`:
  - 2 rows × `(n_variants × 2)` cols
  - rows = question position (1, 2)
  - cols = (variant, correctness): `(naive, correct)`, `(naive, incorrect)`,
    `(rope_shift, correct)`, `(rope_shift, incorrect)`
  - each subplot: 3 lines (cache_A, cache_B, question) with mean ± std
    bands as `fill_between`, plus an n-count in the title

No other figures are produced. The previously-generated
`pair_accuracy_{naive,rope_shift,diff}.{png,pdf}` and
`attn_mass_before_heatmap.png` are deleted from the aggregator path; if you
need any of those views, the underlying numbers are in
`per_pair/results.json` and `summary.json`.

## 7. Caveats

- **Pooled std is descriptive, not inferential.** Each pair contributes a
  per-bucket std (across that pair's 14ish "correct" or 6ish "incorrect"
  questions). We then average those per-pair stds across pairs. This is a
  rough pooled estimate of within-bucket variability, not a proper standard
  error of the mean. Treat the bands as a "scale" indicator — wide bands =
  noisy signal — not as a confidence interval.
- **n_correct and n_incorrect are unbalanced.** A typical 70%-accuracy pair
  has ~14 correct and ~6 incorrect questions per position. The `incorrect`
  bucket's mean is therefore noisier than the `correct` bucket's mean.
  Cross-pair averaging in the full sweep helps, but the bands will still
  reflect this imbalance.
- **Parser failures (`pred=None`) count as incorrect.** This is the same
  convention as the per-patient run — if the model produces an answer the
  parser can't extract a choice from, we score it as wrong. This affects the
  `incorrect` bucket's composition (some entries are "wrong answer", some
  are "unparseable answer").
- **`A_mean + B_mean + Q_mean` may differ from 1.0** by float-roundoff
  (~1e-4 in practice).
- **The instrumented forward path mutates `CompactedPrefixCache` layers via
  `layer.update()`** — the implementation must rebuild `cache_gpu` from
  `cache_cpu` for the real generation call, the same way the original
  implementation did. (Unchanged.)
- **`attn_mass_before` is independent of question content** and therefore
  identical across all questions in a pair. It's stored once at pair level.
- **Per-question raw `attn_per_layer` is preserved** — the aggregate is
  derived from it and can always be recomputed (e.g., with a different
  correctness criterion or a different splitting axis).
