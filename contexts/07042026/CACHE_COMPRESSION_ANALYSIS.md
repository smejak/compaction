# Per-Patient KV Cache Compression Analysis — 2026-04-07

Investigation of how aggressively the per-patient compacted KV caches in
`long-health/patient_{01..20}/cache.pt` (produced by yesterday's SLURM run, see
`contexts/06042026/PER_PATIENT_RUN_SUMMARY.md`) actually compress the original
~10-13k token KV caches. The investigation started from a confusion about what
the `max_layer_len` field in `results.json` (~10k for patient_09) represents,
and ended with a clear separation between **attention-work compression**
(governed by per-head real K/V slots) and **persistent memory compression**
(governed by per-layer padded K/V tensor sizes).

Companion analyzer: `scripts/analyze_patient_caches.py`. Companion JSON output:
`long-health/cache_compression_summary.json`.

## TL;DR

| metric (mean over 20 patients) | value |
|---|---|
| original full KV length | ~11744 tokens |
| compaction target (`int(article_len * 0.1) + non_article`) | ~1196 |
| `max_layer_len` (single worst layer's padded length) | ~10643 |
| `mean_layer_len` (governs persistent memory) | ~4129 |
| dense-equivalent compacted length (governs attention work) | ~1196 |
| **attention compression** | **9.82×** ≈ design target of 10× |
| **memory compression** | **2.84×** — far below the attention number |
| **padding waste** (`-inf` slots / total slots) | **71.0%** |
| eval accuracy | 0.85 mean (matches `PER_PATIENT_RUN_SUMMARY`) |

The 20 per-patient rows are nearly identical to the means above (mem_x and
att_x and pad% are constant to 2 decimal places across patients) because the
budget profile is patient-agnostic and per-head budgets scale linearly with
`int(article_len * 0.1)`.

**Headline finding.** AM-OMP-fast under `optimized_agnostic.json` does compact
attention/softmax work to essentially the targeted 10×, but loses ~7×/10× of
the potential persistent-memory savings to within-layer cross-head padding.
Roughly 71% of the K/V slots stored on disk and in GPU memory are `-inf` beta
padding that is masked out at attention time. The single number
`max_layer_len` reported in `results.json` is the worst-layer's padded length,
which is dominated by 1-2 greedy heads per layer and is **not** a meaningful
summary of either compression number.

## Source of the confusion: what `max_layer_len` actually is

`results.json` contains a single size-related field, `"max_layer_len"`. For
patient_09 it is `9369`, and the user reasonably read that as "the compacted
KV cache is roughly 9.4k tokens long". Their question was whether that number
means (a) the compacted cache length, (b) the seq_len used for RoPE positional
embeddings, or (c) the original uncompacted full KV length.

**The answer is none of the three exactly.** Tracing
`scripts/run_per_patient.py`:

- Line 100: `seq_len = extract_full_kv_cache(...)[0]` — the full prefilled
  context length, ~10-13k for LongHealth. **NOT in results.json.** Stored
  inside `cache.pt` as `original_seq_len` (line 132).
- Lines 109-111: `target = int(article_len * 0.1) + non_article` — only the
  article portion is targeted for 10× compression; the ~25 non-article
  scaffold tokens (chat-template/system prompt) pass through unchanged.
- Line 152: `max_layer = max(c1.shape[2] for c1, _, _ in cache_cpu)`.
- Line 191: `"max_layer_len": max_layer` is what gets written to results.json.

So `max_layer_len` is **the maximum across the 36 transformer layers of the
C1 K-tensor's sequence dimension `t`**, where C1 has shape
`(B, KV_heads, t, head_dim)` (`models/cache.py:13-16`). Each layer has its own
`t_l`, and `max_layer_len` exposes the single largest one as its summary.

It is a fourth thing from the user's options: the largest per-layer
**padded** K-tensor length. The next two sections explain what that padding
is and why the largest such length is so close to the original seq_len.

## How `ratio=0.1` flows into per-head budgets

`compaction/compaction_methods/per_layer_head_on_policy.py:230-260`:

1. The article portion's compaction target is
   `actual_target_size = int(article_len * 0.1)` — about 900-1300 tokens for
   typical LongHealth articles.
2. Per-head budgets are read from
   `head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json`,
   which has 288 entries (36 layers × 8 KV heads, GQA) summing to 1.0.
3. Each head's absolute budget is computed as
   `int(proportion[l, h] * actual_target_size * total_heads)`. With 288 heads
   summing to 1.0, the **mean** per-head budget equals `actual_target_size`
   (~900 tokens). A perfectly uniform profile would give every head exactly
   `actual_target_size` tokens.
4. The total budget across all heads is roughly hard:
   `Σ_{l,h} budget[l, h] ≈ actual_target_size × total_heads ≈ 259200` real KV
   slots. This is the same total as if every head were uniformly compressed
   to ~900 tokens. The ratio is enforced **in aggregate**, not per-head.
5. The actual `optimized_agnostic.json` profile is extremely skewed:
   - 288 entries, all summing to 1.0
   - max/min ≈ 66.6×
   - mean / median ≈ 7.4×
   - max proportion ≈ 0.031 → max head budget ≈ `0.031 × 900 × 288 ≈ 8000`
     tokens (essentially the full sequence)
   - median proportion ≈ 0.00047 → median head budget ≈ ~120 tokens
6. The 0.1 ratio is also not strictly enforced from below: the per-head OMP
   solve can return fewer keys than asked when their coefficients fall under
   `drop_key_beta_cutoff = -7`
   (`per_layer_head_on_policy.py:474-490`). Heads short-changed this way
   contribute less than their nominal budget.

**"On-policy"** (`on_policy=True`, `nnls_interval=2`) is orthogonal to
budgeting. It only controls *which queries OMP minimizes residual against* —
re-extracted from on-policy generations every 2 layers, rather than from a
fixed precomputed query set. It does not affect per-head budget sizes.

## The within-layer padding scheme — and its memory cost

`per_layer_head_on_policy.py:566-587`. Each layer's heads are compacted
independently to their per-head budgets, producing tensors of varying lengths.
Then the layer's heads are **right-padded to a common length within that
layer** before being concatenated into the layer's K/V tensors:

```python
target_seq_len = max(h.shape[2] for h in C1_heads)
for i in range(len(C1_heads)):
    curr_len = C1_heads[i].shape[2]
    if curr_len < target_seq_len:
        pad_len = target_seq_len - curr_len
        C1_heads[i] = torch.cat([
            C1_heads[i],
            C1_heads[i].new_zeros(1, 1, pad_len, head_dim)
        ], dim=2)
        # ...
        beta_heads[i] = torch.cat([
            beta_heads[i],
            beta_heads[i].new_full((1, 1, pad_len), float('-inf'))
        ], dim=2)
```

So `t_l = max_h budget[l, h]`, and on-disk the `(K, V)` tensors for layer `l`
are sized `(1, 8, t_l, 128)` regardless of how short most heads' real
allocations are. The padding slots have `beta = -inf` so the attention softmax
multiplies them by zero — they contribute nothing to attention output, but
they **do** consume real K and V memory.

`max_layer_len` is then `max_l t_l`, i.e. `max_l max_h budget[l, h]` — the max
over both axes. For a profile where one head can be 60-70× larger than the
median, this is dominated by a tiny minority of (layer, head) slots and
easily reaches ~80-95% of `original_seq_len`.

## The critical subtlety: no cross-layer padding

A natural worry (which we worked through carefully) is whether HF's KV-cache
machinery requires all layers to share the same KV length, in which case
short layers would also be padded up to `max_l t_l` and the memory footprint
would be `n_layers × kv_heads × max_l t_l × head_dim × 2` regardless of how
non-uniform the per-layer budgets are. **It does not.**

`models/cache.py:31-33`: `CompactedPrefixLayer.__init__` assigns
`self.keys = C1` directly without padding. The 36 layers in
`CompactedPrefixCache.layers` literally hold 36 K-tensors of 36 different
sizes. `per_layer_head_on_policy.py:589-594` appends each per-layer tensor
independently to the result list — there is no second pass that aligns
layers.

`models/cache.py:72-73`: when a new generated token is appended during decode,
each layer just does `torch.cat([self.keys, key_states], dim=-2)` on its own
tensor. After several decode steps the cache may simultaneously hold
`layer_0.keys.shape == (1, 8, 8001, 128)` and
`layer_5.keys.shape == (1, 8, 201, 128)`.

The way this coexists with HF's preference for a single attention mask is
documented in the `CompactedPrefixCache.get_seq_length` docstring at
`models/cache.py:230-243`:

> *"For CompactedPrefixCache with variable-length global layers, we return
> the MAXIMUM compacted length across all global layers. This ensures the
> attention mask created by HF's masking utilities is large enough to
> accommodate all layers. **Individual layers will slice the mask to their
> actual KV length.** Note: cache_position will start at this maximum value,
> but that's okay because: (1) Attention masks are sliced per-layer to match
> actual KV lengths, (2) RoPE positions are corrected with per-layer
> rope_base offsets, (3) KV cache updates don't use cache_position (they
> just concatenate)."*

So HF allocates **one** transient attention mask of shape
`(B, 1, query_len, max_l t_l + query_len)` per forward pass. Each layer's
attention call slices that mask down to its own `[..., :t_l + query_len]`
before computing attention. The mask is freed when the forward returns and is
trivially small for `query_len = 1` decode (~8 KB at `max_l t_l = 8000`); it
does not turn into persistent KV storage.

**Persistent KV-cache bytes** are therefore:

```
compacted_bytes = 2 (K+V) × Σ_{l=0..n_layers-1} (KV_heads × t_l × d_head × dtype_size)
                = (KV_heads × d_head × dtype_size × 2) × Σ_l t_l
                = (constant) × n_layers × mean_l(t_l)
```

vs.

```
uncompacted_bytes = (same constant) × n_layers × original_seq_len
```

so:

```
memory_compression_ratio = original_seq_len / mean_l(t_l)
```

This is governed by **how many layers have a greedy head**, not by the single
worst layer or by the per-head dense-equivalent count.

## Two compression numbers, decoupled

The budget profile decouples two things that would otherwise be measured by a
single ratio:

- **Attention/softmax work compression** = `original_seq_len /
  dense_equivalent`, where `dense_equivalent = (Σ_{l, h} real_per_head[l, h])
  / (n_global_layers × kv_heads)`. This is determined by the *aggregate* of
  per-head budgets, which is hard-pinned to `actual_target_size × total_heads`.
- **Persistent memory compression** = `original_seq_len / mean_l(t_l)`.
  This is determined by the *layer-level distribution* of per-head budgets:
  if greedy heads cluster into a few layers, memory compression approaches
  the attention compression; if they spread across all layers, memory
  compression collapses toward 1×.

### Worked example (same total budget, two extreme distributions)

36 layers, 8 KV heads, `d_head = 128`, bf16 (2 bytes), `original_seq_len = 10000`.
Uncompacted KV bytes ≈ `2 × 36 × 8 × 10000 × 128 × 2 ≈ 1.47 GB`.

The budget profile assigns 288 head budgets summing to ~259200 total real KV
slots (`actual_target_size × total_heads = 900 × 288`) — that aggregate is
fixed by ratio=0.1 regardless of how it's distributed. Two scenarios:

| | Scenario A (clustered) | Scenario B (spread) |
|---|---|---|
| greedy heads concentrated in | 4 layers | all 36 layers |
| `t_l` for those layers | 8000 | 8000 |
| `t_l` for other layers | 200 (32 layers) | n/a |
| `Σ_l t_l` | `4×8000 + 32×200 = 38400` | `36×8000 = 288000` |
| `mean_l(t_l)` | 1067 | 8000 |
| compacted KV bytes | ~157 MB | ~1.18 GB |
| **memory compression** | **9.4×** | **1.25×** |
| total real KV slots | 259200 | 259200 |
| dense-equivalent length | 900 | 900 |
| **attention compression** | **11.1×** | **11.1×** |

Both scenarios do the same softmax work and store the same number of real
(non-padding) K/V entries. Their persistent memory footprints differ by
~7.5×. The single number `max_l(t_l)` is 8000 in both cases and tells you
nothing about which scenario you're in — only the full per-layer `t_l`
distribution does.

The actual `optimized_agnostic.json` profile sits between A and B. The
analyzer (next section) tells us exactly where.

## RoPE handling — the third thing the user worried about

`models/cache.py:122-127, 187-189`: when the cache is loaded into
`CompactedPrefixCache`, `rope_base = original_seq_len -
max_compacted_prefix_len`. New query tokens during eval get position IDs
starting at `past_seen_tokens` (which equals `max_l t_l` because of the
get_seq_length override discussed above), so applying `rope_base` puts the
*first new token* at original-sequence position `original_seq_len`,
immediately after the prefilled context.

The compacted prefix tokens themselves keep whatever RoPE positions were
rotated into K during prefill — the K tensors stored in `cache.pt` are
**already RoPE'd**. So the cache does not see densely re-numbered positions
`[0..K)`. It sees the original positions, with new tokens correctly tacked
onto position `original_seq_len`.

The "RoPE seq_len" the model effectively sees during eval is
`original_seq_len + new_tokens`, regardless of how aggressively the cache was
compressed. So neither (b) nor (c) from the user's original framing is what
`max_layer_len` represents.

## The analyzer

`scripts/analyze_patient_caches.py` — read-only script that loads each
`patient_XX/cache.pt` (CPU tensors only, no GPU needed) and reports the
metrics derived above. It cross-references each patient's accuracy from the
matching `results.json`.

### Verification (passed)

- For patient_09 the script reports `max_l = 9369`, exactly matching the
  `max_layer_len: 9369` already in `long-health/patient_09/results.json`.
  This validates that the script's tensor-shape interpretation matches the
  writer's.
- `original_seq_len` is in [10002, 13198] for all 20 patients — matches the
  ~10-13k LongHealth article ballpark.
- `target` is consistently `~int(article_len * 0.1) + ~25` (the scaffold in
  these single-patient articles is only ~25 tokens because `format_context`
  generates a very short chat-template wrapping for a one-article context).

### Per-patient summary (from `python scripts/analyze_patient_caches.py`)

```
patient      orig article   tgt max_l mean_l min_l p50_l dense  mem_x  att_x  pad%   acc
----------------------------------------------------------------------------------------
patient_01  12265   12240  1249 11119   4313   543  3716  1248  2.84x  9.82x 71.1%  0.80
patient_02  12601   12576  1282 11418   4429   557  3816  1281  2.84x  9.83x 71.1%  0.90
patient_03  12831   12806  1305 11626   4509   567  3885  1305  2.85x  9.83x 71.1%  0.90
patient_04  11517   11492  1174 10439   4051   512  3490  1174  2.84x  9.81x 71.0%  0.90
patient_05  10075   10050  1030  9134   3546   451  3056  1029  2.84x  9.79x 71.0%  0.85
patient_06  11288   11263  1151 10231   3970   502  3421  1151  2.84x  9.81x 71.0%  0.80
patient_07  10982   10957  1120  9950   3861   489  3327  1119  2.84x  9.81x 71.0%  0.90
patient_08  10002    9977  1022  9061   3518   447  3031  1021  2.84x  9.79x 71.0%  0.75
patient_09  10339   10314  1056  9369   3637   462  3134  1056  2.84x  9.79x 71.0%  0.95
patient_10  11099   11074  1132 10058   3903   494  3363  1131  2.84x  9.81x 71.0%  0.95
patient_11  12995   12970  1322 11781   4569   574  3936  1322  2.84x  9.83x 71.1%  0.85
patient_12  13198   13173  1342 11962   4639   583  3997  1342  2.84x  9.84x 71.1%  0.95
patient_13  12624   12599  1284 11436   4436   558  3822  1284  2.85x  9.83x 71.1%  0.75
patient_14  11579   11554  1180 10493   4071   514  3508  1180  2.84x  9.82x 71.0%  0.70
patient_15  11413   11388  1163 10339   4012   507  3457  1162  2.84x  9.82x 71.0%  0.95
patient_16  10939   10914  1116  9913   3847   487  3315  1116  2.84x  9.81x 71.0%  0.85
patient_17  11536   11511  1176 10457   4058   513  3496  1175  2.84x  9.81x 71.0%  0.90
patient_18  12432   12407  1265 11264   4369   550  3764  1264  2.85x  9.83x 71.1%  0.80
patient_19  13073   13048  1329 11844   4594   577  3957  1329  2.85x  9.84x 71.1%  0.75
patient_20  12099   12074  1232 10965   4254   536  3665  1232  2.84x  9.82x 71.0%  0.75
----------------------------------------------------------------------------------------
mean        11744   11719  1196 10643   4129   521  3558  1196  2.84x  9.82x 71.0%  0.85
```

Column meanings: `orig` = `original_seq_len`; `tgt` = compaction target;
`max_l/mean_l/min_l/p50_l` = per-layer `t_l` summary statistics; `dense` =
dense-equivalent compacted length (governs attention work); `mem_x` =
`orig / mean_l` = memory compression; `att_x` = `orig / dense` = attention
compression; `pad%` = `1 - real_slots / padded_slots` = fraction of stored
slots that are `-inf` padding; `acc` = pulled from each patient's
`results.json`.

**Three observations:**

1. **The 20 rows are essentially identical** in `mem_x`, `att_x`, and `pad%`
   to two decimal places. This is because the budget profile is
   patient-agnostic and the per-head budgets scale linearly with
   `int(article_len * 0.1)`. Compression ratios are determined entirely by
   the profile shape, not the patient content.
2. **Attention compression ≈ 9.82× ≈ design target.** The `dense_equivalent`
   column equals the `tgt` column to within rounding. AM-OMP-fast is doing
   what it was asked to do at the per-head budget level.
3. **Memory compression ≈ 2.84× ≪ attention compression.** Roughly 71% of
   the K/V slots stored in cache.pt (and held in GPU memory at eval time)
   are `-inf` padding. This is the cost of within-layer cross-head padding
   under the `optimized_agnostic.json` profile.

**Accuracy variance (0.70-0.95) is question-difficulty noise**, not
compaction quality — compression is constant across patients, so we can't
read a "better-compression-implies-worse-accuracy" tradeoff out of these
20 rows.

### Per-layer profile for patient_09

`python scripts/analyze_patient_caches.py --patients 9 --per-layer patient_09`:

```
Per-layer profile: patient_09
  n_layers=36  kv_heads=8  orig=10339  article=10314  target=1056
    l    t_l real_sum h_min h_p50 h_max
  -------------------------------------
    0   1055     4881   165   610  1055
    1   1055     2507   165   165  1055
    2   1352     4288   165   313  1352
    3   2837     6960   165   165  2837
    4    462     1617   165   165   462
    5   1055     2210   165   165  1055
    6    462     1914   165   165   462
    7   6103     7258   165   165  6103
    8   1055     2804   165   165  1055
    9   6103    12601   165   907  6103
   10   1055     3694   165   313  1055
   11   6103     8446   165   462  6103
   12   3134     6663   165   313  3134
   13   8479    19135   165   165  8479
   14   1946     3992   165   313  1946
   15   9369    17649   165   758  9369
   16   1649     4585   165   313  1649
   17   5212    14680   165  1352  5212
   18   4025    14381   165  1649  4025
   19   6697    12007   165   610  6697
   20   5212    16758   165  1649  5212
   21   7291    17055   165  1500  7291
   22   5806    12007   165   758  5806
   23   3134     8741   165   610  3134
   24   9369    25666   165  2243  9369
   25   2837     3992   165   165  2837
   26   3134     8445   165   610  3134
   27   2540     3992   165   165  2540
   28   2540     6662   165   610  2540
   29   3134     7555   165   462  3134
   30   3134     4883   165   165  3134
   31   1946     4882   165   165  1946
   32   3431     8742   165   313  3431
   33   3728    12007   165  1203  3728
   34   2540     7257   165   165  2540
   35   1946     3101   165   165  1946
```

Where the columns are: `l` = layer index, `t_l` = padded length for that
layer, `real_sum` = total real (non-padding) K/V slots summed across the 8
heads in that layer, `h_min/h_p50/h_max` = min/median/max real-slot count
across the 8 heads in that layer.

**What this shows.**

- `t_l` varies wildly across layers — from `462` (layers 4, 6) to `9369`
  (layers 15, 24). The greediest two layers are essentially uncompressed
  (~91% of the article body).
- **`h_min = 165` in every single layer.** Every layer has at least one head
  whose real-slot count equals 165 — the per-head "floor" budget produced
  by the smallest entries in `optimized_agnostic.json`. The pattern
  everywhere is "1-3 greedy heads + the rest at 165".
- **Middle layers (12-24) are the worst memory offenders.** `t_l` of 5000-9000
  in this range. Early (0-11) and very late (25-35) layers compress
  noticeably better.
- **Layer 24 is the extreme:** `t_l = 9369`, median head holds `2243` real
  keys, max head holds `9369`. Multiple greedy heads on a single layer
  compound the problem.
- The actual `optimized_agnostic.json` profile sits **firmly between Scenario
  A and B but closer to A** — greedy heads are spread across most layers
  (preventing the 9.4× memory ratio of pure clustering), but they're not so
  uniformly distributed that memory collapses to the 1.25× of the spread
  case. Result: memory compression ~2.84×.

The other 19 patients have qualitatively identical per-layer shapes (the
profile is patient-agnostic) — only the absolute `t_l` numbers scale with
each patient's `article_len`.

## Implications

1. **`max_layer_len` is a misleading summary.** It reports the worst layer's
   padded length, dominated by 1-2 greedy heads. It tells you nothing about
   either (a) attention/softmax work or (b) memory footprint. If results.json
   is going to expose a single number, the right ones are `mean_layer_len`
   and `dense_equivalent` — they cleanly separate the two distinct
   compression axes. (Could be added as a follow-on edit to
   `scripts/run_per_patient.py:181-193` so that future runs persist them.)

2. **The 0.1 ratio target is met for attention work but missed by ~3.5× for
   memory.** If the goal of compaction is to reduce GPU memory pressure (so
   longer contexts fit, or larger batches fit), the current AM-OMP-fast +
   `optimized_agnostic.json` config is leaving most of the available savings
   on the table. The fix is at the budget-profile level, not the algorithm
   level: either flatten the profile (less skew) or constrain it so greedy
   heads are spread more uniformly across layers (so no single layer has a
   greedy head, which would minimize each layer's `t_l`). A per-layer cap on
   `max_h budget[l, h]` would directly bound `t_l` and translate per-head
   compression into memory compression.

3. **The 0.1 ratio target is comfortably met for attention work.** The
   `dense_equivalent` column equals the `tgt` column in every patient. So if
   the goal is to reduce softmax compute (e.g. long-context inference
   throughput where attention is the bottleneck), the current config is
   already at the design target.

4. **Accuracy is independent of compression in this batch** because
   compression is constant. To study a compression-vs-accuracy curve, you'd
   need to vary `ratio` or the budget profile and rerun.

## Files

- **Analyzer:** `scripts/analyze_patient_caches.py` (read-only,
  CPU-only, ~200 LOC). Run modes:
  - `python scripts/analyze_patient_caches.py` — summary table for all 20.
  - `python scripts/analyze_patient_caches.py --patients 9 --per-layer patient_09`
    — per-layer table for one patient.
  - `python scripts/analyze_patient_caches.py --json
    long-health/cache_compression_summary.json` — also dump full per-patient
    stats.
- **Summary JSON:** `long-health/cache_compression_summary.json` — full
  per-patient dict including per-layer `t_l` arrays and per-head real-length
  arrays for downstream plotting.
- **Source files traced during the investigation:**
  - `scripts/run_per_patient.py:100-152, 181-193` — where `seq_len`,
    `max_layer`, and the on-disk cache.pt are produced.
  - `models/cache.py:13-16, 31-33, 72-73, 122-127, 187-189, 230-278` — the
    `CompactedPrefixLayer` / `CompactedPrefixCache` storage model and the
    nonuniform-length attention-mask trick.
  - `compaction/compaction_methods/per_layer_head_on_policy.py:230-260,
    456-490, 566-594` — where per-head budgets are read and where the
    within-layer cross-head padding happens.
  - `head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json`
    — 288 head proportions, summing to 1.0, max/min ≈ 66.6×.

## Pointers

- **Yesterday's run narrative:** `contexts/06042026/PER_PATIENT_RUN_SUMMARY.md`
- **Plan that produced the analyzer:**
  `~/.claude/plans/zippy-painting-plum.md`
- **Higher-level experiment context:** `contexts/06042026/EXPERIMENT_LOG.md`
