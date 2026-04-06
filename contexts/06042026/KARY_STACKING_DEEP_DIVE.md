# k-ary Stacking — Context, Storage, and Position-Range Analysis

This document inventories what we actually know about stacking `k` pre-compacted
LongHealth KV caches with Qwen3-4B, with the goal of deciding the maximum
tractable `k` for a follow-up experiment that extends the existing pair
(`k=2`) experiment to arbitrary depths. It is the prerequisite for any plan
to run the k-ary sweep — the budget calculus has to be settled first.

This is an analysis document, not a plan. All numbers are measured from the
caches currently on disk in `long-health/patient_{01,03..08}/cache.pt` (the 7
patients the pair experiment uses), and the relevant code in
`models/cache.py`, `compaction/compaction_methods/chunked.py`, and
`scripts/run_pair_experiment.py`.

## TL;DR

There are **two** quantities that get called "the length of a compacted cache"
and they are **not** the same:

1. **Compacted storage size** — how many K/V vectors actually live in the
   cache (memory cost). Highly **non-uniform per layer**: median 3.7 k tokens,
   max 11.6 k, min 0.5 k for a single patient. Across all 36 layers, ~155 k
   tokens per patient.
2. **Effective position range** — the span of RoPE positions the model still
   sees. Determined entirely by the cache's `original_seq_len` field, **not**
   by the compacted storage. For the 7 patients here this is 10.0–12.8 k each
   regardless of how aggressively the cache was compacted.

Stacking `k` caches additively grows **both** numbers. Storage grows linearly
in `k` (mean per-layer ≈ 4 k × `k`, total per-layer-sum ≈ 156 k × `k`), and
position range grows linearly in `k` (≈ 11.3 k × `k`).

The binding constraint for the k-ary experiment is **position range**, not
memory:

| `k` | position range | vs Qwen3-4B 40 960-token native window | mean per-layer storage | total stored across 36 layers | est. eager-fwd peak (B=1) |
|---|---|---|---|---|---|
| 1 | 11 280 | within (28 %) | 4 313 | 155 280 | 2.0 GB |
| 2 | 22 560 | within (55 %) | 8 626 | 310 560 | 4.0 GB |
| 3 | 33 840 | **within (83 %)** | 12 939 | 465 840 | 5.9 GB |
| 4 | 45 120 | +10 % over | 17 252 | 621 120 | 7.6 GB |
| 5 | 56 400 | +38 % over | 21 565 | 776 400 | 9.4 GB |
| 6 | 67 680 | +65 % over | 25 878 | 931 680 | 11.2 GB |
| 7 | 78 960 | **+93 % over** | 30 191 | 1 086 960 | 12.8 GB |

(Numbers above use the **mean** per-patient cache; the actual values for the 7
real patients are slightly different — see §3 — and the per-`k` table in §7
uses the actual cumulative-prefix construction `[patient_01, patient_03, …]`
rather than the mean.)

GPU memory is **not** the binding constraint: even at `k=7`, total
working-set fits inside ~30 GB on H100 80 GB. The binding constraint is that
the model is asked to do RoPE attention at positions up to ~79 k while it
was trained on 40 960. Qwen3 has `rope_theta = 1 000 000` (very large) which
gives more graceful extrapolation than vanilla `rope_theta = 10 000`, but
"more graceful" is not "free" — beyond ~2× the native window the standard
finding in long-context literature is that retrieval-heavy tasks degrade
materially. We have no in-house empirical measurement of where Qwen3-4B in
particular starts to drop off.

The remainder of this document substantiates each of the claims above with
code citations and measurements.

## 1. The two notions of "length" in a compacted cache

A `cache.pt` file produced by `scripts/run_per_patient.py` is a
`{model, ratio, article_len, compaction_time, original_seq_len, cache}` dict
plus per-layer KV tuples. The two relevant length-like fields are:

### 1.1 Compacted storage size

Each layer's `cache[layer_idx]` is a tuple `(C1, beta, C2)` of tensors with
shape:

- `C1`: `(B=1, KV_heads=8, t_layer, head_dim=128)` — the compacted "key" prefix
- `beta`: `(B=1, KV_heads=8, t_layer)` — log-space attention biases used by
  the compaction-aware attention to recover the missing-token mass
- `C2`: `(B=1, KV_heads=8, t_layer, head_dim=128)` — the compacted "value"
  prefix

`t_layer` varies from layer to layer because the AM-OMP compactor allocates a
non-uniform per-layer head budget. For `patient_01`, `t_layer` ranges from
**543** (layers 4 and 6) to **11 119** (layers 15 and 24). Median is **3 716**,
mean is **4 313**. This is **for one patient**.

Total stored tokens for one patient (sum across all 36 layers):
**155 280 K/V vectors**.

### 1.2 Effective position range

`cache.pt` also stores `original_seq_len` — the length of the article in
**uncompacted** tokens. For `patient_01` this is **12 265**. The compactor
discarded ~65 % of those tokens during compaction but the survivors retain
their **original** RoPE positions: each compacted key is RoPE-rotated to the
position it occupied in the full uncompacted sequence.

This is critical: a compacted cache is **not** a "shorter document". From the
model's point of view, the document is still 12 265 tokens long. The cache
just stores a sparse subset of the keys, and queries attend to those sparse
keys at their **original** positions.

### 1.3 Why the two differ

Compaction trades **storage** for **fidelity**. The position grid stays the
same (you cannot shorten a sentence by deleting words and pretending it was
shorter — the remaining words still need to know where they were). The
storage shrinks because most keys are redundant for downstream attention and
can be summarized by a smaller set of representative keys with calibrated
biases (`beta`).

This is why "10× compression" buys memory, not context window.

## 2. Per-patient measurements (the 7 patients the pair experiment uses)

Measured directly from `long-health/patient_*/cache.pt`:

| patient | `original_seq_len` | min `t_layer` | median | mean | max | sum across 36 layers |
|---|---|---|---|---|---|---|
| patient_01 | 12 265 | 543 | 3 716 | 4 313 | 11 119 | 155 280 |
| patient_03 | 12 831 | 567 | 3 885 | 4 509 | 11 626 | 162 341 |
| patient_04 | 11 517 | 512 | 3 490 | 4 051 | 10 439 | 145 819 |
| patient_05 | 10 075 | 451 | 3 056 | 3 546 | 9 134 | 127 660 |
| patient_06 | 11 288 | 502 | 3 421 | 3 970 | 10 231 | 142 920 |
| patient_07 | 10 982 | 489 | 3 327 | 3 861 | 9 950 | 139 011 |
| patient_08 | 10 002 | 447 | 3 031 | 3 518 | 9 061 | 126 642 |
| **mean** | **11 280** | **502** | **3 418** | **3 967** | **10 223** | **142 810** |
| **sum (k=7)** | **78 960** | — | — | — | **71 560** † | **999 673** |

† The "sum" row's "max" column is the **max-after-summing-across-patients**:
i.e., the worst-case stacked layer at `k=7` is layer 15, where every patient
contributed close to its own per-layer max, summing to 71 560. This is the
number that determines the most memory-hungry layer in eager attention at
`k=7`.

Distribution observations:

- **Highly skewed**. The bottom-third of layers have ~500–2 000 stored
  tokens each. The top-third have ~6 000–11 000. The median is well below the
  mean. This is the AM-OMP compactor allocating headroom to the layers it
  thinks need it.
- **Patients are structurally similar**. The non-uniform layer profile
  (which layers compress aggressively, which don't) is consistent across
  patients — layer 15 is always the largest, layers 4/6 always the smallest.
  Suggests it's an artifact of the compaction algorithm or the model
  architecture, not patient-specific content.
- **`original_seq_len` is tightly clustered** (10.0 – 12.8 k). The 7 patients
  have similarly-sized clinical notes, so the position-range arithmetic is
  almost linear in `k`.

## 3. The `ratio: 0.1` parameter — what it actually means

Each `cache.pt` records `ratio: 0.1` — but the **achieved** mean compression
is not 10 %. It is:

| patient | mean `t_layer` / `original_seq_len` | median / `original_seq_len` |
|---|---|---|
| patient_01 | 35.2 % | 30.3 % |
| patient_03 | 35.1 % | 30.3 % |
| patient_04 | 35.2 % | 30.3 % |
| patient_05 | 35.2 % | 30.3 % |
| patient_06 | 35.2 % | 30.3 % |
| patient_07 | 35.2 % | 30.3 % |
| patient_08 | 35.2 % | 30.3 % |

Notice the achieved ratios are **identical to three significant figures**
across all 7 patients. This strongly suggests `ratio: 0.1` is being
interpreted by the AM-OMP-fast compaction code as a **per-layer compute
budget multiplier**, not as a target compression ratio. The actual
compression that comes out is determined by the rank-selection logic
(probably `t_layer = clamp(round(orig_len * f(layer_idx)), …)` where the
per-layer factor `f` is allocated by the compactor's fidelity heuristic).

The practical consequence: **`ratio: 0.1` ≠ "compacted to 10 % of original"**.
The achieved mean is ~35 %; the worst layer keeps ~91 %. If we wanted a
genuinely 10 %-compacted set we would need to re-run `run_per_patient.py`
with a different parameterization (and re-validate accuracy on the original
single-patient task), which is out of scope here.

For this experiment, we **must take the existing caches as given**. The
budget arithmetic in §1 and §2 is what we have to plan around.

## 4. How `CompactedPrefixCache` handles RoPE

This is the heart of the issue. The relevant code lives in
`models/cache.py:130-258`.

### 4.1 Construction (`models/cache.py:130-193`)

```python
def __init__(self, compacted_cache, original_seq_len=None, …):
    layers = []
    max_compacted_len = 0
    for layer_idx, (C1, beta, C2) in enumerate(compacted_cache):
        layer = CompactedPrefixLayer(C1, beta, C2, clone=clone)
        layers.append(layer)
        max_compacted_len = max(max_compacted_len, C1.shape[-2])
    if original_seq_len is not None and max_compacted_len > 0:
        self._rope_base = int(original_seq_len) - int(max_compacted_len)
```

Three things to note:

1. **`max_compacted_len`** is the maximum across the per-layer `t_layer`
   values — for `patient_01` this is 11 119, not the mean 4 313.
2. **`_rope_base`** is computed as `original_seq_len - max_compacted_len`.
   For `patient_01`: `12 265 - 11 119 = 1 146`.
3. The `_rope_base` is **per-cache, not per-layer**. All layers share the
   same RoPE offset, even though their actual stored lengths differ.

### 4.2 What `get_seq_length()` returns (`models/cache.py:230-258`)

```python
def get_seq_length(self, layer_idx: int = 0) -> int:
    max_len = 0
    for i, layer in enumerate(self.layers):
        if i not in self._sliding_layer_indices:
            layer_len = layer.get_seq_length()
            if layer_len > max_len:
                max_len = layer_len
    return max_len
```

Returns the **max compacted length across all layers** — `max_compacted_len`
again, **not** `original_seq_len`. The docstring explains:

> This ensures the attention mask created by HF's masking utilities is
> large enough to accommodate all layers. Individual layers will slice the
> mask to their actual KV length.

So when the forward pass runs, `past_seen_tokens = max_compacted_len`, the
attention mask is sized to `max_compacted_len + input_len`, and each layer
internally slices off the tail it doesn't need.

### 4.3 Where the query token "lives" in position space

In `scripts/run_pair_experiment.py:286-300`:

```python
past_seen_tokens = cache.get_seq_length()       # = max_compacted_len
cache_position = torch.arange(
    past_seen_tokens, past_seen_tokens + input_len,
)
```

So `cache_position` runs from `max_compacted_len` to
`max_compacted_len + input_len - 1`. But the model also adds `_rope_base`
to compute the actual RoPE-applied position (this is done inside
`models/qwen3/modeling_qwen3.py` using `cache.rope_base()`).

Net effect: the **first query token's RoPE position** is

```
max_compacted_len + _rope_base
  = max_compacted_len + (original_seq_len - max_compacted_len)
  = original_seq_len
```

Exactly what we'd want — the question is being asked at position
`original_seq_len`, the natural "next token after the document". The
compacted keys live at their original positions in `[0, original_seq_len)`.
Compaction is invisible to the position arithmetic.

This **only** works because every compacted key was RoPE-rotated to its
correct original position during compaction (see
`compaction/compaction_methods/chunked.py:apply_rotary_pos_emb_to_cache` and
how it's used in the AM-OMP compaction loop).

### 4.4 Implication for stacking

When we stack `[cache_A, cache_B]`, we hand the result to
`CompactedPrefixCache(stacked_cache, original_seq_len=seq_len_A + seq_len_B)`.
Then `_rope_base = (seq_len_A + seq_len_B) - max_compacted_len_stacked`,
and the query lands at position `seq_len_A + seq_len_B`.

For `cache_B`'s compacted keys to be at coherent positions in this stacked
context, `_stack_caches` (`run_pair_experiment.py:71-130`) RoPE-shifts every
key in `cache_B` by `+seq_len_A` (when `variant=rope_shift`). After the
shift, `cache_B`'s keys live at `[seq_len_A, seq_len_A + seq_len_B)` — the
range immediately preceding the query, exactly where they would be in a
contiguous concat of the two original documents.

This generalizes naturally to `k` caches: cache `i` gets shifted by
`offsets[i] = sum(seq_len_j for j < i)`. The k-ary version of `_stack_caches`
does `k - 1` separate `compute_rope_correction` calls (one per cache after
the first), or one call with a `(k-1, 1)`-shaped target tensor — both are
cheap.

## 5. Stacking arithmetic

Given the per-patient table in §2, the stacked-cache properties at each `k`
follow mechanically. The construction we're modeling is "take patients in
some order, stack them with `rope_shift`-style position offsets, query the
combined cache about a question from one of the constituent patients."

For each `k`, we have these (k-dependent) quantities:

| quantity | scaling | binding constraint? |
|---|---|---|
| `stacked_original_seq_len` | sum of `original_seq_len_i` | **yes — RoPE positions exceed Qwen3 native window past `k=3`** |
| `max_compacted_len_stacked` | sum of `max(t_layer_i)` over layers, then max over the stacked layer | drives `attention_mask` size and `past_seen_tokens` |
| total stored K/V vectors | sum of `sum(t_layer_i)` for each layer | drives KV-cache memory ~ linear in this sum |
| max `t_layer` after stacking | `max_l ( sum_i t_{layer_l, i} )` | drives the worst-case eager-attention tile per layer |
| number of questions to evaluate | `20 × k` (LongHealth has 20 questions per patient) | drives total wall time per run |

## 6. Qwen3-4B context capacity

From `AutoConfig.from_pretrained("Qwen/Qwen3-4B")`:

```
max_position_embeddings: 40960
rope_theta:              1000000
rope_scaling:            None
```

Three observations:

1. **Native window is 40 960, not 32 768.** The `Qwen3-4B` checkpoint we're
   using was trained at a 40k window, not the 32k base of older Qwen
   variants. We have ~25 % more headroom than my earlier (incorrect) 32k
   assumption.

2. **`rope_theta = 1 000 000`** is unusually large (the vanilla LLaMA value
   is 10 000). This is the Qwen team's long-context-friendly RoPE base
   choice — it makes the cosine wavelengths span a much larger position
   range, which means RoPE extrapolation past the trained window degrades
   more gracefully than it would for a model with `rope_theta = 10 000`. The
   informal long-context-finetuning literature suggests Qwen3 with this
   theta usually holds up to roughly `1.5–2×` its native window before
   retrieval tasks visibly degrade. **We have no in-house measurement of
   this for Qwen3-4B specifically on LongHealth.**

3. **`rope_scaling: None`.** YaRN is not enabled by default. Qwen3 models
   officially support YaRN scaling with `factor=4.0` to extend the window to
   ~160k, but you have to set it explicitly in the config or via
   `rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}`
   at load time. Enabling YaRN changes the model's behavior for **all**
   positions, not just the extrapolated tail — so it would be a confound
   when comparing against pair-experiment results that didn't use YaRN.

The "safe operating range" for the k-ary experiment, defined as "stays
inside the trained window where we know quality is intact", is **`k ≤ 3`**.
The "probably-OK extrapolation range" is **`k ≤ 5`** if Qwen3's `rope_theta`
buys us ~1.5×; **`k ≤ 7`** if it buys us 2×. We don't know which is true
without measuring.

## 7. Per-`k` budget tables (using actual cumulative-prefix construction)

These tables use the **actual** patients in alphabetical order, prefix
construction `[patient_01, patient_03, patient_04, …]`. They're more
informative than mean-based estimates because the patients have different
sizes. (For a real run we'd want to run all permutations, not just this one
prefix — see §9.)

### 7.1 Position range vs. native window

| `k` | patients (cumulative) | `stacked_original_seq_len` | vs. 40 960 |
|---|---|---|---|
| 1 | 01 | 12 265 | within (30 %) |
| 2 | 01,03 | 25 096 | within (61 %) |
| 3 | 01,03,04 | 36 613 | within (89 %) |
| 4 | 01,03,04,05 | 46 688 | **+5 728 over (+14 %)** |
| 5 | 01,03,04,05,06 | 57 976 | +17 016 over (+42 %) |
| 6 | 01,03,04,05,06,07 | 68 958 | +28 002 over (+68 %) |
| 7 | all 7 | 78 960 | **+38 000 over (+93 %)** |

### 7.2 Storage and per-layer budgets

| `k` | mean per-layer `t` (stacked) | max per-layer `t` (stacked) | total stored across 36 layers | gen `batch_size` heuristic |
|---|---|---|---|---|
| 1 | 4 313 | 11 119 | 155 280 | 2 |
| 2 | 8 823 | 22 745 | 317 621 | 1 |
| 3 | 12 873 | 33 184 | 463 440 | 1 |
| 4 | 16 419 | 42 318 | 591 100 | 1 |
| 5 | 20 389 | 52 549 | 734 020 | 1 |
| 6 | 24 251 | 62 499 | 873 031 | 1 |
| 7 | 27 769 | 71 560 | 999 673 | 1 |

The `batch_size` column uses the existing pair-experiment heuristic
`max(1, min(20, int(25000 / max_layer_len)))`. From `k=2` upward, generation
is single-question (because `25 000 / max_layer_len < 2`).
**Batching does not help at any `k ≥ 2`** with this heuristic. If we want
batched generation at higher `k`, the heuristic itself would need revising.

### 7.3 Eager-attention peak memory at instrumented forward (B=1)

The instrumented forward (used to capture per-layer attention mass — see
`contexts/06042026/ATTENTION_MASS_SPEC.md`) runs at `batch_size=1` and asks
for `output_attentions=True`, which materializes a full
`(1, q_heads=32, q_len, kv_len_layer)` tensor per layer and accumulates all
36 of them in `out.attentions`. Peak memory contribution:

```
attn_peak ≈ q_len × num_q_heads × sum_l(t_layer_stacked_l) × 2 bytes
         = 200 × 32 × total_stored × 2
         = 12 800 × total_stored bytes
```

Plus the model weights (~8 GB in bf16), the KV cache itself (~`total_stored
× 8 KV-heads × 128 head-dim × 2 (k+v) × 2 bytes` = `~4 KB × total_stored`),
and a constant ~5 GB of activations and PyTorch buffers:

| `k` | total_stored | attn peak | KV cache | model | other | **total** |
|---|---|---|---|---|---|---|
| 1 | 155 280 | 2.0 GB | 0.6 GB | 8.0 GB | ~5 GB | **~15.6 GB** |
| 2 | 317 621 | 4.1 GB | 1.3 GB | 8.0 GB | ~5 GB | **~18.4 GB** |
| 3 | 463 440 | 5.9 GB | 1.9 GB | 8.0 GB | ~5 GB | **~20.8 GB** |
| 4 | 591 100 | 7.6 GB | 2.4 GB | 8.0 GB | ~5 GB | **~23.0 GB** |
| 5 | 734 020 | 9.4 GB | 3.0 GB | 8.0 GB | ~5 GB | **~25.4 GB** |
| 6 | 873 031 | 11.2 GB | 3.6 GB | 8.0 GB | ~5 GB | **~27.8 GB** |
| 7 | 999 673 | 12.8 GB | 4.1 GB | 8.0 GB | ~5 GB | **~29.9 GB** |

**OOM is not a concern up to `k=7` on H100 80 GB** at `batch_size=1`. There
is ~50 GB of headroom even at the maximum. The only memory-related risk is
if we try to batch the instrumented forward (don't — `ATTENTION_MASS_SPEC §5`
already rules this out).

### 7.4 Per-question wall time (rough estimate)

Forward pass on Qwen3-4B with eager attention is dominated by the
`q_len × kv_len` matmul in attention. For a fixed `q_len = 200` (a typical
LongHealth question prompt), this scales linearly in the per-layer KV
length, summed across layers:

| `k` | est. instrumented fwd / question | est. generation / question | per question | per run (`20k` questions) |
|---|---|---|---|---|
| 1 | ~1 s | ~5 s | ~6 s | ~2 min |
| 2 | ~2 s | ~10 s | ~12 s | ~8 min |
| 3 | ~3 s | ~15 s | ~18 s | ~18 min |
| 4 | ~4 s | ~22 s | ~26 s | ~35 min |
| 5 | ~5 s | ~30 s | ~35 s | ~58 min |
| 6 | ~7 s | ~40 s | ~47 s | ~94 min |
| 7 | ~8 s | ~50 s | ~58 s | ~135 min (~2.3 h) |

These are pessimistic guesses based on linear scaling of the binding
quantity (`total_stored`) and a "pair takes 8–10 min in practice" anchor
extrapolated from the existing single pair_experiment run we have on disk.
The numbers should be sanity-checked with a single `k=3` and a single `k=5`
SLURM run before committing to a large array.

## 8. Implications for the k-ary stacking experiment

### 8.1 Three operating regimes

| regime | `k` range | what's in the picture | what's NOT confounded |
|---|---|---|---|
| **In-window** | 1 ≤ k ≤ 3 | stacking depth + position-aliasing (naive vs rope_shift) + per-layer attention bucketing | RoPE extrapolation: model is operating inside its trained window throughout |
| **Borderline** | k = 4 | as above + ~14 % position extrapolation | Extrapolation is small and likely tolerable for Qwen3 with `rope_theta=1e6`, but the assertion is unverified |
| **Extrapolating** | 5 ≤ k ≤ 7 | as above + 38 %–93 % position extrapolation | Any accuracy degradation at these `k` is a mixture of (a) "more competing context" and (b) "position outside trained range" — we cannot cleanly attribute it to one or the other without a control |

### 8.2 What the experiment can cleanly answer at each regime

- **`k ≤ 3`** answers: "How does retrieval accuracy and last-query attention
  mass change as you stack more compacted patient documents into context,
  and does the rope_shift fix recover from any aliasing introduced by naive
  concat?" This is the cleanest version of the experiment and should be
  the **default**.
- **`k ≤ 5`** answers the same question with a depth axis that's longer but
  with a known ~38 % extrapolation tail at the high end. Useful for an
  exploratory "do trends continue?" probe but the curve will conflate two
  effects.
- **`k ≤ 7`** is the "go all the way" version. The full curve will be
  **descriptive** ("at `k=7` the model gets X % accuracy") but **not
  causally interpretable** ("X % is because of stacking depth Y % and
  extrapolation Z %"). We'd need controls to disentangle.

### 8.3 What it would take to push higher safely

Two viable paths:

1. **Enable YaRN scaling at model load.** Add
   `rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}`
   to the `Qwen3ForCausalLM.from_pretrained` call. This pushes the practical
   window to ~160k, well past `k=7`'s 79k position range. The catch is that
   YaRN changes the cosine wavelengths globally — it slightly perturbs
   attention even for in-window positions. So:
   - The pair experiment was run **without YaRN**, so a YaRN-enabled
     k-ary run can only be compared against itself, not against
     pair_experiment numbers.
   - We would want to run an **additional** YaRN-enabled `k=2` baseline as
     an apples-to-apples reference for the YaRN-enabled k=3..7 numbers.
   - YaRN's effect on accuracy at the small-`k` end (where extrapolation
     is irrelevant) would itself be a small but measurable shift, which is
     worth reporting.

2. **Pre-shift positions to a smaller virtual range.** This is a research
   direction, not an existing capability. Current `_stack_caches` pads cache
   `i` into the position range `[offsets[i], offsets[i] + seq_len_i)`,
   which preserves the original document layout but uses the full sum.
   An alternative would be to shift each cache into a much narrower range
   (e.g., place all cache `i` keys at positions `[i, i + max_compacted_len_i)`
   instead of at their original sparse positions). This is conceptually
   like "re-tokenizing" the compacted cache as if its compacted form *were*
   the original. It would require either (a) a different RoPE base for
   each cache or (b) a per-layer RoPE re-rotation to a packed range.
   Whether the model still does retrieval correctly under this scheme is
   an open question and would need its own validation. **Out of scope for
   the immediate k-ary plan.**

### 8.4 OOM concerns

**Not the binding constraint up to k=7.** All `k` fit comfortably on H100
80 GB with `batch_size=1` instrumented and `batch_size=1` generation. The
only thing the existing `ATTENTION_MASS_SPEC §5` warned about — eager
attention at higher batch sizes — already does not apply because the
heuristic forces `batch_size=1` for `k ≥ 2`. The escalation to a forward
hook that drops `attn_weights` after capture, mentioned in the spec, is
**not needed**.

The actual binding constraint on **cost** is wall time:

- `k=2` exhaustive (42 perms): ~6 GPU·h per variant
- `k=3` exhaustive (210 perms): ~63 GPU·h per variant
- `k=4` exhaustive (840 perms): ~490 GPU·h per variant
- `k=5` exhaustive (2 520 perms): ~2 440 GPU·h per variant
- `k=6` exhaustive (5 040 perms): ~7 880 GPU·h per variant
- `k=7` exhaustive (5 040 perms): ~11 340 GPU·h per variant

Even confining to `k ≤ 3` and running both variants exhaustively is only
~138 GPU·h, ~17 hours wall-clock at the Marlowe community-norm 8-job cap.
This is the smallest viable scientific design and very tractable.

## 9. Open questions to resolve before settling the plan

These are the questions I would want to answer (or have the user explicitly
decide on) before the k-ary plan is finalized.

1. **What is the actual quality of Qwen3-4B at 40k–80k positions on LongHealth?**
   We have no measurement. The cleanest way to find out is to run a single
   patient_01 query at progressively-larger artificial position offsets
   (e.g., add a dummy left-padded prefix of 0, 10k, 20k, 40k tokens before
   the question) and see when accuracy starts to drop. ~30 minutes of GPU
   time, would tell us whether `k=4` and `k=5` are usable.

2. **Should we enable YaRN?** If yes: are we OK with a non-comparable
   relationship to the pair-experiment results, and willing to run an extra
   YaRN `k=2` baseline?

3. **Is the user OK with `k ≤ 3` as the "definitive" version of the
   experiment, with `k=4..7` as an explicitly-flagged "exploratory tail"?**
   The two parts have fundamentally different interpretation — `k ≤ 3` is
   causally clean, `k ≥ 4` is descriptive only. A combined plan that runs
   both but documents the boundary clearly is one option; another is to
   drop the tail entirely until we have a YaRN baseline.

4. **Do we want to remeasure the pair (k=2) results within the k-ary
   framework?** Currently `long-health/pair_experiment/` has only **1 of 42**
   pair results on disk (just `pair_patient_01_patient_03`, both variants).
   The pair sweep is mostly unrun. If the k-ary experiment runs `k=2`
   internally, we get a consistent dataset and don't need to depend on the
   incomplete pair sweep at all. This is the recommended option.

5. **Is the existing AM-OMP-fast compaction `ratio=0.1` the right setting
   for this experiment?** The achieved compression is ~35 % per layer, not
   10 %. If we wanted a more aggressive compression (e.g., 20 % or 10 %
   actual), we would need to re-run `run_per_patient.py` with a different
   parameterization, validate that single-patient accuracy is still
   acceptable, and then run the k-ary experiment on the new caches.
   **Out of scope unless the user wants to take this on as a separate
   prerequisite.**

6. **Does the AM-OMP-fast compaction store any data at positions beyond
   `original_seq_len`?** From the code in §4 the answer is "no" — keys are
   strictly inside `[0, original_seq_len)`. But I have not confirmed by
   inspecting actual stored RoPE-rotated keys. A 30-second sanity check
   reading one cache layer's positional metadata would close this question.

## Appendix A — Numbers I would like to verify before finalizing

- [ ] Empirical eager-fwd peak memory at `k=3` (smoke test): expected ~21 GB
- [ ] Empirical wall-clock for one `k=3` run (60 questions): expected ~18 min
- [ ] Empirical wall-clock for one `k=7` run (140 questions): expected ~2.3 h
- [ ] LongHealth single-patient accuracy at synthetic prefix offsets
  `(0, 10k, 20k, 40k, 80k)` to chart Qwen3-4B's quality decay vs. position
- [ ] Whether YaRN-enabled `k=2` matches non-YaRN `k=2` accuracy on
  `pair_patient_01_patient_03` (the one pair we already have a number for)

## Appendix B — Code citations

| claim | file | lines |
|---|---|---|
| Compacted layer K/V tuple shapes | `models/cache.py` | 9–37 |
| `CompactedPrefixCache` constructor and `_rope_base` derivation | `models/cache.py` | 130–193 |
| `get_seq_length()` returning max compacted length | `models/cache.py` | 230–258 |
| `rope_base()` method exposing the offset to the model | `models/cache.py` | 198–211 |
| Pair `_stack_caches` and per-layer concat | `scripts/run_pair_experiment.py` | 71–130 |
| Pair RoPE shift via `compute_rope_correction(target_positions=[seq_len_a])` | `scripts/run_pair_experiment.py` | 96–104 |
| `compute_rope_correction` definition | `compaction/compaction_methods/chunked.py` | 115–186 |
| `apply_rotary_pos_emb_to_cache` definition | `compaction/compaction_methods/chunked.py` | 87–112 |
| Multi-chunk concat reference (`_concatenate_with_template`) | `compaction/compaction_methods/chunked.py` | 1570–1682 |
| Instrumented forward at B=1 + bucketing | `scripts/run_pair_experiment.py` | 236–340 |
| Generation batch-size heuristic | `scripts/run_pair_experiment.py` | 437 |
| Pair-experiment attention-mass spec | `contexts/06042026/ATTENTION_MASS_SPEC.md` | 1–230 |
