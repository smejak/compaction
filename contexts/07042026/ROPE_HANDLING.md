# RoPE Handling in AM-OMP-fast Compaction — 2026-04-07

How rotary position embeddings flow through the AM-OMP-fast KV-cache
compaction + eval pipeline. Companion to
`contexts/07042026/CACHE_COMPRESSION_ANALYSIS.md`, which documented the
size/memory side of the same caches. This file answers three concrete
questions:

1. Is it possible to see *exactly which* original-sequence RoPE positions are
   present in each layer/head of the per-patient compacted caches and the
   pair-patient stacked caches?
2. What RoPE positions do new query tokens get when a compacted cache is
   loaded for eval? If the cache contains "10k worth of RoPE", does the first
   query token sit at position 10000 or somewhere else?
3. What does that mean for inference performance and for how the model
   interprets the compacted cache?

## TL;DR

- **Each compacted slot is an exact original K vector with its
  original-position RoPE rotation baked in.** OMP picks a *subset* of the
  original keys (`C1[j] = K[selected_indices[j]]`) — no blending, no
  re-rotation. Stored K is post-RoPE because prefill already applied
  rotation inside the attention forward.
- **Eval-time queries land at RoPE position `original_seq_len + n`** — i.e.
  exactly where they would be in the uncompacted sequence, immediately after
  the original ~10k context. This is achieved by setting
  `rope_base = original_seq_len - max_l(t_l)` once at cache load time and
  adding it to `cache_position` when forming the model's `position_ids`.
- **The model interprets the prefix as "the original sequence with most
  keys missing".** Relative-position reasoning (e.g. relative angle
  `10000 - 3742 = 6258` between a query at position 10000 and a kept key
  originally at position 3742) is identical to the uncompacted case for
  every kept key. Compaction is *which keys to keep*, not *where they sit*.
- **Direct answer to question 1:** **No**, with the existing 20
  `cache.pt` files we cannot recover the per-(layer, head) original
  positions. The `selected_indices` are computed inside OMP but stripped
  before saving (see §5). Recovering them would require enabling
  `verbose_logging=True`, modifying the saver to keep the indices, and
  re-running compaction.

## 1. K tensors in cache.pt are post-RoPE — and never re-rotated

`extract_full_kv_cache` (`evaluation/utils.py:295-359`) calls
`model(input_ids, use_cache=True)` and returns `outputs.past_key_values`
verbatim:

```python
348:    with torch.no_grad():
349:        outputs = model(
350:            input_ids,
351:            use_cache=True
352:        )
...
359:    return seq_len, outputs.past_key_values, article_indices, formatted_context, original_token_length
```

HF/Qwen3 attention applies RoPE inside the attention forward during prefill,
so each K vector in the returned `past_kv` is **already rotated** to its
prefill-time absolute position. Nothing in the AM compaction pipeline
(`compaction/compaction_methods/per_layer_head_on_policy.py`,
`compaction/algorithms/omp.py`) un-rotates or re-rotates K — those modules
operate on the rotated K vectors as opaque `(T, head_dim)` matrices.

So before we even get to compaction, every K vector in the original full
cache has its absolute position baked in via RoPE rotation.

## 2. OMP selects a *subset* of original K vectors, no blending

`compaction/algorithms/omp.py:25-112` is the reference OMP implementation.
The greedy loop tracks `selected_indices` as a list of integer positions
into the original `K`:

```python
82:    selected_indices = []
83:    mask = torch.zeros(T, dtype=torch.bool, device=device)
84:    current = torch.zeros_like(target)
85:
86:    for _ in range(t):
87:        residual = target - current
88:        ...
94:        idx = corr.argmax().item()
95:        selected_indices.append(idx)
...
107:    indices_tensor = torch.tensor(selected_indices, device=device)
108:    C1 = K[indices_tensor]
109:    beta = torch.log(B).to(dtype)
```

So for compacted slot j, `C1[j]` is **exactly** `K[selected_indices[j]]` — a
single original K vector with whatever RoPE rotation was baked in at prefill
for that original sequence position. There is **no linear combination, no
synthesis, no blend.** The OMP `beta` (log-weights from the NNLS solve, line
110) is an additive attention bias that gets added to the dot-product score
during eval — it does **not** modify the K vector itself.

The wider per-(layer, head) wrapper preserves this. Each per-head call at
`compaction/compaction_methods/per_layer_head_on_policy.py:470-490`:

```python
470:                    C1_compact, beta_compact, C2_compact, selected_indices = algorithm.compute_compacted_cache(
471:                        K, V, queries_for_compaction, head_target_size
472:                    )
```

returns `selected_indices` as a list of integers per (layer, head). Different
heads pick different indices independently from the same original `K`.

**One small caveat about V (irrelevant for RoPE).** The `c2_method='lsq'`
config means the compacted *values* `C2_compact` may be a least-squares fit
rather than a clean `V[selected_indices]` — so values *can* be blends. But
RoPE only lives on K, so this doesn't affect any of the position reasoning.

## 3. Per-position semantics in the per-patient cache

Putting §1 and §2 together: for any compacted slot `(layer l, head h, slot
j)` in a per-patient `cache.pt`, the K vector `C1[l][0, h, j, :]` represents
the original token at position `selected_indices[l][h][j]` in `[0,
original_seq_len)`. The RoPE rotation for that exact position is already
applied to the vector.

The subtle thing — and the answer to question 1 — is that **the labeling is
not stored on disk**. The K tensor is there, but the integer `selected_indices`
that would tell you "this slot used to be position 3742" is gone (see §5).

Independent of the labeling, we already know from
`CACHE_COMPRESSION_ANALYSIS.md` that:

- Per-layer compacted prefix length `t_l = max_h budget[l, h]` varies
  layer-to-layer.
- Within a layer, heads with smaller per-head budgets are right-padded to
  `t_l` with `-inf` beta. Those padding slots have garbage K (zeros from
  `K.new_zeros(num_padding, head_dim)` at
  `per_layer_head_on_policy.py:483-484`) — they're masked out by `-inf` beta
  at attention time, but if you were to interpret them naively they would
  look like "RoPE position 0" because zero-vector K has no rotation.
  **Don't include padding slots in any per-position analysis** — gate on
  `torch.isfinite(beta)`.

For the kept (non-padding) slots, all we can say from the existing cache.pt
files is the *count*: per-layer total real slots, per-head real-slot count.
We already have those in `long-health/cache_compression_summary.json`. What
we *don't* have is which integer positions those slots correspond to.

## 4. Query positions at eval time

The position formula for new query tokens during eval lives at
`models/qwen3/modeling_qwen3.py:440-457`:

```python
440:        # All layers use the same rope_base since cache_position is based on max_cache_length
441:        rope_position_ids = None
442:        if position_ids is None and not isinstance(past_key_values, CompactedPrefixCache):
443:            position_ids = cache_position.unsqueeze(0)
444:
445:        if isinstance(past_key_values, CompactedPrefixCache):
446:            position_ids = cache_position.unsqueeze(0)
447:            rope_base = past_key_values.rope_base()
448:            pad_counts = past_key_values.pad_counts()
449:
450:            if pad_counts is not None:
451:                rope_position_ids = cache_position.unsqueeze(0) - pad_counts.unsqueeze(1) + rope_base
452:                rope_position_ids = torch.clamp(rope_position_ids, min=0)
453:            else:
454:                rope_position_ids = position_ids + rope_base
```

Where `cache_position` comes from `models/generate.py:404, 420-425`:

```python
404:    past_seen_tokens = cache.get_seq_length()
...
420:    cache_position = torch.arange(
421:        past_seen_tokens,
422:        past_seen_tokens + input_len,
423:        device=device,
424:        dtype=torch.long,
425:    )
```

`cache.get_seq_length()` is overridden at `models/cache.py:230-258` to return
**`max_l(t_l)`** (the global max across layers), with the docstring at lines
234-243 explaining why this is safe (per-layer mask slicing, see §6 of
`CACHE_COMPRESSION_ANALYSIS.md`).

And `rope_base` is computed once at cache construction time at
`models/cache.py:188-189`:

```python
188:        if original_seq_len is not None and max_compacted_len > 0:
189:            self._rope_base = int(original_seq_len) - int(max_compacted_len)
```

Putting it together for the unbatched case (single sequence, no
`pad_counts`), the first new query token gets:

```
rope_position = past_seen_tokens + 0 + rope_base
              = max_l(t_l) + (original_seq_len - max_l(t_l))
              = original_seq_len
```

And the Nth new token sits at `original_seq_len + N`. So **query tokens
during eval are positioned exactly where they would be in the uncompacted
sequence**, immediately after the original ~10k context. The user's
intuition is correct: yes, queries start "at 10k+ indices for RoPE",
specifically at `original_seq_len`.

**The batched case** (`pad_counts is not None`) handles left-padding for
ragged batches: if a batch element was padded by `k` tokens at the front,
its `pad_counts[batch_idx] = k`, and the formula becomes
`rope_position_ids[b] = cache_position - k + rope_base`. The `clamp(min=0)`
prevents pad tokens from getting negative RoPE indices. This doesn't change
the per-sequence semantics — every real token still lands at its
uncompacted-sequence position.

**Why this works without re-rotating the prefix.** RoPE in HF/Qwen3 is
applied via `position_embeddings = self.rotary_emb(hidden_states,
rope_pos_ids)` at `modeling_qwen3.py:491-492`, which produces cos/sin tables
indexed by `rope_pos_ids`. Inside each layer's attention, only the **query**
gets rotated by the new cos/sin (the Q vectors are fresh from the linear
projection on the new tokens' hidden states); the cached K is used as-is
from the prefix. So:

- Q at position `original_seq_len + n` → rotated by RoPE angle
  `original_seq_len + n`.
- K at compacted slot j → already rotated by RoPE angle
  `selected_indices[l][h][j]` (baked in at prefill).
- Their inner product picks up the relative angle
  `(original_seq_len + n) - selected_indices[l][h][j]` exactly as RoPE
  intends.

This is *identical* to what would happen on the uncompacted cache. The model
sees no difference in positional reasoning. The only difference is which
keys exist in the prefix.

## 5. Recoverability of exact RoPE positions — answering question 1 directly

**Short answer: not from the existing 20 `long-health/patient_*/cache.pt`
files.** The information was discarded before save.

Where it lives in code:

- **Computed**: at `compaction/algorithms/omp.py:82-95` (greedy loop) and
  emerging at line 96 `selected_indices.append(idx)`. Returned at line 112
  along with `C1` and `beta`.
- **Stored into per-head stats**: at
  `compaction/compaction_methods/per_layer_head_on_policy.py:524-541`:

  ```python
  525:                head_stats = {
  526:                    'layer': layer_idx,
  527:                    'head': head_idx,
  528:                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
  529:                    'selected_indices_stats': {
  530:                        'count': len(selected_indices),
  531:                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
  532:                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
  533:                    },
  ...
  ```

  So `selected_indices` is only stored when `verbose_logging=True`. Note that
  even with verbose_logging off, the script *does* keep aggregate stats
  (`count`, `min`, `max`) per head — but those aren't useful for
  per-position visualization, only for "the largest position any head kept"
  / "the smallest" type questions.

- **Stripped before save**: at `scripts/run_per_patient.py:142-147`:

  ```python
  142:    # Clean up stats (may contain tensor refs)
  143:    if stats:
  144:        stats.pop("_original_chunk_caches", None)
  145:        stats.pop("_compacted_chunk_caches", None)
  146:        stats.pop("per_layer_head_metrics", None)
  ```

  `per_layer_head_metrics` is the dict that *contains* the per-(layer, head)
  `head_stats`, including `selected_indices` (when verbose_logging is on).
  The strip happens before `torch.save(...)` at line 130, so even if
  verbose_logging had been on for yesterday's run, the indices would still
  not have made it into `cache.pt`.

- **Yesterday's run**: used the AM-OMP-fast config from
  `scripts/run_per_patient.py:40-50`, which does **not** pass
  `verbose_logging=True`, and the strip ran. So the existing 20 `cache.pt`
  files have `K` (post-RoPE), `V`, and `beta`, but **not** the integer
  indices.

**What it would take to recover them.** Three changes, none of which I am
making in this writeup:

1. **Enable `verbose_logging=True` at the compaction call site.** This is a
   per-method flag passed through the `AM_OMP_FAST_KWARGS` dict in
   `scripts/run_per_patient.py:40-50`, or via a constructor argument to the
   compaction method. Verify the exact mechanism by reading the method's
   `__init__` and the loop body around `per_layer_head_on_policy.py:524-541`
   before flipping the flag.
2. **Persist the indices into `cache.pt`.** The minimal change is to
   extract `selected_indices` from `stats['per_layer_head_metrics']` into a
   compact `[36 layers][8 KV heads][variable per-head budget]` nested list
   right before `run_per_patient.py:142-147` strips the heavy fields, then
   add it to the saved dict at line 130-139 under a new key like
   `"selected_indices"`. The strip can stay in place; we only want the
   integer indices, not the rest of the per-head metrics blob.
3. **Re-run compaction for all 20 patients.** Reuse
   `scripts/marlowe/per_patient.sh` (the SLURM array script from yesterday).
   Existing `cache.pt` files would be overwritten, so move them aside or
   delete them first (they aren't committed — the previous commit
   intentionally excluded the large `.pt` files). Per
   `feedback_marlowe_no_login_compute.md`, this MUST go through `sbatch` —
   no login-node compaction. Yesterday's wallclock under 8-way concurrency
   was 44m to 1h23m per patient, so the array completes in ~1-2 hours total.

Once the indices are on disk, a small read-only analyzer
(`scripts/analyze_rope_positions.py`, parallel to
`scripts/analyze_patient_caches.py`) can:

- Per patient: histogram of selected positions across all (layer, head)
  pairs.
- Per (layer, head): the raw integer list, plus summary stats (early/middle/
  late thirds of the article, density per article subsection, etc.).
- Per pair: by §6, both naive and rope_shift pair stacks have positions
  derivable purely from the per-patient indices + each cache's
  `original_seq_len` field — no rerun of the pair experiment is needed to
  visualize stacked positions.

## 6. Pair experiment — naive vs rope_shift

The pair stacking happens at `scripts/run_pair_experiment.py:70-129`. Both
variants concatenate `(C1, beta, C2)` along the sequence dim, but they
differ in whether cache_B's K is re-rotated first.

### Naive variant

`scripts/run_pair_experiment.py:122-126`:

```python
122:        # Concat along sequence dim (-2). beta and C2 are position-independent.
123:        C1 = torch.cat([C1_a, C1_b], dim=-2)
124:        beta = torch.cat([beta_a, beta_b], dim=-1)
125:        C2 = torch.cat([C2_a, C2_b], dim=-2)
```

Both halves keep their original RoPE rotations untouched. **Position
aliasing**: if patient_A had a key at original position 3742 and patient_B
also had a key at original position 3742, after stacking both end up rotated
for position 3742. From RoPE's perspective the model sees two keys claiming
the same absolute position. Their pairwise relative angle is zero, so they
look spatially indistinguishable to attention — even though their content is
completely different. The model has no positional handle to tell them apart
within the prefix.

After stacking, `stacked_seq_len = seq_len_a + seq_len_b` (line 128), so
query tokens at eval land at RoPE position `seq_len_a + seq_len_b` (per the
formula in §4 with `original_seq_len = stacked_seq_len`). So causality at
the *score-bias* level is preserved (every prefix position is strictly less
than the query position). But the prefix itself has internal collisions:
patient_A occupies `[0, seq_len_a)` and patient_B *also* re-occupies
`[0, seq_len_b)`, with positions `< min(seq_len_a, seq_len_b)` aliased
between the two patients.

### rope_shift variant

`scripts/run_pair_experiment.py:96-119`:

```python
 96:    if variant == "rope_shift":
 97:        cos_diff, sin_diff = compute_rope_correction(
 98:            model,
 99:            current_positions=torch.tensor([0], device=device),
100:            target_positions=torch.tensor([seq_len_a], device=device),
101:            device=device,
102:            dtype=dtype,
103:        )
...
115:        if variant == "rope_shift":
116:            # Move cache_B keys to GPU for RoPE correction, then back to CPU
117:            C1_b_gpu = C1_b.to(device=device, dtype=dtype)
118:            C1_b_shifted = apply_rotary_pos_emb_to_cache(C1_b_gpu, cos_diff, sin_diff)
119:            C1_b = C1_b_shifted.to(device="cpu", dtype=C1_a.dtype)
```

`compaction/compaction_methods/chunked.py:115-186` (`compute_rope_correction`)
exploits the fact that two RoPE rotations compose: rotating a K already
rotated for position `i` by an additional angle `Δ` produces a K that
represents position `i + Δ`. The math is in the docstring at
`chunked.py:122-133`:

> *To shift RoPE from position `p` to position `p'`:
> `K_new = apply_rotary(apply_inverse_rotary(K_old, cos_p, sin_p), cos_p', sin_p')`,
> which is equivalent to applying rotation with angle `(p' - p)`:
> `cos_diff = cos(p' - p)`, `sin_diff = sin(p' - p)`.*

So `compute_rope_correction(current=0, target=seq_len_a)` precomputes the
cos/sin tables for offset `seq_len_a`, and `apply_rotary_pos_emb_to_cache`
(`chunked.py:97-112`) applies that rotation to every K vector in cache_B.
After the shift, every key in cache_B that previously represented position
`i ∈ [0, seq_len_b)` now represents position `i + seq_len_a ∈ [seq_len_a,
seq_len_a + seq_len_b)`. **Disjoint from cache_A's `[0, seq_len_a)`. No more
position aliasing.**

`beta` and `C2` are position-independent and concatenate unchanged at lines
124-125. After stacking, `stacked_seq_len = seq_len_a + seq_len_b` (same as
naive), and query tokens at eval land at RoPE position
`seq_len_a + seq_len_b`. The difference vs naive is purely in the prefix
position layout: clean tail-to-head concatenation with no overlaps.

This is the variant whose results live under
`long-health/pair_experiment/rope_shift/pair_patient_*` per yesterday's git
status.

### Visualizing pair-stack positions from per-patient indices

If the per-patient `selected_indices` were available (see §5), the
pair-stack position sets would be derivable purely from each patient's
indices + their `original_seq_len` field — no rerun of the pair experiment
needed:

- **naive `pair(A, B)`**, layer `l`, head `h`:
  `selected_indices_A[l][h] ∪ selected_indices_B[l][h]` (positions can
  overlap; this is the aliasing).
- **rope_shift `pair(A, B)`**, layer `l`, head `h`:
  `selected_indices_A[l][h] ∪ {i + original_seq_len_A
  for i in selected_indices_B[l][h]}` (positions are disjoint by construction).

## 7. Implications

### Performance

- **Zero model-side overhead from compaction itself.** RoPE is applied once
  at prefill and frozen into K. At eval time, query tokens get RoPE applied
  fresh (one cos/sin lookup + one rotate-half multiply per query, both
  cheap), and dot products against the prefix K proceed exactly as on an
  uncompacted cache. The model has no idea the cache is compacted. The only
  extra op inside attention is one fused additive `beta` bias on prefix
  slots — see `models/qwen3/modeling_qwen3.py:481-486` where the bool causal
  mask is converted to float specifically to make room for the per-slot
  beta values. That's a single add per attention call.
- **One-time cost for rope_shift in pair stacking.** Per pair load:
  - One `compute_rope_correction(0, seq_len_a)` call → produces a
    `(1, 1, head_dim)` cos/sin table.
  - 36 `apply_rotary_pos_emb_to_cache(C1_b_layer, cos_diff, sin_diff)`
    calls, each just a per-element multiply-add over cache_B's K.
  - All 36 happen on GPU then move back to CPU (`run_pair_experiment.py:117-119`).
  - This is trivial compared to a prefill (~minutes) or an eval (~minutes
    per question batch).

### How the model interprets the compacted cache

Exactly: **"the prefix is the original ~10k token sequence, but I am only
getting to attend to a subset of its keys, with a small additive bias on
each kept key to compensate for the discarded attention mass."**

- The query at position `original_seq_len + n` reasons relative to the
  *actual* original positions of the kept keys. For a kept key originally
  at position 3742, the relative angle is `(original_seq_len + n) - 3742`
  — exactly what RoPE would have produced on the full uncompacted cache.
- The discarded keys are gone. The model has no way to know they ever
  existed. The `beta` bias on the kept keys serves as a point-mass
  correction so the kept attention sums roughly equal what the full
  attention sums would have been.
- The "lossiness" of compaction is purely *which keys to keep*, never
  *where the kept keys sit*. There is no "position warping" or "dense
  re-numbering" anywhere in the pipeline.

This makes the choice of OMP target objective (matching the partition
function over attention scores, see `omp.py:78-85`) directly meaningful: the
goal of compaction is to preserve the model's attention output as if it had
seen the full sequence at the same positions, and the design achieves that
by leaving positions completely alone and only deciding which slots to keep.

## Files cited

- `evaluation/utils.py:295-359` — `extract_full_kv_cache`; line 359 is the
  return point that hands the post-RoPE `past_key_values` to the rest of
  the pipeline.
- `compaction/algorithms/omp.py:25-112` — reference OMP implementation;
  lines 82-96 are the greedy loop, line 108-109 is `C1 = K[indices_tensor]`.
- `compaction/compaction_methods/per_layer_head_on_policy.py:470-490,
  524-541` — per-(layer, head) wrapper; line 528 is the `verbose_logging`
  gate that decides whether `selected_indices` is stored.
- `scripts/run_per_patient.py:40-50, 100-152` — AM-OMP-fast config and the
  cache.pt save site; line 146 is the strip that drops
  `per_layer_head_metrics`.
- `models/cache.py:122-189, 230-278` — `CompactedPrefixCache` constructor
  with `rope_base = original_seq_len - max_compacted_len`, and the
  `get_seq_length` / `get_mask_sizes` overrides that report
  `max_l(t_l)` to HF.
- `models/qwen3/modeling_qwen3.py:440-457, 481-492` — eval-time
  `rope_position_ids` derivation, including the `pad_counts` branch and the
  bool→float mask conversion that enables additive `beta` biases.
- `models/generate.py:380-430` — `cache_position` construction and
  `CompactedPrefixCache` instantiation at the start of
  `generate_with_compacted_cache_batch`.
- `scripts/run_pair_experiment.py:70-129` — pair stacking, both `naive` and
  `rope_shift` variants.
- `compaction/compaction_methods/chunked.py:97-186` —
  `apply_rotary_pos_emb_to_cache` and `compute_rope_correction`, the
  primitives that make rope_shift work.

## Pointers

- **Companion analysis (memory + attention compression):**
  `contexts/07042026/CACHE_COMPRESSION_ANALYSIS.md`
- **Per-patient run narrative:** `contexts/06042026/PER_PATIENT_RUN_SUMMARY.md`
- **Pair experiment workflow:**
  `contexts/06042026/PAIR_EXPERIMENT_PHASE1_CLOSEOUT_AND_PHASE2_LAUNCH.md`
