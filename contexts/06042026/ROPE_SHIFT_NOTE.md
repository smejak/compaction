# Pair-Stacked Caches: RoPE Shift vs Naive Concatenation

Short note on the two stacking variants in the pair-stacking recency-bias
experiment (`scripts/run_pair_experiment.py`).

## The problem with naive concatenation

Each pre-computed cache in `long-health/patient_XX/cache.pt` was produced by
running one patient end-to-end through `extract_full_kv_cache` + AM-OMP-fast
compaction, starting from position 0. That means every stored key in
`cache.pt["cache"][layer][0]` (i.e. `C1`) already has RoPE baked in for its
original absolute position in `[0, original_seq_len_X)`. Nothing about
compaction removes or shifts those position embeddings — OMP picks a subset of
keys/values, but it does not rotate them.

If we now stack two of these caches as

```
[cache_A.C1] ++ [cache_B.C1]    along the sequence dim
```

we get two halves where cache_A's keys carry RoPE embeddings for positions
`[0, seq_len_A)` and cache_B's keys carry RoPE embeddings for positions
`[0, seq_len_B)`. The two ranges **overlap** at the RoPE-angle level. From the
attention mechanism's perspective, a cache_A key at "position 500" and a
cache_B key at "position 500" look like they live at the same place in the
sequence. Self-attention distinguishes tokens positionally through the
rotation angle applied to Q and K; if two K's live at the same angle, Q can't
tell them apart from a positional standpoint (only content distinguishes
them).

Concretely this is expected to:

1. **Hurt retrieval on the patient that would normally need fine-grained
   positional lookups**, because the position signal is scrambled.
2. **Create cross-interference** between cache_A and cache_B tokens that
   happen to alias onto the same RoPE angle, especially at indices where
   both patients' contexts had meaningful content.
3. **Bias attention toward the prompt** (the question, which lives at a clean,
   non-aliased position range) relative to either cache half — observable in
   the `attn_mass_after` telemetry.

We still set `original_seq_len = seq_len_A + seq_len_B` so that the prompt
tokens (the eval question) continue at RoPE positions
`[seq_len_A + seq_len_B, ...)`. That part is consistent between variants; only
the stored keys' angles differ.

## The fix: shift cache_B's keys

Before concatenating, rotate every cached key in cache_B by a uniform offset
of `seq_len_A`, mapping the stored keys from their chunk-local angle range
`[0, seq_len_B)` to the target range `[seq_len_A, seq_len_A + seq_len_B)`.

The repo already has this exact identity in
`compaction/compaction_methods/chunked.py` (used by text-based chunked
compaction, which concatenates independently-prefilled chunks):

```python
cos_diff, sin_diff = compute_rope_correction(
    model,
    current_positions=torch.tensor([0]),    # where the key currently is
    target_positions=torch.tensor([offset]), # where we want it to be
    device, dtype,
)
C1_b_shifted = apply_rotary_pos_emb_to_cache(C1_b, cos_diff, sin_diff)
```

Two practical notes:

- **Only keys (`C1`) need correction.** `beta` (the compacted attention
  biases) is scalar-valued and position-independent. `C2` (values) is also
  position-independent — only the key-query dot product carries rotation.
- **The offset is the same for every token in cache_B.** Because all of
  cache_B was processed starting at position 0 in its solo prefill, every
  stored key needs the same uniform shift of `+seq_len_A`. One call to
  `compute_rope_correction` per pair is enough; the returned cos/sin can be
  reused for every layer.

Sliding-window layers would normally need separate local-RoPE handling, but
Qwen3-4B is a dense model — `sliding_layer_indices = set()` — so no special
casing is needed.

## Caveat that survives the fix

Each saved cache embeds its own chat-template prefix (system message + user
turn opening) and suffix (`<|im_end|>` markers) inside its compacted tokens.
Stacking two caches puts patient B's chat-template prefix in the middle of
the joint context. RoPE shifting places these tokens at sensible positions,
but the sequence is still "weird" from a formatting standpoint — there is no
way to remove that without re-running prefill from scratch.

Because the same asymmetry applies to every pair in both orderings, it
cancels out when we compute `acc_pos2 − acc_pos1`. So this caveat does not
invalidate the recency-bias comparison, but it does put a ceiling on absolute
accuracy relative to a solo-patient run.

## What we expect to see in phase 2 vs phase 1

- **`acc_pos2` should improve** (second-position patient benefits most from
  no longer aliasing onto cache_A's angles).
- **`attn_mass_after` for cache_B should rise**, and cache_B should capture a
  share of attention that better matches its a-priori share in
  `attn_mass_before`.
- **The `acc_pos2 − acc_pos1` gap should shrink** if the gap under naive
  concat is driven by RoPE aliasing rather than some inherent "recency is
  good" effect.

If phase 2 shows no improvement over phase 1, that would be interesting on
its own — it would suggest either (a) that the naive aliasing doesn't bite
hard enough to matter in this regime (possible, since the stored compacted
cache only retains ~10% of the original keys and the aliased conflicts may
be rare) or (b) that something more fundamental about multi-document
stacking is the bottleneck.
