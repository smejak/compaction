# Pair-Stacked KV Cache Eval — Build Summary & Handoff

Status: **Phase 1 complete (42/42 ordered pairs × 2 variants, all `COMPLETED 0:0` 2026-04-06).**
Phase 2 expansion to 20 patients planned — see `PAIR_EXPERIMENT_REPORT.md` § Phase 2 full sweep.
Plan file: `~/.claude/plans/adaptive-stargazing-curry.md` (original phase-1 plan);
`~/.claude/plans/declarative-wandering-fox.md` (phase-1 closeout + phase-2 launch).

## Goal

For each ordered pair `(A, B)` of the 7 already-compacted LongHealth patients,
stack their pre-computed compacted KV caches as `[cache_A, cache_B]`, evaluate
all 40 questions (20 per patient) against the stacked cache, and compare
accuracy + per-layer attention-mass distribution across pair orderings. Tests
whether a recency bias shows up in stacked-cache retrieval, and whether
RoPE-shifting cache_B's stored keys (to fix position aliasing) changes the
picture.

## Files built

### Code
| File | Purpose |
|---|---|
| `scripts/run_pair_experiment.py` | Standalone CLI — loads two `cache.pt`, stacks them (`naive` or `rope_shift`), runs instrumented forward pass for `attn_mass_after`, runs real generation, saves `results.json`. Idempotent skip if the output file already exists. |
| `scripts/marlowe/pair_experiment.sh` | SLURM array job (0..41, 8 concurrent) that runs one pair per task via `$SLURM_ARRAY_TASK_ID`. Reads `VARIANT` env var (default `naive`) so the same script is reused for both phases. H100, 8 h wall. |
| `scripts/aggregate_pair_results.py` | Post-run aggregator — 7×7 heatmaps (overall, `acc_pos1`, `acc_pos2`, `delta`), cross-variant diff, per-layer `attn_mass_after` line plots, `attn_mass_before` heatmap, `summary.json` with marginals and recency-bias estimate. |
| `scripts/marlowe/aggregate_pair.sh` | SLURM wrapper for the aggregator. No GPU, 30 min wall, 16 GB RAM. Reads `VARIANTS` env var (default `"naive rope_shift"`). |

### Docs
| File | Purpose |
|---|---|
| `ROPE_SHIFT_NOTE.md` | Short explainer — why naive concat aliases RoPE positions, the uniform-shift fix, what stays broken (chat-template prefix lands mid-context), phase-1-vs-2 expectations. Exists **before** phase 2 so the rationale is committed alongside the results. |
| `PAIR_EXPERIMENT_HANDOFF.md` (this file) | Build summary and run checklist. |

## What still needs to run

In order:

```bash
# 0. Sanity-check: one pair locally on a small SLURM allocation
srun -G 1 -A marlowe-m000120-pm05 -p batch -t 01:00:00 --pty bash
module load cudatoolkit/12.5 cudnn/cuda12/9.3.0.75 conda/24.3.0-0
/users/jsmekal/.conda/envs/hard_drive/bin/python -u \
    scripts/run_pair_experiment.py --pair-idx 0 --variant naive
# Expect: results.json under long-health/pair_experiment/naive/pair_patient_01_patient_03/
#         with 40 per-question entries and non-empty attn_mass_before/after.
exit

# 1. Phase 1 — 42-pair naive array (8 concurrent)
sbatch scripts/marlowe/pair_experiment.sh

# 2. Monitor
squeue -u $USER
tail -f logs/pair_lh_pair_*_0.out

# 3. Interim aggregation after phase 1 finishes
VARIANTS="naive" sbatch --export=ALL,VARIANTS scripts/marlowe/aggregate_pair.sh
# Inspect long-health/pair_experiment/figures/pair_accuracy_naive.png

# 4. Phase 2 — 42-pair rope_shift array
VARIANT=rope_shift sbatch --export=ALL,VARIANT scripts/marlowe/pair_experiment.sh

# 5. Full aggregation after phase 2 finishes
sbatch scripts/marlowe/aggregate_pair.sh
# Inspect:
#   long-health/pair_experiment/figures/pair_accuracy_naive.{png,pdf}
#   long-health/pair_experiment/figures/pair_accuracy_rope_shift.{png,pdf}
#   long-health/pair_experiment/figures/pair_accuracy_diff.{png,pdf}
#   long-health/pair_experiment/figures/attn_mass_after_per_layer.{png,pdf}
#   long-health/pair_experiment/figures/attn_mass_before_heatmap.png
#   long-health/pair_experiment/summary.json
```

## Key design decisions

- **42 ordered pairs, no self-pairs** (user choice). `scripts/run_pair_experiment.py:PAIRS` is the deterministic pair list, pair-idx 0..41 is stable across submissions.
- **Both naive and rope_shift variants are implemented.** User requested naive first, then rope_shift after all naive pairs finish. The SLURM script reuses the same binary via a `VARIANT` env var — no code duplication.
- **Eager attention** (`attn_implementation="eager"`) is required so `output_attentions=True` returns per-layer attention tensors. This differs from `scripts/run_per_patient.py` which uses SDPA. The model is loaded directly via `Qwen3ForCausalLM.from_pretrained(..., attn_implementation="eager")` rather than through `load_model_and_tokenizer`, which hard-codes SDPA.
- **Two forward passes per question**: one instrumented pass (discarded) to collect attention weights for the last question token, then a fresh `cache_gpu` rebuild and the real generation call. Necessary because `CompactedPrefixCache` layers mutate in place during a forward (`layer.update()` appends new tokens to `self.keys`/`self.values`), so we must rebuild for generation to preserve the "clean cache per question" invariant.
- **`attn_mass_before`** is computed as `exp(beta).sum()` per cache region — `beta` is the log-space attention bias stored during compaction, so `exp(beta)` gives mass-like weights. Stored at pair level (question-independent).
- **`attn_mass_after`** is bucketed into `cache_A` / `cache_B` / `question` regions per layer using layer-specific `t_A_per_layer` and `t_B_per_layer` widths (per-layer nonuniform head budgets mean each layer has a different split point). Averaged over heads and batch before saving; per-head detail is discarded to keep the JSON manageable.
- **RoPE correction for phase 2**: a single `compute_rope_correction(..., target_positions=[seq_len_A])` call reuses the same `(cos_diff, sin_diff)` for every layer of cache_B — the shift is spatially uniform. Mirrors `compaction/compaction_methods/chunked.py:_concatenate_with_template`.
- **Batch size heuristic**: `max(1, min(20, int(25000 / max_layer_len)))`, same formula as `scripts/run_per_patient.py`. Stacked context is ~2× so batches will typically be ~half the solo-patient size.

## Things to watch on the first real run

1. **OOM on the instrumented forward** — eager attention materializes a full `(B, heads, q_len, k_len)` tensor per layer. If pair 0 OOMs during the `_run_instrumented_forward` step, the fix is to drop the instrumented batch size to 1 (generation can stay at the normal batch size). The current implementation uses one shared batch size for both, which is conservative but may still be tight on large pairs.
2. **Wall-clock budget** — eager is slower than SDPA, and we do ~2× forwards per question. 8 h per pair should be enough (40 questions, ≤21 k stacked context), but watch the first few completions. If they're pushing past 4 h each, bump `--time` or lower `max_new_tokens` on the generation call.
3. **First pair is the smoke test** — don't launch phase 2 until you've verified pair_idx=0 naive completes cleanly AND the numbers in `results.json` look sane (accuracies in a plausible range, attention-mass fractions summing to ~1 per layer).
4. **All 20 patients now have `cache.pt` on disk** (per `PER_PATIENT_RUN_SUMMARY.md`). The original 7-patient pair set is the phase-1 baseline; phase 2 expands `PATIENT_IDS` in `scripts/run_pair_experiment.py` to all 20 patients (P01–P20) and re-runs the SLURM array — phase-1 results are skipped via the existing idempotency check.

## Quick file index

```
scripts/
├── run_pair_experiment.py          # main CLI
├── aggregate_pair_results.py       # login-node-safe aggregator (also runnable via SLURM)
└── marlowe/
    ├── pair_experiment.sh          # SLURM array for the 42 pairs
    └── aggregate_pair.sh           # SLURM wrapper for aggregation

ROPE_SHIFT_NOTE.md                  # rope_shift explainer (phase 2 rationale)
PAIR_EXPERIMENT_HANDOFF.md          # this file
```

Pre-existing files that this experiment depends on (no changes made to any of them):
- `long-health/patient_{01,03..08}/cache.pt` — the 7 compacted caches to stack
- `models/qwen3/modeling_qwen3.py` — eager attention returns `attn_weights`
- `models/cache.py` — `CompactedPrefixCache`
- `models/generate.py` — `generate_with_compacted_cache_batch`
- `compaction/compaction_methods/chunked.py` — `compute_rope_correction`, `apply_rotary_pos_emb_to_cache`
- `evaluation/utils.py` — `format_question`, `parse_model_choice`
- `evaluation/datasets.py` — `load_dataset("longhealth")`
