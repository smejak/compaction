# Pair-stacked KV cache eval — data and figures index

This document is the discoverability index for the pair experiment. If you
arrive here later wondering "where is the per-pair attention data" or "what
do the numbers actually mean", start with the table below and follow the
references.

For the **schema** of every JSON field referenced here, see
`contexts/06042026/ATTENTION_MASS_SPEC.md`.
For the **build summary and run instructions**, see
`contexts/06042026/PAIR_EXPERIMENT_HANDOFF.md`.
For the **rope_shift rationale**, see
`contexts/06042026/ROPE_SHIFT_NOTE.md`.

## Where the data lives

| Path (relative to repo root) | Contents | Schema reference |
|---|---|---|
| `long-health/pair_experiment/<variant>/pair_<A>_<B>/results.json` | Per-pair results: `overall_accuracy`, `acc_pos1`, `acc_pos2`, `attn_mass_before` (per-layer beta-derived), `attn_mass_after_aggregate` (per-position-per-correctness mean+std per layer per region), and **`per_question[i].attn_per_layer`** with the raw per-question per-layer A/B/Q triples. One file per (variant, ordered pair). | ATTENTION_MASS_SPEC.md §3 |
| `long-health/pair_experiment/summary.json` | Per-variant accuracy marginals: `by_first_position`, `by_second_position`, `overall_mean`, `acc_pos1_mean`, `acc_pos2_mean`, `recency_bias_estimate` (= mean of pos2−pos1 across pairs). | aggregate_pair_results.py:_marginals |
| `long-health/pair_experiment/figures/attn_mass_after_per_layer.{png,pdf}` | The single comprehensive figure: 2 rows (pos1/pos2) × (n_variants × 2) cols (variant × {correct, incorrect}), each subplot showing per-layer mean ± std bands for cache_A / cache_B / question regions. | ATTENTION_MASS_SPEC.md §6 |

### Things that are stored as data but **not** rendered as figures

The following exist only in JSON form. If you want to view them, derive a
figure ad-hoc from the JSON; the aggregator does **not** generate them.

| Quantity | Where to read it |
|---|---|
| 7×7 accuracy heatmap (overall, pos1, pos2, delta) per variant | Reconstruct from per-pair `overall_accuracy`/`acc_pos1`/`acc_pos2` in each `results.json`; `aggregate_pair_results.py:_accuracy_matrices` is the reference implementation. |
| Cross-variant accuracy diff heatmap | `summary.json[variant]` for marginals; per-pair files for cell-level diffs. |
| `attn_mass_before` 7×7 heatmap (cache_A share per ordered pair) | Per-pair `results.json["attn_mass_before"]["mean_A"]` (and `mean_B`). |
| Per-question raw attention masses | `results.json["per_question"][i]["attn_per_layer"]["A"\|"B"\|"Q"]`. |

## Smoke test results

Single ordered pair `(patient_01 → patient_03)`, both variants, run on
2026-04-06 to validate the pipeline before the full 42-pair sweep.

| variant | overall | pos1 (P01) | pos2 (P03) | wall | output dir |
|---|---|---|---|---|---|
| naive | 80% | 85% | 75% | 14m 37s | `long-health/pair_experiment/naive/pair_patient_01_patient_03/` |
| rope_shift | 70% | 65% | 75% | 15m 04s | `long-health/pair_experiment/rope_shift/pair_patient_01_patient_03/` |

(Numbers above are from the round-2 schema, written by the smoke-test jobs
234630/234631 and preserved by `run_pair_experiment.py`'s idempotency check
during the full sweep. The figure `attn_mass_after_per_layer.{png,pdf}` is
the canonical visualisation; per-variant heatmaps are not rendered.)

## Phase 1 full sweep (7 patients, 42 ordered pairs) — complete

Completed 2026-04-06 by chained array jobs **234790** (naive) → **234791**
(rope_shift) → **234792** (aggregator). Per-pair JSONs (84 files),
`summary.json`, and `attn_mass_after_per_layer.{png,pdf}` are committed in
the same commit that introduces this updated report.

Per-pair wall-clock averaged ~13 min (naive) / ~15 min (rope_shift); each
SLURM array task ran a chunk of 6 pairs (~1h 20m per task).

| variant | mean overall | mean acc_pos1 | mean acc_pos2 | mean recency_bias_estimate |
|---|---|---|---|---|
| naive | 0.7583 | 0.7750 | 0.7417 | −0.0333 |
| rope_shift | 0.7292 | 0.6679 | 0.7905 | +0.1226 |

**Headline.** Naive concat shows a *slight primacy* effect (questions about
the first-position patient score 3.3 pp higher than questions about the
second-position patient). Uniform RoPE-shifting cache_B's keys *flips* the
bias to a strong recency effect (+12.3 pp), while overall accuracy actually
drops by ~3 pp (acc_pos1 loses 11 pp, acc_pos2 gains 5 pp). The bias is
real and is dominated by position aliasing under naive concatenation.

## Phase 2 full sweep (20 patients, 380 ordered pairs) — planned

Phase 1's strong recency-bias signal under `rope_shift` is worth confirming
on the larger patient set now that all 20 LongHealth caches are on disk.
Phase 2 expands `PATIENT_IDS` from 7 to all 20 (P01–P20), giving 20×19 = 380
ordered pairs. The 42 phase-1 pairs are skipped via the existing idempotency
check (`run_pair_experiment.py:389-392`), so phase 2 runs **338 new pairs ×
2 variants = 676 new evaluations**.

Submission shape (sized to fit the 32-job submit limit on
`marlowe-m000120-pm05`):

- `PAIRS_PER_TASK=24`, `--array=0-15` → 16 array tasks per variant
- naive submitted first; rope_shift chained via `--dependency=afterok:<naive_jid>`
- Peak queue depth = 32 (16 naive R/PD + 16 rope_shift PD/Dependency)
- Aggregator submitted manually after rope_shift completes (would be the 33rd job)

Estimated wall ≈ 6–7h naive + 6–7h rope_shift, GPU contention permitting.

After completion, the marginals tables here become 20×20 and the
`summary.json` `by_first_position` / `by_second_position` dicts will have
20 keys instead of 7.
