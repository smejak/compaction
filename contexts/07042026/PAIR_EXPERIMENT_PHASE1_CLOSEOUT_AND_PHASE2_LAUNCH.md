# Pair Experiment — Phase 1 closeout + Phase 2 launch (2026-04-07)

Session journal for the work done on 2026-04-07. Companion to
`contexts/06042026/PAIR_EXPERIMENT_REPORT.md` (the long-lived index of
where data lives) and `contexts/06042026/PER_PATIENT_RUN_SUMMARY.md` (the
post-mortem of the per-patient SLURM run that produced the 13 missing
caches).

## TL;DR

- Phase 1 of the pair-stacked KV cache eval (42 ordered pairs over 7
  patients × 2 variants) was already finished on disk by the previous
  agent on 2026-04-06 but **never committed**. This session committed
  it (`b55434a`).
- Phase 2 expansion to all 20 LongHealth patients (380 ordered pairs ×
  2 variants = 760 evaluations, of which 676 are new) was prepped
  (`0911b9e`) and submitted in two stages.
- All 32 phase-2 SLURM tasks (16 naive + 16 rope_shift) are now either
  RUNNING or COMPLETED. The aggregator runs after the last rope_shift
  task finishes.

## Headline result (phase 1, unchanged from 2026-04-06)

| variant     | overall | acc_pos1 | acc_pos2 | recency_bias (pos2 − pos1) |
|---|---|---|---|---|
| naive       | 75.83%  | 77.50%   | 74.17%   | **−3.33%** (slight primacy) |
| rope_shift  | 72.92%  | 66.79%   | 79.05%   | **+12.26%** (strong recency) |

The RoPE correction *flips* the position bias. Naive concat shows a small
primacy effect; uniform RoPE-shifting cache_B's keys to non-aliased
positions costs ~11 pp on acc_pos1 but gains ~5 pp on acc_pos2, net
overall accuracy down ~3 pp. The bias is real and is dominated by
position aliasing under naive concatenation. Phase 2 will tell us
whether this holds at the 9× larger pair count.

## What happened in this session

### 1. Discovery phase

Picked up where the previous agent left off (last action was a
`sleep 600 && squeue` poll waiting for the aggregator). State on
arrival:

| component | state |
|---|---|
| naive 234790 (array 0-6) | all COMPLETED 0:0 |
| rope_shift 234791 (array 0-6) | all COMPLETED 0:0 |
| aggregator 234792 | COMPLETED 0:0, wrote summary.json + figure |
| 84 per-pair `results.json` | on disk under `long-health/pair_experiment/{naive,rope_shift}/pair_*/` |
| stale figures from pre-round-2 aggregator | still tracked in git, no longer emitted |
| `PAIR_EXPERIMENT_REPORT.md` | full of `_TBD_` placeholders |
| `PAIR_EXPERIMENT_HANDOFF.md` line 3 | `Status: all code written, nothing submitted to SLURM yet` |
| `git status` | 100+ untracked / modified entries, **no commit had been made** |

Also discovered (not in the previous handoff) that **all 20 patient
caches** are now on disk — patient_02 + patient_09..20 were compacted
on 2026-04-06 in the parallel SLURM per-patient run documented in
`contexts/06042026/PER_PATIENT_RUN_SUMMARY.md`. The phase-1 7-patient
scope was a constraint at the time the pair experiment was designed,
not a deliberate choice. Verified all 20 caches use the same shape:
`(1, 8, t, 128)` C1/C2, `(1, 8, t)` beta, 36 layers each.

### 2. Phase 1 closeout commit (`b55434a`)

One commit, 100 files changed (548,965 inserts, 2,975 deletes), bundling:

- **Round-2 schema rewrite** (already on disk, modified-but-uncommitted):
  - `scripts/run_pair_experiment.py` — per-question raw `attn_per_layer`
    triples + correctness-split `attn_mass_after_aggregate`
  - `scripts/aggregate_pair_results.py` — lean aggregator producing only
    the per-layer attention figure + `summary.json`
  - `scripts/marlowe/pair_experiment.sh` — chunked array task with
    PAIRS_PER_TASK to fit the 32-job submit limit
  - `contexts/06042026/ATTENTION_MASS_SPEC.md` — round-2 schema contract
- **84 per-pair `results.json` files** (42 naive + 42 rope_shift)
- **`summary.json`** with phase-1 marginals
- **`attn_mass_after_per_layer.{png,pdf}`** (overwritten by phase-1 aggregator)
- **`PAIR_EXPERIMENT_REPORT.md` edits**: smoke-test table filled in (80%/85%/75% naive, 70%/65%/75% rope_shift, walls 14m37s/15m04s); "Full sweep (TBD)" section renamed to "Phase 1 full sweep — complete" with the marginals table and headline; new "Phase 2 full sweep — planned" section appended
- **`PAIR_EXPERIMENT_HANDOFF.md` edits**: status header → phase 1 complete; patient_02-missing note → all 20 patients have caches
- **7 stale figures `git rm`'d** (`pair_accuracy_{naive,rope_shift,diff}.{png,pdf}`, `attn_mass_before_heatmap.png`) — the round-2 aggregator no longer emits them

### 3. Phase 2 expansion commit (`0911b9e`)

Four files, all small:

- `scripts/run_pair_experiment.py:39-51` — `PATIENT_IDS` from 7 → 20
  (P01–P20). `PAIRS = [(a,b) for a in PATIENT_IDS for b in PATIENT_IDS
  if a != b]` auto-extends to 380. The phase-1 results.json files are
  skipped via the existing idempotency check at line 389. Docstring
  index range `0..41` → `0..379`.
- `scripts/aggregate_pair_results.py:32-40` — same `PATIENT_IDS`
  expansion. The 7×7 numpy matrices auto-resize to 20×20 because they
  use `N = len(PATIENT_IDS)`.
- `scripts/marlowe/pair_experiment.sh:10,48,49` — `--array=0-15`,
  `PAIRS_PER_TASK=24` (default), `TOTAL_PAIRS=380`. Header comment
  rewritten to explain the 32-job submit-limit fit.
- `contexts/06042026/ATTENTION_MASS_SPEC.md` — scope numbers
  (`84 → 760` pairs in storage estimate, `7×7 → 20×20` matrices).

### 4. SLURM submission gymnastics

The 32-job submit limit on `marlowe-m000120-pm05/medium` ate this
sequence in three rounds:

**Round 1 — naive submission** (succeeded immediately):

```bash
JID_N=$(sbatch --parsable scripts/marlowe/pair_experiment.sh)
# → 236625 (naive, --array=0-15, 16 tasks)
```

**Round 2 — rope_shift first half**: tried to submit the full 16-task
rope_shift array chained on naive completion, denied with
`MaxSubmitJobsPerAccount`. Account state was 32/32 already because
ccitren+jjlunger collectively held 8 running jobs and my naive 16 added
to 24. Manually reduced to `--array=0-7` (8 tasks):

```bash
VARIANT=rope_shift sbatch --parsable --array=0-7 \
    --dependency=afterok:236625 \
    --export=ALL,VARIANT scripts/marlowe/pair_experiment.sh
# → 236687 (rs_first_half)
```

This filled the queue to exactly 32/32. The remaining 8 rope_shift
tasks (`--array=8-15`) had to wait for slots to free up.

**Round 3 — rope_shift second half** (after a gap of several hours):
the naive 236625 array completed, the rs_first_half 236687 array
started running, and the account queue dropped to 6/32. Tried
`--array=8-15 --dependency=afterok:236625`, which **failed with `Job
dependency problem`** — slurm appears to reject `afterok` on a job
array that has already fully completed and aged out of the active
queue. Retried *without* the dependency (naive is done, nothing to
wait for):

```bash
VARIANT=rope_shift sbatch --parsable --array=8-15 \
    --export=ALL,VARIANT scripts/marlowe/pair_experiment.sh
# → 237733 (rs_second_half)
```

All 8 tasks immediately RUNNING.

### 5. Loop scheduling

Set up a `/loop` cron at `7,37 * * * *` (every 30 min, off the popular
:00/:30 marks per harness guidance) to keep polling and submit the
remaining rope_shift fragment when slots free. The cron continues to
fire and will catch the rope_shift completion → aggregator submission
step.

## Final state at end of session

| job | array | tasks | state | end ETA |
|---|---|---|---|---|
| naive 236625 | 0-15 | 16 | **all COMPLETED** ✓ | done |
| rs_first_half 236687 | 0-7 | 8 | 5 RUNNING, 3 COMPLETED | ~4h |
| rs_second_half 237733 | 8-15 | 8 | **all 8 RUNNING** | ~8h |

Tracker file: `.pair_phase2_jids` (untracked, kept locally).

Pending tasks in the in-session task list:

- **#10** ~~Submit rope_shift second half~~ — done (237733)
- **#11** Submit aggregator — pending; waits on 236687 + 237733 both
  showing all-COMPLETED, then `sbatch
  scripts/marlowe/aggregate_pair.sh`. The aggregator script needs no
  changes (it globs all results.json files under
  `long-health/pair_experiment/<variant>/pair_*/`).

## Things to remember for the next pickup

1. **The phase-2 aggregator must be submitted manually** after both
   rope_shift batches finish. It would have been the 33rd job at
   submission time, exceeding the 32-job cap; deferring it was the
   only way to land all 32 task slots simultaneously. The cron loop is
   still running at the time of writing — when it sees both
   rope_shift jids COMPLETED, the next instance should fire the
   aggregator submission.

2. **The `Job dependency problem` failure mode is real**: slurm
   rejects `--dependency=afterok:<jid>` if the dependency target has
   completed and been purged from the active queue. Workaround is to
   drop the dependency entirely once the target is verifiably done
   (or use `--dependency=afterok:<jid>?` to allow degraded
   dependencies, but I didn't try that).

3. **Phase-1 results live in the same per-pair directories as phase-2
   results.** The phase-2 aggregator will load all 760 result files
   (84 phase-1 + 676 phase-2) without distinction — the phase-1
   results were generated under the round-2 schema, so the
   correctness-split `attn_mass_after_aggregate` is present in every
   file and the aggregator's `_gather_attn_after` works uniformly
   across phases.

4. **Verification after the aggregator runs:**
   - `find long-health/pair_experiment/naive -name results.json | wc -l` → **380**
   - `find long-health/pair_experiment/rope_shift -name results.json | wc -l` → **380**
   - `jq '.naive.by_first_position | length' long-health/pair_experiment/summary.json` → **20**
   - The figure should now have `n=` counts in the per-cell titles roughly 9× phase 1
   - Update `PAIR_EXPERIMENT_REPORT.md` § Phase 2 with the actual marginals

5. **Push is still pending.** Both commits (`b55434a`, `0911b9e`) live
   only on the Marlowe login node. `gh` is not installed and the SSH
   key is denied (memory `reference_marlowe_git_auth.md`). User needs
   to run `! git push` from their end.

## Pointers

- **Plan file for this session:** `~/.claude/plans/declarative-wandering-fox.md`
- **Phase-1 plan:** `~/.claude/plans/adaptive-stargazing-curry.md`
- **Long-lived data index:** `contexts/06042026/PAIR_EXPERIMENT_REPORT.md`
- **Round-2 schema spec:** `contexts/06042026/ATTENTION_MASS_SPEC.md`
- **Per-patient run that produced the 13 new caches:** `contexts/06042026/PER_PATIENT_RUN_SUMMARY.md`
- **RoPE shift rationale:** `contexts/06042026/ROPE_SHIFT_NOTE.md`
- **k=5 stacking probe (parallel work):** `contexts/07042026/K5_PROBE_PLAN.md`
- **Job tracker:** `.pair_phase2_jids` (untracked)
