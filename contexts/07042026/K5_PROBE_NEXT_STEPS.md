# k=5 Stacking Probe — Next Steps (2026-04-07)

Companion to `K5_PROBE_PLAN.md` (the design doc) and
`PAIR_EXPERIMENT_PHASE1_CLOSEOUT_AND_PHASE2_LAUNCH.md` (the parallel,
unrelated workstream still running on the cluster). This document lists
the actionable next steps for the k=5 probe and the decision tree for
what to do once results are in.

## Status as of writing

Implementation finished, **not yet committed, not yet submitted**.

| artifact | path | state |
|---|---|---|
| Plan | `contexts/07042026/K5_PROBE_PLAN.md` | committed `54bb92f` |
| Probe script | `scripts/run_kary_experiment.py` | written, untracked |
| SLURM wrapper | `scripts/marlowe/kary_single.sh` | written, untracked, executable |
| `--help` smoke check | n/a | passes on login node |
| Module import smoke check | n/a | passes (all helpers present) |
| Bash syntax check | n/a | passes (`bash -n`) |
| Cache files | `long-health/patient_{01,03,04,05,06}/cache.pt` | all present |

The phase 2 pair experiment is **still running** in parallel
(`PAIR_EXPERIMENT_PHASE1_CLOSEOUT_AND_PHASE2_LAUNCH.md`). The probe is a
single SLURM job that lives at most 1 / 32 of the account submit
ceiling, so it does not interfere with the in-flight pair workload.
Different output directory (`long-health/kary_experiment/` vs
`long-health/pair_experiment/`); no path collisions.

## Step 1 — Commit the two new scripts

Stage and commit only `scripts/run_kary_experiment.py` and
`scripts/marlowe/kary_single.sh`. Do **not** stage the long-health
result directories or the `.pair_*_jids` files — those belong to
the still-running pair experiment.

```bash
git add scripts/run_kary_experiment.py scripts/marlowe/kary_single.sh
git commit -m "Add k=5 stacking probe script and SLURM wrapper"
```

Suggested commit body: "k-ary generalization of run_pair_experiment.py;
single-job SLURM wrapper for the canonical k=5 rope_shift run over
patients 01,03,04,05,06. See contexts/07042026/K5_PROBE_PLAN.md."

## Step 2 — Submit the probe

```bash
sbatch scripts/marlowe/kary_single.sh
```

Capture the returned job ID. Wall budget is 2 h (expected ~1 h). Job
will write to `long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json`
and to `logs/kary5_lh_kary5_<jobid>.{out,err}`.

**SLURM submit-limit sanity check** before sbatching: the
marlowe-m000120-pm05 account has a 32-job ceiling in medium QOS. The
phase 2 pair experiment is at ~32 jobs at peak. If `squeue -u $USER`
shows you near the ceiling, wait for one of the rope_shift array tasks
to finish before submitting the kary job. The probe is non-urgent —
nothing depends on it landing immediately.

## Step 3 — Monitor

```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
tail -f logs/kary5_lh_kary5_<jobid>.out
```

Expected log signal during the run:

- After ~5 min: model loaded, stacking complete, `attn_mass_before` line
  printed with 5 cache shares.
- For ~50 min: 100 lines like `Q<n>: ok|x  pred=... gold=... [patient_NN posM]`
  (one per question — instrumented forward + generation, ~30 s/question
  at this stacked length).
- Final block: `k5 patient_01,patient_03,...,patient_06 (rope_shift):`
  with overall + per-position accuracies and the saved-path line.

If the wall clock blows past 90 min without per-question lines flowing,
something is wrong (likely OOM or a CUDA driver mismatch — check the
`.err` file for `cudaErrorSystemDriverMismatch` / error 803, see
`contexts/06042026/PER_PATIENT_RUN_SUMMARY.md` for that postmortem).

## Step 4 — Post-run validation

```bash
ls -la long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json
```

Then run the inline schema + invariant checker (also in
`K5_PROBE_PLAN.md §Verification §Post-run validation`):

```bash
/users/jsmekal/.conda/envs/hard_drive/bin/python -c "
import json
r = json.load(open('long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json'))
assert r['total'] == 100, f'expected 100 questions, got {r[\"total\"]}'
for pos in range(1, 6):
    bucket = r['attn_mass_after_aggregate'][f'position_{pos}']
    assert bucket['n'] == 20, f'pos {pos} n={bucket[\"n\"]}'
    for entry in bucket['per_layer']:
        total = sum(entry['cache_means']) + entry['Q_mean']
        assert 0.95 < total < 1.05, (
            f'pos {pos} layer {entry[\"layer\"]}: cache+Q={total}'
        )
for a in r['acc_per_position']:
    assert 0.0 <= a <= 1.0
print('schema OK; all invariants hold')
print(f'overall: {r[\"overall_accuracy\"]:.0%}  per-pos: {r[\"acc_per_position\"]}')
"
```

## Step 5 — Compare against baselines

Pull the k=1 baseline accuracies for the 5 patients:

```bash
for p in patient_01 patient_03 patient_04 patient_05 patient_06; do
  acc=$(/users/jsmekal/.conda/envs/hard_drive/bin/python -c "
import json; print(json.load(open('long-health/$p/results.json'))['accuracy'])
")
  printf "  %-12s k=1 acc=%s\n" "$p" "$acc"
done
```

From `PER_PATIENT_RUN_SUMMARY.md`: 80, 90, 90, 85, 80 → mean **85 %**.

Pull the k=2 anchor (`pair_patient_01_patient_03`, both variants):

```bash
for v in naive rope_shift; do
  /users/jsmekal/.conda/envs/hard_drive/bin/python -c "
import json
r = json.load(open('long-health/pair_experiment/$v/pair_patient_01_patient_03/results.json'))
print(f'$v: overall={r[\"overall_accuracy\"]:.0%} pos1={r[\"acc_pos1\"]:.0%} pos2={r[\"acc_pos2\"]:.0%}')
"
done
```

Compute and write down:

| metric | k=1 baseline (mean of 5) | k=2 anchor (rope_shift, 01→03) | k=5 probe |
|---|---|---|---|
| overall accuracy | 85 % | (fill from above) | (fill from probe) |
| acc on patient_01 | 80 % | (acc_pos1) | (acc_per_position[0]) |
| acc on patient_03 | 90 % | (acc_pos2) | (acc_per_position[1]) |

## Step 6 — Decision tree

Two numbers determine what comes next: the **overall drop** vs k=1
baseline, and the **shape** of `acc_per_position`.

### Branch A: overall drop ≤ 5 pp

Position extrapolation past the 40k window is **not** ruining things.
k=5 is "fine". Next plan should be a multi-permutation k=5 sweep (e.g.
12 random orderings of 5 from the 20 patients × 2 variants × 240
questions each = 5760 evals — feasible inside a SLURM array on the same
account ceiling). Goal: estimate variance over orderings and check
whether the position-bias shape generalizes.

Also worth: a single k=7 probe (cumulative-prefix subset of the same
patients) to find out where the cliff actually is.

### Branch B: overall drop 5–20 pp

Mixed signal. Look at `acc_per_position`:

- **Monotonic decay** (pos1 highest, pos5 lowest, smooth in between) →
  classic position-extrapolation hurt. Worth running the same probe
  with a **YaRN-enabled** model variant to disentangle "more competing
  context" from "RoPE positions out of trained range". A YaRN run is
  the cleanest experiment to settle the question.
- **U-shape** ("lost in the middle") → not extrapolation, but a
  long-context attention pathology that has been seen in the
  literature. Pursue a different intervention (e.g., position-specific
  attention boost, or query rewriting that puts the relevant slot at
  the edge).
- **Flat** → the probe didn't actually find a position effect; the
  drop is uniform across stack positions, which would be surprising
  and worth a second probe at a different patient subset before
  drawing conclusions.

### Branch C: overall drop ≥ 20 pp

k=5 is too aggressive. Pivot to a **k≤3 sweep** as the next experiment
(much smaller per-run cost, all positions inside the 40k window for
typical patients). The k=5 result still has analytical value — it
quantifies how far past the cliff we already are.

## Step 7 — Write up

Whichever branch we land in, the next deliverable is:

1. A short results note in `contexts/07042026/` (or
   `contexts/<next-day>/` if it spills over): `K5_PROBE_RESULTS.md`,
   covering: overall accuracy + per-position breakdown, comparison
   against k=1 + k=2 baselines, observed position-bias shape, and the
   chosen branch with one-line rationale.
2. A new plan file (`K_SWEEP_PLAN.md` or similar) for whichever
   follow-up the decision tree picks.

Both should land before any further compute is committed.

## Things explicitly NOT in scope of the next steps

- **Touching the running pair experiment.** Phase 2 is in flight; do
  not interrupt, restart, or amend its scripts. The probe runs in a
  separate output directory and shares only the SLURM account.
- **Running naive variant.** The plan was deliberate about rope_shift
  only (see `K5_PROBE_PLAN.md §Why rope_shift only`). A naive run is a
  follow-up if and only if the rope_shift result is ambiguous about
  the position-aliasing mechanism — and even then, it's downstream of
  the decision tree above.
- **Refactoring the duplicated helpers.** `_load_cache`,
  `_load_model_eager`, `_build_cache_gpu`, etc. are intentionally
  duplicated between `run_pair_experiment.py` and
  `run_kary_experiment.py` (see `K5_PROBE_PLAN.md §Why a new file`).
  Extract a `scripts/_stack_common.py` only when a third caller exists.

## Open questions

These don't block submission — they're just things worth thinking about
before the results land:

1. **Does the per-question logging output overwhelm the SLURM log file?**
   100 questions × ~3 lines each + the long startup banner ≈ 400 lines.
   Fine. (The pair experiment writes ~50 % more than this per task and
   has been OK.)
2. **Does the aggregator's std estimate make sense at n=20 per bucket?**
   Population std with n=20 is fine for a probe; the per-question raw
   triples are *not* stored (aggregate-only telemetry per the
   `feedback_attention_telemetry_aggregate` memory), so we cannot
   recompute later. If the std turns out to be the most interesting
   number we want from this run, we'd need to flip a switch in the
   k-ary script for any follow-up sweep — note this in the results
   writeup.
3. **What's the right next probe if Branch B + monotonic?** YaRN
   requires either a different checkpoint or a config override on the
   existing Qwen3-4B. Worth scoping that before we get there so the
   decision branch isn't blocked by "we don't know how to enable YaRN".
