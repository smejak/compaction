# Single k=5 Stacking Probe — Plan

## Context

The deep dive (`contexts/06042026/KARY_STACKING_DEEP_DIVE.md`, committed in
`ad9d625`) showed that GPU memory is **not** the binding constraint for k-ary
stacking up to k=7 — the binding constraint is **RoPE position
extrapolation** past Qwen3-4B's 40,960 native window. At k=5, the stacked
original sequence length is ~58k tokens, which is **+38% past the trained
window**, putting us in the regime where any accuracy degradation conflates
two effects: (a) more competing context, and (b) RoPE positions outside the
trained range.

Before committing to a multi-run sweep that would consume hundreds of GPU·h,
we want **one** k=5 run to answer two concrete questions:

1. **Does the existing stacking code generalize correctly from k=2 to k=5?**
   The pair experiment hardcodes a 2-region attention bucketing and a single
   RoPE shift. Generalizing to k requires changing both, and we need to
   validate the generalization works before scaling.
2. **At k=5, does accuracy drop noticeably vs the per-patient (k=1) baseline,
   and if so, *how* does it drop?** The "how" we care about most is the
   per-position breakdown (which slot in the stack — first, middle, last —
   suffers more) and the per-layer attention-mass distribution at the last
   query token.

This is explicitly **a probe**, not a study. Out of scope for this plan:
multi-permutation sweeps, hero-subset enumeration, full aggregation, both
variants. Those come later, in a follow-up plan that this probe will inform.

## Scope

**One run.** Patients, ordering, and variant are fixed:

- **Patients (in stack order)**: `patient_01, patient_03, patient_04, patient_05, patient_06`
  — the first 5 of the 7 already-pair-experimented patients, in alphabetical
  order. This makes the result trivially reproducible and matches the
  cumulative-prefix construction tabulated in `KARY_STACKING_DEEP_DIVE.md §7.1`.
- **Variant**: `rope_shift` only. Naive aliases positions, which would
  conflate position-aliasing with the stack-depth signal we're trying to
  isolate. Single probe should use the cleanest config.
- **Questions**: 100 total — 20 per patient × 5 patients — drawn from
  `evaluation/datasets.load_dataset("longhealth")`.
- **Expected wall time**: ~1 h on one H100 (~58 min from the deep dive's
  per-`k` table). Budget 2 h to leave margin.
- **Expected stacked context**: `stacked_original_seq_len = 57,976`,
  `max_layer_len ≈ 52,549`. Memory peak ~25 GB on H100 80 GB — comfortable.

The ordering choice (`01, 03, 04, 05, 06`) means: patient_01's content sits
at positions `[0, 12265)`, patient_06 at `[46688, 57976)`, and the question
is asked at position 57976 — well past the 40k window. Whichever patient is
**queried** for a given question lives at a different position in the stack
(positions 1..5), and we capture per-position accuracy in the output.

## Design decisions

### Why a new file (`scripts/run_kary_experiment.py`) instead of generalizing `run_pair_experiment.py`

The pair experiment is mid-flight (1 of 42 results on disk; see deep dive
§9.4) and has its own SLURM array workflow. Modifying it in place to take a
variable `k` would (a) risk breaking the pair workflow, (b) require a
byte-equivalence verification step against the existing pair result, and
(c) tangle two scopes. A standalone `run_kary_experiment.py` keeps the
probe self-contained and lets the pair experiment continue independently.

There **will** be code duplication between the two files (cache loading,
model loading, `attn_mass_before`, the eager forward path). This is OK for
the probe. The duplication can be cleaned up later by extracting a
`scripts/_stack_common.py` if the k-ary experiment grows into a sweep.

### Why rope_shift only (no naive)

The probe is asking "is k=5 viable and how does accuracy break?". Naive
concat introduces a known artifact (position aliasing — two patients'
keys at overlapping positions) that we already understand from the pair
experiment. Including naive doubles the run cost without changing the
answer to the probe's question. If the naive vs rope_shift contrast turns
out to matter at higher k, the user can run a naive variant separately as
a follow-up — the new code path supports both via `--variant`.

### Why this specific 5-patient subset

The first 5 of the alphabetically-sorted PATIENT_IDS list. Reasons:

- **Reproducible**: deterministic, no randomness, no hidden seed
- **Continuity with pair experiment**: patient_01 and patient_03 are the
  first pair the pair experiment ran (`pair_patient_01_patient_03`), so we
  have a sanity-check anchor at k=2 within this 5-tuple
- **Representative position range**: ~58k stacked, the canonical k=5 number
  from the deep dive — generalizes cleanly to "the average k=5 case"

### Memory plan

At k=5 with `batch_size=1` instrumented forward (per
`ATTENTION_MASS_SPEC §5`), the eager attention peak is ~9.4 GB across all
36 layers, plus model ~8 GB plus KV cache ~3 GB plus ~5 GB other ≈
**~25 GB total** on H100 80 GB. No need for the hook-based attention
escalation. No need to drop instrumentation.

Generation runs at `batch_size = max(1, min(20, int(25000 / max_layer_len)))`
which clamps to 1 at k=5 (since `max_layer_len ≈ 52549 > 25000`). So
generation is single-question, same as the pair experiment.

## Code changes

### New file: `scripts/run_kary_experiment.py` (~350 lines)

Mirrors `scripts/run_pair_experiment.py` structure but parameterized by a
list of patients instead of a pair index. Key generalizations from the pair
code:

#### `_stack_caches_kary(cache_list, variant, model, device, dtype)`

Replaces `_stack_caches`. Input: ordered list of `k` cache dicts. Logic:

```python
def _stack_caches_kary(cache_list, variant, model, device, dtype):
    num_layers = len(cache_list[0]["cache"])
    seq_lens = [int(c["original_seq_len"]) for c in cache_list]
    k = len(cache_list)

    # Cumulative offsets: cache i starts at sum of seq_lens of caches 0..i-1
    offsets = [0]
    for i in range(k - 1):
        offsets.append(offsets[-1] + seq_lens[i])

    # For rope_shift: precompute one (cos_diff, sin_diff) per cache (except cache 0).
    # compute_rope_correction is cheap; k-1 calls is fine.
    rope_corrections = [None]  # cache 0 stays at position 0
    if variant == "rope_shift":
        for i in range(1, k):
            cos_diff, sin_diff = compute_rope_correction(
                model,
                current_positions=torch.tensor([0], device=device),
                target_positions=torch.tensor([offsets[i]], device=device),
                device=device,
                dtype=dtype,
            )
            rope_corrections.append((cos_diff, sin_diff))

    # Per-cache, per-layer compacted lengths
    t_per_cache_per_layer = [[] for _ in range(k)]
    cache_cpu = []
    for layer_idx in range(num_layers):
        per_cache_C1 = []
        per_cache_beta = []
        per_cache_C2 = []
        for i in range(k):
            C1, beta, C2 = cache_list[i]["cache"][layer_idx]
            t_per_cache_per_layer[i].append(int(C1.shape[-2]))
            if variant == "rope_shift" and i > 0:
                C1_gpu = C1.to(device=device, dtype=dtype)
                C1_shifted = apply_rotary_pos_emb_to_cache(C1_gpu, *rope_corrections[i])
                C1 = C1_shifted.to(device="cpu", dtype=cache_list[0]["cache"][layer_idx][0].dtype)
                del C1_gpu, C1_shifted
            per_cache_C1.append(C1)
            per_cache_beta.append(beta)
            per_cache_C2.append(C2)
        # Concat along seq dim (-2)
        C1_cat = torch.cat(per_cache_C1, dim=-2)
        beta_cat = torch.cat(per_cache_beta, dim=-1)
        C2_cat = torch.cat(per_cache_C2, dim=-2)
        cache_cpu.append((C1_cat.contiguous(), beta_cat.contiguous(), C2_cat.contiguous()))

    stacked_seq_len = sum(seq_lens)
    return cache_cpu, stacked_seq_len, t_per_cache_per_layer
```

#### `_attn_mass_before_kary(cache_cpu, t_per_cache_per_layer)`

Replaces `_attn_mass_before`. Generalizes from 2-way [A,B] split to k-way
[c_0, c_1, …, c_{k-1}] split:

```python
def _attn_mass_before_kary(cache_cpu, t_per_cache_per_layer):
    k = len(t_per_cache_per_layer)
    per_layer = []
    for layer_idx, (_, beta, _) in enumerate(cache_cpu):
        w = torch.exp(beta.float())  # (1, KV, t_total)
        boundaries = [0]
        for i in range(k):
            boundaries.append(boundaries[-1] + t_per_cache_per_layer[i][layer_idx])
        layer_masses = []
        for i in range(k):
            layer_masses.append(float(w[..., boundaries[i]:boundaries[i+1]].sum()))
        total = sum(layer_masses)
        if total > 0:
            per_layer.append([m / total for m in layer_masses])
        else:
            per_layer.append([1.0 / k] * k)
    means = [sum(p[i] for p in per_layer) / len(per_layer) for i in range(k)]
    return {"per_layer": per_layer, "means": means}
```

#### `_init_agg_kary` / `_accumulate_agg_kary` / `_finalize_agg_kary`

Generalizes the aggregator from 2 position buckets to k. Each bucket holds
`cache_sums[i]` (length k list of `np.zeros(num_layers)` arrays) plus
`cache_sq_sums[i]` plus `Q_sum`/`Q_sq_sum`. Per question, the bucket for
the source patient's position is updated with k cache-region masses + the
question region mass. On finalize, each bucket emits per-layer
`{cache_means: [k floats], cache_stds: [k floats], Q_mean, Q_std}`.

#### `_run_instrumented_forward_single_kary`

Replaces `_run_instrumented_forward_single`. The only change from the pair
version is the bucketing loop at the end:

```python
# Replaces:
#   t_a = t_A_per_layer[layer_idx]; t_b = t_B_per_layer[layer_idx]
#   row = attn[0, :, q_last, :].float().mean(dim=0)
#   mass_a = float(row[:t_a].sum())
#   mass_b = float(row[t_a:t_a + t_b].sum())
#   mass_q = float(row[t_a + t_b:].sum())
#
# With:
boundaries = [0]
for i in range(k):
    boundaries.append(boundaries[-1] + t_per_cache_per_layer[i][layer_idx])
row = attn[0, :, q_last, :].float().mean(dim=0)
cache_masses = [float(row[boundaries[i]:boundaries[i+1]].sum()) for i in range(k)]
mass_q = float(row[boundaries[k]:].sum())
layers_kbq.append({"layer": layer_idx, "cache_masses": cache_masses, "Q": mass_q})
```

Everything else (tokenization, cache_gpu construction, model forward,
mutation-rebuild discipline) is byte-identical to the pair version. Cribbed
verbatim.

#### `run_kary(patients, variant, results_dir, caches_dir, model_name)`

The main orchestrator. Parallels `run_pair`:

```python
def run_kary(patients, variant, results_dir, caches_dir, model_name):
    # Idempotent skip
    subset_id = "_".join(p.replace("patient_", "") for p in patients)
    out_dir = os.path.join(results_dir, variant, f"k{len(patients)}_{subset_id}")
    result_path = os.path.join(out_dir, "results.json")
    if os.path.exists(result_path):
        print(f"skip k={len(patients)} {variant} {subset_id} (already done)")
        return

    # Load caches in stack order
    caches = [_load_cache(os.path.join(caches_dir, p, "cache.pt")) for p in patients]
    seq_lens = [int(c["original_seq_len"]) for c in caches]

    # Load model (eager attention required for output_attentions=True)
    model, tokenizer = _load_model_eager(model_name, "cuda")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Stack
    cache_cpu, stacked_seq_len, t_per_cache_per_layer = _stack_caches_kary(
        caches, variant, model, device, dtype
    )
    max_layer_len = max(c1.shape[-2] for c1, _, _ in cache_cpu)

    # attn_mass_before (pair-level, derived from beta)
    amb = _attn_mass_before_kary(cache_cpu, t_per_cache_per_layer)

    # Free raw caches
    del caches; gc.collect(); torch.cuda.empty_cache()

    # Load questions for each patient, annotate with position (1-indexed)
    data = load_dataset("longhealth")
    by_pid = {art["article_id"].replace("longhealth_", ""): art for art in data}
    questions = []
    for pos, pid in enumerate(patients, start=1):
        for q in by_pid[pid]["questions"]:
            questions.append(dict(q, patient=pid, position=pos))
    # Total: 20 * k questions

    # Batch size heuristic (will clamp to 1 at k>=2)
    batch_size = max(1, min(20, int(25000 / max_layer_len)))

    # Eval loop: per question, run instrumented forward at B=1 then real generation
    num_layers = len(cache_cpu)
    agg = _init_agg_kary(num_layers, k=len(patients))
    results = []
    for bs in range(0, len(questions), batch_size):
        be = min(bs + batch_size, len(questions))
        batch = questions[bs:be]
        prompts = [format_question(tokenizer, q["question"], q.get("options"), model_name) for q in batch]

        # 1. Instrumented forwards, one prompt at a time
        for q, prompt in zip(batch, prompts):
            layers_kbq = _run_instrumented_forward_single_kary(
                model, tokenizer, prompt, cache_cpu, stacked_seq_len,
                t_per_cache_per_layer, device, dtype,
            )
            _accumulate_agg_kary(agg, q["position"], layers_kbq)

        # 2. Real batched generation with fresh cache_gpu
        cache_gpu = _build_cache_gpu(cache_cpu, device, dtype)
        answers = generate_with_compacted_cache_batch(
            model, tokenizer, prompts, cache_gpu,
            max_new_tokens=2048, original_seq_len=stacked_seq_len,
        )
        del cache_gpu; torch.cuda.empty_cache()

        for q, ans in zip(batch, answers):
            mc = parse_model_choice(ans, max_options=len(q.get("options", [])))
            gold = q.get("gold_label")
            ok = (mc == gold) if mc and gold else False
            results.append({
                "qid": q["question_unique_id"],
                "patient": q["patient"],
                "position": q["position"],
                "correct": ok, "pred": mc, "gold": gold,
            })

    # Compute per-position accuracy
    k = len(patients)
    acc_per_position = []
    for pos in range(1, k + 1):
        rs = [r for r in results if r["position"] == pos]
        acc_per_position.append(sum(r["correct"] for r in rs) / len(rs) if rs else 0.0)
    overall = sum(r["correct"] for r in results) / len(results)

    # Build result JSON, save, print summary (see §Output schema below)
```

#### CLI

```python
parser.add_argument("--patients", type=str, required=True,
    help="Comma-separated list of patient IDs (e.g. patient_01,patient_03,...)")
parser.add_argument("--variant", type=str, required=True,
    choices=["naive", "rope_shift"])
parser.add_argument("--results-dir", default="long-health/kary_experiment")
parser.add_argument("--caches-dir", default="long-health")
parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
```

Patients are split on `,`, validated against the on-disk caches directory,
and passed in order to `run_kary`.

### New file: `scripts/marlowe/kary_single.sh` (~70 lines)

Single-job SLURM script (no array — one run only). Modeled byte-for-byte
on `scripts/marlowe/pair_experiment.sh` with these differences:

```bash
#!/bin/bash
#SBATCH --job-name=lh_kary5
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000120-pm05
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --error=logs/kary5_%x_%j.err
#SBATCH --output=logs/kary5_%x_%j.out

# Single k=5 stacking probe over patients 01,03,04,05,06.
# rope_shift variant only. See ~/.claude/plans/composed-singing-reef.md
# for the full plan and contexts/06042026/KARY_STACKING_DEEP_DIVE.md for
# the analysis that motivated this configuration.

set -euo pipefail

# -------- Environment --------
# Marlowe setup: call the env's python directly. Do NOT load the cudatoolkit
# or cudnn modules — they break torch CUDA init via LD_LIBRARY_PATH
# poisoning (cudaErrorSystemDriverMismatch / error 803). See
# contexts/06042026/PER_PATIENT_RUN_SUMMARY.md for the postmortem.
PY="${PY:-/users/jsmekal/.conda/envs/hard_drive/bin/python}"

cd "${SLURM_SUBMIT_DIR:-$HOME/compaction}"

export HF_HOME="${HF_HOME:-/projects/m000120/jsmekal/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p logs long-health/kary_experiment

# -------- Job info --------
start_time=$(date +%s)
echo "Start:   $(date -d @"$start_time")"
echo "Node:    $(hostname)"
echo "Job:     $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "CWD:     $PWD"
echo "Python:  $PY"
"$PY" --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -------- Run k=5 probe --------
"$PY" -u scripts/run_kary_experiment.py \
    --patients patient_01,patient_03,patient_04,patient_05,patient_06 \
    --variant rope_shift \
    --results-dir long-health/kary_experiment \
    --caches-dir long-health

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Elapsed: %dh %dm %ds\n' $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
```

**Compliance with Marlowe norms**: this is one job, well under the 8-job
threshold. No Slack notification needed. Per the
`feedback_marlowe_no_login_compute` memory, even small operations should
use sbatch — this script does that.

## Output schema

`long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json`:

```json
{
  "variant": "rope_shift",
  "k": 5,
  "patients": ["patient_01", "patient_03", "patient_04", "patient_05", "patient_06"],
  "seq_lens": [12265, 12831, 11517, 10075, 11288],
  "stacked_original_seq_len": 57976,
  "max_layer_len": 52549,
  "t_per_cache_per_layer": [
    [t_layer_0, t_layer_1, ..., t_layer_35],   // patient_01, length 36
    [t_layer_0, ..., t_layer_35],              // patient_03
    [t_layer_0, ..., t_layer_35],              // patient_04
    [t_layer_0, ..., t_layer_35],              // patient_05
    [t_layer_0, ..., t_layer_35]               // patient_06
  ],
  "attn_mass_before": {
    "per_layer": [[s0, s1, s2, s3, s4], ...],  // 36 entries, each length 5; sums to 1
    "means": [m0, m1, m2, m3, m4]              // length 5
  },
  "attn_mass_after_aggregate": {
    "position_1": {
      "n": 20,
      "per_layer": [
        {
          "layer": 0,
          "cache_means": [c0, c1, c2, c3, c4],
          "cache_stds": [s0, s1, s2, s3, s4],
          "Q_mean": q,
          "Q_std": q_std
        },
        ...
      ]
    },
    "position_2": { ... },
    "position_3": { ... },
    "position_4": { ... },
    "position_5": { ... }
  },
  "overall_accuracy": float,
  "correct": int,
  "total": 100,
  "acc_per_position": [a1, a2, a3, a4, a5],
  "model": "Qwen/Qwen3-4B",
  "per_question": [
    {"qid": str, "patient": str, "position": 1..5, "correct": bool, "pred": int, "gold": int},
    ...
  ]
}
```

The script prints a one-line summary at the end (mirroring the pair
experiment's final print):

```
k5 patient_01,patient_03,patient_04,patient_05,patient_06 (rope_shift):
  overall=68%  pos1=85%  pos2=80%  pos3=70%  pos4=60%  pos5=45%
  saved long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json
```

(Numbers are illustrative — the whole point of the probe is to find out
what they actually are.)

## Verification

### Pre-run sanity checks (login node, before sbatch)

1. **Run enumeration check**: import the new module and call its `run_kary`
   function with `--help` to verify the CLI parses. Done locally on the
   login node, no GPU needed:
   ```bash
   /users/jsmekal/.conda/envs/hard_drive/bin/python scripts/run_kary_experiment.py --help
   ```
2. **Cache file check**: confirm all 5 patient cache files exist:
   ```bash
   ls long-health/patient_{01,03,04,05,06}/cache.pt
   ```

### During-run monitoring

1. `squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"` to watch
   job progress
2. `tail -f logs/kary5_*.out` to follow output — the script prints progress
   per-batch and a per-question line, same as pair_experiment

### Post-run validation

1. **Output JSON exists**: `ls long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json`
2. **All 100 questions evaluated**: `total == 100`
3. **Attention mass invariant**: for each layer and position, the 5
   `cache_means` plus `Q_mean` should sum to ≈1.0 (modulo float drift). A
   one-liner check:
   ```python
   import json
   r = json.load(open("long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json"))
   for pos in range(1, 6):
       for entry in r["attn_mass_after_aggregate"][f"position_{pos}"]["per_layer"]:
           total = sum(entry["cache_means"]) + entry["Q_mean"]
           assert 0.95 < total < 1.05, f"pos {pos} layer {entry['layer']}: total={total}"
   ```
4. **Per-position accuracies in [0, 1]**: `0 <= acc <= 1` for all 5

### Comparison against baselines

After the probe completes, compare against existing data:

| comparison | source | what to expect |
|---|---|---|
| k=1 baselines (per-patient) | `long-health/patient_{01,03,04,05,06}/results.json` `accuracy` field | Mean of these 5 ≈ 85 % (from PER_PATIENT_RUN_SUMMARY: patient_01=80, 03=90, 04=90, 05=85, 06=80 → mean 85 %) |
| k=2 anchor (one pair we have) | `long-health/pair_experiment/{naive,rope_shift}/pair_patient_01_patient_03/results.json` | Comparable to the k=2 prefix `[01, 03]` of our k=5 stack at positions 1,2 |
| k=5 probe overall | this run | unknown — that's what we're measuring |
| k=5 probe per-position | this run | the position-effect curve we're chasing |

Two specific things to look for in the k=5 result:

1. **Overall accuracy drop vs k=1 baseline**: how much worse is k=5 stacked
   than the simple average of single-patient accuracies? A drop of ≤ 5 pts
   would suggest k=5 is "fine" and we can extend the sweep upward; a drop
   of ≥ 20 pts would suggest position extrapolation is hurting badly and we
   should stick to k ≤ 3 for any further work.
2. **Position-bias signal**: does `acc_per_position` show monotonic
   degradation (e.g., pos1 highest, pos5 lowest)? Or U-shaped ("lost in
   the middle")? Or flat? Each pattern points at a different mechanism.

## Execution workflow

```bash
# 0. Smoke checks (login node, no GPU)
ls long-health/patient_{01,03,04,05,06}/cache.pt   # all 5 files exist
/users/jsmekal/.conda/envs/hard_drive/bin/python scripts/run_kary_experiment.py --help

# 1. Submit the k=5 probe
sbatch scripts/marlowe/kary_single.sh
# Note the job ID returned. Wall: ~1 h expected, 2 h budgeted.

# 2. Monitor
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
tail -f logs/kary5_lh_kary5_<jobid>.out

# 3. Verify completion
ls -la long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json

# 4. Inspect results manually
/users/jsmekal/.conda/envs/hard_drive/bin/python -c "
import json
r = json.load(open('long-health/kary_experiment/rope_shift/k5_01_03_04_05_06/results.json'))
print(f'Overall: {r[\"overall_accuracy\"]:.0%} ({r[\"correct\"]}/{r[\"total\"]})')
print('Per-position accuracy:')
for i, a in enumerate(r['acc_per_position'], 1):
    print(f'  position {i} ({r[\"patients\"][i-1]}): {a:.0%}')
print()
print('attn_mass_before (averaged over layers):')
for i, m in enumerate(r['attn_mass_before']['means'], 1):
    print(f'  cache {i}: {m:.3f}')
"
```

## Critical files to create

- **Create**: `scripts/run_kary_experiment.py` (~350 lines, see §Code changes)
- **Create**: `scripts/marlowe/kary_single.sh` (~70 lines, see §Code changes)

**No modifications to existing files.** The pair experiment, the per-patient
experiment, and all shared infrastructure (`models/cache.py`,
`compaction/compaction_methods/chunked.py`, `evaluation/utils.py`,
`models/generate.py`) are reused unchanged.

## Reused code (no modifications needed)

- `compaction/compaction_methods/chunked.py:compute_rope_correction` — called
  k-1 times for the rope_shift variant
- `compaction/compaction_methods/chunked.py:apply_rotary_pos_emb_to_cache`
  — applies each correction
- `models/cache.py:CompactedPrefixCache` — accepts the per-layer concat
  tuple and the summed `original_seq_len`, computes `rope_base` correctly
  for the stacked context
- `models/generate.py:generate_with_compacted_cache_batch` — single-question
  batched generation, unchanged
- `evaluation/utils.py:format_question, parse_model_choice` — unchanged
- `evaluation/datasets.py:load_dataset("longhealth")` — unchanged

## Future expansion (not part of this plan, but enabled by it)

This single-run scaffold trivially extends to a sweep when we're ready:

1. **Multi-permutation sweep**: add a `_enumerate_runs()` helper that builds
   a deterministic list of `(patients_tuple, variant)` entries, change
   `--patients` to `--run-idx`, and convert `kary_single.sh` to a SLURM
   array script with `--array=0-N%8`.
2. **Both variants**: trivially supported via the existing `--variant` flag.
3. **Different k values**: trivially supported via `--patients` (any length).
4. **Aggregator**: a new `scripts/aggregate_kary_results.py` would walk the
   `kary_experiment/{variant}/k*/` directories, the same way
   `aggregate_pair_results.py` walks `pair_experiment/{variant}/pair_*/`.

The follow-up plan can mention all of this, but the probe itself ships
without any of it.

## What success looks like

A successful k=5 probe answers:

1. **Code generalization**: the k-ary stacking code runs end-to-end at k=5
   with no OOM, no shape errors, and produces a `results.json` matching the
   schema in §Output schema. The attention-mass invariant
   (`sum(cache_means) + Q_mean ≈ 1`) holds for every layer × position. Pass.
2. **Accuracy measurement**: we have a concrete `overall_accuracy` number
   for k=5 stacked rope_shift over patient_01..06, plus per-position
   breakdown.
3. **Decision input for the next plan**: the size of the accuracy drop vs
   k=1 baselines and the shape of the position-bias curve give us enough
   to decide whether to (a) commit to a k≤3 sweep, (b) commit to a full
   k≤7 sweep, or (c) invest in a YaRN-enabled run before scaling further.
