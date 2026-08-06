# A5 Support-Aligned Query Supervision and Outer-TTT Stabilization

## Executive summary

This change set closes two coupled A5 training problems:

1. Query supervision was too weakly aligned with the Support updates that it
   was intended to supervise. In the old execution order, most Support chunks
   were consumed before a final Query, so only the most recent bounded
   second-order window received a strong causal signal.
2. The raw temporal target scale drifted during A5. The same raw TTT loss was
   used both for episode-local Inner SGD and as a persistent outer auxiliary,
   so target-scale growth could dominate the long-term optimizer even when the
   fast-weight update itself remained numerically valid.

The new A5 path:

- represents an episode as causally valid Support segments followed by one or
  more real Query labels;
- keeps all 3,055 official training Queries as weight-1 meta supervision;
- executes no zero-Support or diagnostic-only Query in the training graph;
- bounds each retained second-order graph to at most eight Support updates;
- keeps the raw TTT objective unchanged for Inner SGD;
- converts only the outer temporal prediction auxiliary to a detached
  target-relative MSE;
- records Query gradients, deferred VJPs, fast-weight updates, target/prediction
  scales, parameter deltas, and per-rank execution evidence;
- destroys the distributed process group on trainer exit; and
- supports atomic-final-only production checkpoint publication.

A real four-H200 production run reached optimizer step 129 with 129 successful
updates, zero skipped or non-finite updates, no OOM/NCCL/traceback, and no
premature checkpoint. Over the first and most recent 32-step windows, effective
outer TTT fell by 79.8% and temporal error RMS fell by 45.5%.

## Scope

This report covers the complete branch relative to upstream `main`, including:

- Support-aligned A5 manifest schema and episode validation;
- dense Query bundles after each causal Support segment;
- cache identity changes required by supervised segment boundaries;
- dynamic-graph-safe ZeRO-1 execution;
- distributed process-group cleanup;
- smoke and atomic-final checkpoint policies;
- explicit outer-only segment auditing;
- raw temporal-scale and optimizer-parameter-delta auditing; and
- the outer-only TTT scale fix.

It does not change:

- the A2 checkpoint used to initialize A5;
- the raw Inner-SGD TTT objective or fast optimizer hyperparameters;
- Query weights (all retained official Queries remain weight `1.0`);
- the `0.1` A5 support auxiliary coefficient;
- the maximum Support depth `K=8`;
- the state architecture, Bank/FSM, Operator/Task definitions, or Retrieval MIL
  semantics; or
- the Answer Query video path.

## Data and execution redesign

### Previous behavior

The previous manifest could execute many Support chunks before a single final
Query. A5 retained only a bounded second-order graph, so a final Query mainly
supervised the most recent Support window. Query-only rows could also dilute
the signal without providing a real Support-to-Query transition.

### New supervised segments

Each real episode is represented as ordered supervised segments:

```text
Episode
├─ Segment 0
│  ├─ 1..8 Support chunks
│  └─ 1..N official Query labels, each with weight 1.0
└─ Segment 1 (when causally representable)
   ├─ 1..8 Support chunks
   └─ 1..N official Query labels, including the final Query
```

The manifest builder sorts official Queries, splits only at causally
representable gaps, and validates:

```text
previous Query time < Support end time < current Query time
```

When an update cannot be represented safely (for example,
`insufficient_time`), the segment is explicit `outer_only`; it is not silently
reported as a successful fast update.

### Current train manifest

The v4 production manifest contains:

| Item | Count |
|---|---:|
| Real episodes | 865 |
| Support chunks | 7,583 |
| Meta Queries | 3,055 |
| Diagnostic Queries executed during training | 0 |
| Supervised segments | 1,666 |
| Zero-Support Query rows | 0 |

The corresponding FP16 `support + state_query` cache contains 11,003 valid
entries, with zero missing and zero corrupt entries. The verified on-disk size
is 164,591,777,955 bytes (about 153.3 GiB). Answer Query inputs remain
uncached.

## Inner TTT, Query meta-gradient, and Outer TTT

### Inner TTT

For each Support observation, the raw TTT objective is:

\[
L_{\mathrm{TTT,inner}}
=
L_{\mathrm{pred}}
+0.5L_{\mathrm{identity}}
+0.5L_{\mathrm{event}}.
\]

It performs episode-local functional SGD:

\[
\theta_{t+1}
=
\theta_t-\alpha\nabla_{\theta_t}L_{\mathrm{TTT,inner},t}.
\]

These fast weights are transient episode state. They determine the state read
by subsequent Queries but are not independently published as model
checkpoints.

### Query meta-gradient

A Query evaluates the adapted fast state:

\[
L_Q(\theta_T).
\]

Its gradient is captured at the adapted state, then a deferred vector-Jacobian
product propagates that signal through the bounded Support update graph. This
is the actual meta-learning path:

\[
L_Q
\rightarrow
\theta_T
\rightarrow
L_{\mathrm{TTT,inner}}
\rightarrow
\theta_0.
\]

Every retained Query has weight `1.0`. Query losses are summed rather than
divided by the number of Queries, so dense supervision does not become weaker
when an episode contains more official Query labels.

### Outer TTT auxiliary

The persistent outer optimizer also receives a support-side auxiliary:

\[
L_{\mathrm{A5}}
=
\sum_q L_Q
+0.1\,\operatorname{mean}_tL_{\mathrm{TTT,outer},t}.
\]

Before this fix, `L_TTT,outer` reused the raw Inner-SGD loss. Consequently, a
larger hidden-state target RMS increased the persistent outer contribution even
when it did not represent worse relative prediction.

The fixed outer prediction scale is:

\[
s
=
\frac{1}
{\max(\operatorname{mean}(y^2),1)}.
\]

The outer auxiliary is:

\[
L_{\mathrm{TTT,outer}}
=
s\,L_{\mathrm{pred}}
+0.5L_{\mathrm{identity}}
+0.5L_{\mathrm{event}}.
\]

Important properties:

- `s` is detached, so the model cannot manipulate the denominator;
- large targets use relative MSE;
- targets below RMS 1 are never amplified;
- the raw Inner-SGD loss and its gradients are unchanged;
- identity and event weights remain frozen at `0.5`; and
- the public A5 support coefficient remains `0.1`.

The identity cosine loss is clamped at zero to prevent tiny floating-point
roundoff from producing a negative consistency loss for identical normalized
vectors.

## Root-cause evidence

The pre-fix real diagnostic run showed:

| Temporal audit | Early | Late |
|---|---:|---:|
| Hidden RMS | 6.92 | 13.04 |
| Target RMS | 6.96 | 13.03 |
| Prediction RMS | 0.187 | 5.145 |
| Error RMS | 6.94 | 8.17 |
| Raw TTT | 41.05 | 61.26 |

Predictor, W0, and shared-state parameters were updating, so this was not a
disconnected graph or a no-update failure. The raw target scale itself nearly
doubled and directly inflated the outer auxiliary. Normalizing the Inner loss
was tested and rejected because it changed fast-weight update geometry and
made the synthetic functional-SGD acceptance non-finite. The accepted fix
therefore changes only the outer auxiliary.

## Implementation details

### Losses

- `compute_ttt_loss(...)` remains the sole source of the raw Inner-SGD loss.
- `compute_ttt_outer_auxiliary_loss(...)` derives a detached relative
  prediction scale from the temporal audit.
- empty temporal pairs preserve differentiable zero behavior.
- identity cosine roundoff is clamped to a non-negative loss.

### Meta trainer

For every Support chunk, the runner stores both:

- the raw `TTTLossOutput`, used by functional Inner SGD; and
- the effective outer auxiliary scalar, used only by the persistent optimizer.

Segment outer backward uses the effective auxiliary plus the deferred Query
VJP. Audit aggregation reports both raw and effective TTT means.

### Runtime diagnostics

The trainer records:

- `a5/ttt/raw_mean`;
- `a5/ttt/outer_effective_mean`;
- hidden, target, prediction, and error RMS/max-absolute values;
- Intermediate and Final Query proxy gradient norms;
- per-segment deferred VJP norm;
- fast-version delta, update count, skip count, and skip reason;
- Predictor, W0, and shared-state parameter deltas;
- meta-TTT versus outer-only segment counts;
- Query role, weight, and bundle size; and
- per-step optimizer/training wall time.

Owner/trajectory identifiers in runtime traces are serialized into stable JSON
values so per-rank traces remain parseable.

## Validation

### Local tests

The expanded A5-focused suite passed:

```text
149 passed
```

It covers:

- outer normalization for large targets;
- no amplification below the RMS floor;
- detached scale gradients;
- raw Inner-SGD semantics remaining unchanged;
- segment/query causal validation;
- dense Query bundle weights;
- deferred VJP and parameter updates;
- outer-only auditing; and
- dynamic-graph distributed execution invariants.

Ruff and Python compilation checks passed. A CUDA/BF16 temporal-loss smoke also
produced finite raw and effective losses.

The full local suite reaches an existing CPU tolerance failure at
`tests/test_state_encoder.py::test_batch_matches_independent_rows_and_runtime_storage_is_isolated`
after 459 passes and one optional CUDA skip. The same isolated test fails from
upstream `main` in the same environment, and neither the test nor
`state_encoder.py` is changed by this branch. It is therefore reported as a
baseline issue rather than silently widened into this A5 change.

### Four-H200 64-step smoke

Run:

```text
runs/0727_002552_a5_v4_outerttt_relmse_smoke64
```

Results:

| Item | Result |
|---|---:|
| Attempted / successful updates | 64 / 64 |
| Skipped / non-finite updates | 0 / 0 |
| Optimizer step mean | 15.25 s |
| Optimizer step P50 | 14.05 s |
| Optimizer step P95 | 33.82 s |
| Observed peak memory | about 135,357 MiB/GPU |
| OOM / NCCL / traceback | 0 / 0 / 0 |

All four ranks recorded equal counts for released segments, TTT numerical
audits, backward calls, optimizer calls, and outer loss/gradient collectives.

### Production 4-epoch run: step-129 snapshot

Run:

```text
runs/0727_004943_a5_dense_querybundle_v4_outerttt_relmse_4epoch_finalonly
```

Configuration:

- four H200 GPUs;
- 1,348 optimizer steps over four epochs;
- initialization from the completed A2 epoch-4 checkpoint;
- v4 dense Query-bundle manifest and verified FP16 cache;
- `atomic_final_only` checkpoint policy; and
- no smoke step limit.

Health at step 129:

| Item | Result |
|---|---:|
| Attempted / successful updates | 129 / 129 |
| Skipped / non-finite updates | 0 / 0 |
| Intermediate Query proxy gradient | finite, non-zero |
| Final Query proxy gradient | finite, non-zero |
| OOM / NCCL / traceback | 0 / 0 / 0 |
| Premature checkpoint directories | 0 |
| 128-step speed mean / P50 / P95 | 16.22 / 13.84 / 37.03 s |

First 32 versus most recent 32 steps:

| Metric | First 32 | Recent 32 | Change |
|---|---:|---:|---:|
| Effective outer TTT | 0.889 | 0.180 | -79.8% |
| `0.1 ×` support contribution | 0.0859 | 0.0163 | -81.0% |
| Temporal error RMS | 6.79 | 3.70 | -45.5% |
| Temporal prediction RMS | 0.21 | 6.14 | moving toward target |
| Temporal target RMS | 6.85 | 8.43 | +23.0% |
| Answer loss median | 0.0689 | 0.0205 | -70.2% |
| Task raw mean | 0.561 | 0.517 | -7.7% |
| Operator raw mean | 1.309 | 1.253 | -4.3% |
| Retrieval raw mean, valid bags only | 1.213 | 1.232 | +1.6% |
| Time raw median | 0.0078 | 0.0044 | -43.8% |

The TTT-specific evidence is consistent with real learning: the prediction
scale rises toward the target while temporal error and effective outer loss
fall. Answer median improves, but Answer/Query means remain dominated by a
small number of hard natural samples. Task and Operator show only early,
modest improvement; Retrieval is not yet converged.

## Remaining risks and review points

1. **Outer-only segments remain explicit.** At the earlier 64-step acceptance,
   15 of 120 segment executions per rank were `outer_only`, all explained by
   `insufficient_time`. Dense Query supervision is preserved, but these rows do
   not claim a fast update.
2. **Query loss has a long tail.** Median Answer/Query losses improve while
   means can rise because the natural sampler contains uneven Query bundle
   difficulty. A same-key evaluation is required before attributing quality
   gains.
3. **Retrieval is conditional.** Only rows with a valid positive/negative bag
   contribute Retrieval loss. Its short-run raw mean is currently flat.
4. **Group guard is frequently active.** In the recent 32-step window, the
   state-to-reference ratio is capped at `0.4`, with mean guard about `0.57`.
   This is no longer the near-zero starvation failure, but it should remain
   visible during the full run.
5. **Four data rows exceed video duration.** They are rejected by preflight
   rather than entering training with fabricated temporal labels.
6. **The full four-epoch result is not yet available.** The production run
   continues, and only its final checkpoint will be atomically published.

## Reviewer checklist

- Confirm that target normalization is restricted to the outer auxiliary.
- Confirm that the raw `TTTLossOutput` still drives functional Inner SGD.
- Confirm that Query labels never enter Support/Bank/FSM forward computation.
- Confirm that every retained Query weight is exactly `1.0`.
- Confirm that outer-only segments are explicit and do not increment fast
  version.
- Confirm that the cache identity includes supervised segment boundaries.
- Confirm that all ranks execute the same collective/backward shape.
- Confirm that `atomic_final_only` suppresses periodic model checkpoints but
  still publishes the completed four-epoch result.
