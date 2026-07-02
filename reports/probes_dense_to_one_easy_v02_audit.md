# probes_dense_to_one_easy_v02 audit

Audit date: 2026-07-02.

Scope: `probes_dense_to_one_easy_v02` dense/random density sweep and the later
loss-curve report. This audit covers the generated datasets, configs, runner,
mask/memory implementation, final-eval outputs, aggregation, and report wording.

## Verdict

The experiment is not valid as an evaluation of generalization or as evidence
for sparse random attention/memory performance on the named tasks.

The most serious issue is data contamination by construction: for all three
tasks, the final `test.jsonl` contains the same semantic examples as
`train.jsonl`. IDs and split metadata differ, but the model inputs and targets
overlap 100%. As a result, reported "test" accuracy is best interpreted as
performance on a memorized finite calibration set, not held-out generalization.

There was no evidence that the latest plotting/reporting commit modified data.
The small contaminated datasets were created earlier by
`3dcfd0c probes-dense-to-one-easy-v02-calibration`.

## Critical Findings

### F0. Train/test semantic overlap is 100%

Evidence:

- Dataset generator: `scripts/create_probe_dense_to_one_easy.py`.
- Data root: `datasets/probes_dense_to_one_easy_v02/`.
- The generator writes train/test splits separately, but test is just a reversed
  enumeration of the same finite language.

Measured by comparing `{task, input, target}` while ignoring `id` and split
metadata:

| task | train rows | train unique semantic rows | test rows | test unique semantic rows | unique semantic overlap | overlap fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selective_copy | 256 | 256 | 256 | 256 | 256 | 100% |
| induction_associative_recall | 256 | 24 | 256 | 24 | 24 | 100% |
| lra_listops | 400 | 400 | 400 | 400 | 400 | 100% |

Full JSON rows do not overlap because `id` and metadata differ, but that does
not matter for model evaluation. The supervised examples are the same.

Impact:

- All final/test metrics in `outputs/probes_dense_to_one_easy_v02/runs/**`
  are contaminated.
- Tables in `reports/probes_dense_to_one_easy_v02_report.md` and
  `reports/probes_dense_to_one_easy_v02_density_sweep.md` should not be read as
  held-out test performance.
- Dense reaching 1.0 is expected and not informative.

### F1. The datasets are tiny finite calibration sets

Hard-coded row counts:

```python
ROWS_BY_TASK = {
    "selective_copy": 256,
    "induction_associative_recall": 256,
    "lra_listops": 400,
}
```

The dataset cards explicitly say:

`finite_language_coverage_train_and_test_rows_for_dense_to_one_calibration`

Task-specific construction:

- `selective_copy`: four fixed source positions `[4, 18, 32, 46]`; each target
  value is from `[1,2,3,4]`; total language size is `4^4 = 256`; train and test
  both enumerate all 256.
- `induction_associative_recall`: fixed keys `[10,11,12,13]`, fixed values
  `[30,31,32,33]`, fixed key/value positions, and only 24 query permutations.
  Train/test repeat those 24 permutations up to 256 rows.
- `lra_listops`: depth-1, two-argument expressions only:
  `4 ops * 10 * 10 = 400`; train/test both enumerate all 400.

Impact:

- The task names are misleading. `lra_listops` is not an LRA ListOps
  generalization benchmark; it is a depth-1 finite lookup table.
- `induction_associative_recall` does not require true in-context association:
  the key-to-value mapping is fixed globally, so a model can learn a static
  token mapping.

### F2. The run repeatedly trains over the same finite examples

Configured profile:

- `epochs = 120`
- `steps = 3000` as an upper bound
- effective batch size `16`

Actual completed steps:

| task | train rows | effective batch | epochs | completed steps | total train example presentations | average presentations per unique semantic example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selective_copy | 256 | 16 | 120 | 1920 | 30,720 | 120 |
| induction_associative_recall | 256 | 16 | 120 | 1920 | 30,720 | 1,280 |
| lra_listops | 400 | 16 | 120 | 3000 | 48,000 | 120 |

`induction_associative_recall` is especially severe: only 24 unique semantic
examples are repeated 256 rows per epoch, then repeated for 120 epochs.

Impact:

- The experiment is dominated by memorization pressure.
- Threshold conclusions like "density where random drops below 0.95" are
  thresholds for memorizing this finite set, not for task capability.

### F3. The requested multiscale memory was not run

The density sweep configs use:

```json
"random_memory": {
  "enabled": true,
  "source": "input",
  "update": "lazy",
  "lazy_alpha": 0.5,
  "scale": 2.0,
  "steps": 1,
  "head_merge": "mean",
  "weight_mode": "soft",
  "edge_scope": "all"
}
```

The instantiated model confirms:

- `random_regular`: `rollout_memory=False`, `multiscale_steps=[1]`,
  `multiscale_weights=[1.0]`.
- `random_memory`: `rollout_memory=True`, `multiscale_steps=[1]`,
  `multiscale_weights=[1.0]`.

There is no `multiscale_steps=[1,2]` and no `multiscale_weights=[0.9,0.1]`.

Impact:

- These v02 results do not answer the requested question about
  "90% one-hop + 10% two-hop" weighted multiscale rollout.

## High-Severity Methodology Problems

### H1. Diagnostic loss curves are not validation/test curves

During training, `metrics.jsonl` records:

- `split = "train_diagnostic"`
- `phase = "train_no_test_read"`

The diagnostic rows are the first `min(128, len(train_store))` training rows.
So the plotted `eval_loss` curves are evaluations on a fixed subset of the
training set, not on validation or test.

Impact:

- `reports/probes_dense_to_one_easy_v02_loss_curves.md` should explicitly say
  that the upper panel is train-diagnostic loss.
- It should not be compared to final/test loss.

### H2. No validation split exists

The runner rejects `validation.jsonl` and final eval reads `test.jsonl` only
after training. That avoids direct test reads inside the loop, but it also means
there is no clean validation signal.

Impact:

- No model selection or early stopping can be audited.
- Curves cannot detect overfitting on held-out data.

### H3. Metadata fields are stale or misleading

Problems observed:

- Runner constant `VERSION` remains `probes_corrected_valid_as_test_l8_log5`,
  even though this is a v02 dense-to-one experiment.
- Per-run `git_commit` is `12dead9`, while v02 data/configs were later committed
  as `3dcfd0c`. This indicates the formal run was done on a remote repo at an
  older HEAD plus rsynced files. The identity does record config/data SHA, but
  the git metadata is not a clean reproduction pointer.
- `resolved_main_steps_1epoch` in the v02 task manifest is stale from the base
  manifest:
  - selective_copy: `625`
  - induction_associative_recall: `6250`
  - lra_listops: `6000`
  These values do not match the actual v02 train-set sizes. The training loop
  does not use this field; it uses `len(train_store)`, `epochs`, and `steps`.
- `final_eval.json` says `test_source = "source_validation_jsonl"` even though
  it reads `test.jsonl`.

Impact:

- Reproducibility metadata is not trustworthy without manual cross-checking.
- My previous explanation that emphasized `resolved_main_steps_1epoch` was
  wrong.

### H4. Single seed and shared masks

All formal v02 runs use `seed=0` only.

The random sparse mask is:

- shared across all layers;
- identical for `random_regular` and `random_memory` at a given density;
- effectively identical across tasks at a given density because `T`, block
  size, seed, and density are the same.

Impact:

- No uncertainty estimate.
- No evidence that findings survive random-mask variation.
- Not a layerwise-independent random experiment.

### H5. `random_regular` is a misleading method name here

With the pure density helper, row degree is approximately balanced, not truly
regular in a graph-theoretic sense:

| requested density | actual edge count | actual density | row degree pattern |
| ---: | ---: | ---: | --- |
| 0.1 | 410 | 0.10009765625 | 38 rows with 6 edges, 26 rows with 7 |
| 0.05 | 205 | 0.050048828125 | 51 rows with 3 edges, 13 rows with 4 |
| 0.025 | 102 | 0.02490234375 | 26 rows with 1 edge, 38 rows with 2 |
| 0.015625 | 64 | 0.015625 | 64 rows with 1 edge |

Impact:

- The label is okay as a code method name, but reports should define it as
  "pure random near-row-regular mask", not as a true random regular graph.

### H6. `listops_macro_accuracy` is incorrectly aggregated

For classification, `forward_loss_and_metrics` computes batch macro accuracy,
but the final aggregation uses per-sample rows. Each per-sample row sets
`listops_macro_accuracy` to 1.0 or 0.0, so aggregate macro accuracy becomes
ordinary accuracy.

Impact:

- Any reported `listops_macro_accuracy` is unreliable.
- `listops_accuracy` remains the usable metric.

### H7. Per-run `training_curves.png` is likely broken

`probe_metrics.write_training_curves()` filters:

```python
train_rows = [row for row in metrics_rows if row.get("split") == "train"]
```

But v02 metrics write `split = "train_diagnostic"`.

Impact:

- Per-run `training_curves.png` files are not reliable for v02.
- The later aggregate matplotlib plots are better because they parse
  `metrics.jsonl` directly.

## Things That Are Correct Or At Least Verified

These points do not rescue the experiment, but they matter for separating
implementation bugs from design flaws.

- The latest plotting commit `4f665c5` did not modify datasets or configs.
- Current checked-in v02 dataset/config files have no diff.
- Random masks in this v02 sweep do exclude deterministic block-local edges:
  `local_attention_pair_count = 0` for all tested random densities.
- Actual density is correctly computed as `random_attention_pair_count / 64^2`.
- Relative attention bias is disabled:
  `relative_attention_bias = False`.
- Top-k sparse attention is disabled:
  `attention_top_k = 0`.
- `random_regular` and `random_memory` use the same mask at each density, so
  the one-hop memory comparison is at least mask-matched.
- Training summaries record `test_read_during_training = false` and
  `validation_read_during_training = false`. The contamination is in the split
  construction, not in the training loop reading `test.jsonl`.
- Checkpoint identity includes config SHA, manifest SHA, train SHA, test SHA,
  and graph SHA, so stale checkpoints with a different identity should be
  rejected by `final_eval`.

## Result Tables Should Be Reinterpreted

Existing v02 final metrics:

| task | method | density | steps | final/test loss | reported acc |
| --- | --- | ---: | ---: | ---: | ---: |
| selective_copy | dense | dense | 1920 | 0.00015663423022260758 | 1.0 |
| selective_copy | random_regular | 0.10009765625 | 1920 | 0.0001612252120253288 | 1.0 |
| selective_copy | random_regular | 0.050048828125 | 1920 | 0.00021356190825372323 | 1.0 |
| selective_copy | random_regular | 0.02490234375 | 1920 | 1.3868791707791388 | 0.25 |
| selective_copy | random_regular | 0.015625 | 1920 | 1.3868803922086954 | 0.25 |
| selective_copy | random_memory | 0.10009765625 | 1920 | 0.00027477743617509987 | 1.0 |
| selective_copy | random_memory | 0.050048828125 | 1920 | 0.00031770644829975936 | 1.0 |
| selective_copy | random_memory | 0.02490234375 | 1920 | 1.0409549595788121 | 0.4375 |
| selective_copy | random_memory | 0.015625 | 1920 | 1.3868952970951796 | 0.25 |
| induction_associative_recall | dense | dense | 1920 | 0.00040127422062141704 | 1.0 |
| induction_associative_recall | random_regular | 0.10009765625 | 1920 | 0.00042712000094979885 | 1.0 |
| induction_associative_recall | random_regular | 0.050048828125 | 1920 | 0.0004019346765744558 | 1.0 |
| induction_associative_recall | random_regular | 0.02490234375 | 1920 | 0.4515711551066488 | 0.70703125 |
| induction_associative_recall | random_regular | 0.015625 | 1920 | 1.3168365051969886 | 0.2705078125 |
| induction_associative_recall | random_memory | 0.10009765625 | 1920 | 0.00045428914938838716 | 1.0 |
| induction_associative_recall | random_memory | 0.050048828125 | 1920 | 0.00058746857848746 | 1.0 |
| induction_associative_recall | random_memory | 0.02490234375 | 1920 | 0.9009454280603677 | 0.419921875 |
| induction_associative_recall | random_memory | 0.015625 | 1920 | 1.317342416383326 | 0.271484375 |
| lra_listops | dense | dense | 3000 | 0.0006384486833121627 | 1.0 |
| lra_listops | random_regular | 0.10009765625 | 3000 | 0.0004032587981782854 | 1.0 |
| lra_listops | random_regular | 0.050048828125 | 3000 | 0.0004713946959236637 | 1.0 |
| lra_listops | random_regular | 0.02490234375 | 3000 | 0.7295979380607605 | 0.775 |
| lra_listops | random_regular | 0.015625 | 3000 | 1.2308215022087097 | 0.7575 |
| lra_listops | random_memory | 0.10009765625 | 3000 | 0.46614989205030727 | 0.8225 |
| lra_listops | random_memory | 0.050048828125 | 3000 | 0.024712528941454367 | 1.0 |
| lra_listops | random_memory | 0.02490234375 | 3000 | 0.8630440402030944 | 0.775 |
| lra_listops | random_memory | 0.015625 | 3000 | 1.2312046301364898 | 0.7625 |

These numbers should be relabeled as:

"performance on a tiny finite calibration set whose test split has 100%
semantic overlap with train."

They should not be used to claim:

- held-out generalization;
- LRA ListOps performance;
- true induction/associative recall;
- weighted multiscale rollout effectiveness;
- robustness across seeds or masks.

## Required Fixes Before Any New Claim

1. Regenerate data with disjoint semantic train/validation/test splits.
2. For each task, define a genuine generalization axis:
   - selective_copy: new values, randomized source positions, longer/noisier
     sequences, held-out value/position combinations.
   - induction: randomized key-value mappings per example; held-out keys,
     values, orders, and positions.
   - listops: real depth/length variation with held-out trees, not depth-1
     exhaustive lookup.
3. Use a real validation split for curves and model selection.
4. Run the actual requested memory conditions:
   - plain random;
   - one-hop rollout memory;
   - weighted multiscale `[1,2]` with weights `[0.9,0.1]`.
5. Run multiple seeds for model init and random masks.
6. Decide whether random should be shared across layers or layerwise
   independent, and encode that explicitly in config/results.
7. Fix metadata:
   - version string;
   - git commit/deployed commit;
   - `resolved_main_steps_1epoch`;
   - `test_source`;
   - curve split naming.
8. Fix `listops_macro_accuracy` aggregation or remove it from reports.
9. Rename reports/tables so contaminated v02 results are clearly marked as
   calibration-only and invalid for generalization claims.

