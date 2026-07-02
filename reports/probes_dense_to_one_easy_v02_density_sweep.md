# Probes Dense-to-One Easy v02 Density Sweep

## Scope

This report adds lower-density `random_regular` and `random_memory` runs for `selective_copy`, `induction_associative_recall`, and `lra_listops`.

The sweep keeps the v02 model/training setup unchanged:

- 8 layers.
- Shared random mask across layers, so `是否每层单独随机 = 否`.
- Pure random mask, no deterministic block-local edges.
- `random_regular` and `random_memory` use the same mask at each density.
- `random_memory` remains single-hop rollout memory with lazy update.

The density values below are actual mask densities, not just requested densities. They are computed as `random_attention_pair_count / 64^2`.

For `lra_listops`, the `test token acc` column reports `listops_accuracy`, because the task is sequence-level classification and has no token accuracy field.

## Threshold Notes

- `selective_copy`: first observed density where at least one random method is below `0.95` is `0.02490234375`.
- `induction_associative_recall`: first observed density where at least one random method is below `0.95` is `0.02490234375`.
- `lra_listops`: `random_memory` is already below `0.95` at the original `0.10009765625`; in the lower-density supplement, both methods are below `0.95` at `0.02490234375`.

## selective_copy

| 方法名 | layer数 | 是否每层单独随机 | density | test loss | test token acc |
|---|---:|---|---:|---:|---:|
| random_regular | 8 | 否 | 0.10009765625 | 0.000161225212025 | 1 |
| random_memory | 8 | 否 | 0.10009765625 | 0.000274777436175 | 1 |
| random_regular | 8 | 否 | 0.050048828125 | 0.000213561908254 | 1 |
| random_memory | 8 | 否 | 0.050048828125 | 0.0003177064483 | 1 |
| random_regular | 8 | 否 | 0.02490234375 | 1.38687917078 | 0.25 |
| random_memory | 8 | 否 | 0.02490234375 | 1.04095495958 | 0.4375 |
| random_regular | 8 | 否 | 0.015625 | 1.38688039221 | 0.25 |
| random_memory | 8 | 否 | 0.015625 | 1.3868952971 | 0.25 |

## induction_associative_recall

| 方法名 | layer数 | 是否每层单独随机 | density | test loss | test token acc |
|---|---:|---|---:|---:|---:|
| random_regular | 8 | 否 | 0.10009765625 | 0.00042712000095 | 1 |
| random_memory | 8 | 否 | 0.10009765625 | 0.000454289149388 | 1 |
| random_regular | 8 | 否 | 0.050048828125 | 0.000401934676574 | 1 |
| random_memory | 8 | 否 | 0.050048828125 | 0.000587468578487 | 1 |
| random_regular | 8 | 否 | 0.02490234375 | 0.451571155107 | 0.70703125 |
| random_memory | 8 | 否 | 0.02490234375 | 0.90094542806 | 0.419921875 |
| random_regular | 8 | 否 | 0.015625 | 1.3168365052 | 0.2705078125 |
| random_memory | 8 | 否 | 0.015625 | 1.31734241638 | 0.271484375 |

## lra_listops

| 方法名 | layer数 | 是否每层单独随机 | density | test loss | test token acc |
|---|---:|---|---:|---:|---:|
| random_regular | 8 | 否 | 0.10009765625 | 0.000403258798178 | 1 |
| random_memory | 8 | 否 | 0.10009765625 | 0.46614989205 | 0.8225 |
| random_regular | 8 | 否 | 0.050048828125 | 0.000471394695924 | 1 |
| random_memory | 8 | 否 | 0.050048828125 | 0.0247125289415 | 1 |
| random_regular | 8 | 否 | 0.02490234375 | 0.729597938061 | 0.775 |
| random_memory | 8 | 否 | 0.02490234375 | 0.863044040203 | 0.775 |
| random_regular | 8 | 否 | 0.015625 | 1.23082150221 | 0.7575 |
| random_memory | 8 | 否 | 0.015625 | 1.23120463014 | 0.7625 |

## Evidence

New formal logs:

- `logs/probes_dense_to_one_easy_v02_density_sweep_gpu_state_20260702_115345.txt`
- `logs/probes_dense_to_one_easy_v02_density_sweep_selective_copy_20260702_115345.log`
- `logs/probes_dense_to_one_easy_v02_density_sweep_induction_associative_recall_20260702_115345.log`
- `logs/probes_dense_to_one_easy_v02_density_sweep_lra_listops_20260702_115345.log`

New aggregate outputs:

- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density05/results_train.csv`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density025/results_train.csv`
- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density015625/results_train.csv`

Existing baseline aggregate output:

- `outputs/probes_dense_to_one_easy_v02/runs/dense_to_one_easy_v02_random_density10/results_train.csv`
