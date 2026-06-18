# Corrected Copy v01 final report

Status: completed as a corrected Copy implementation and frozen one-shot final-eval run. Dense solved the corrected task; sparse zigzag/random did not solve under the frozen 1-epoch diagnostic budget; local behaved as the documented unreachable negative control.

## Isolation and provenance

- worktree_path: `/Users/sxye/Documents/expander-copy-corrected-v01`
- remote_worktree_path: `/home/huiwei/ysx/zigzag_attention_copy_corrected_v01`
- branch_name: `codex/copy-corrected-v01`
- branch_point_commit: `64845eb22b149fa5496dacecdfcdb610fbdc1cbb`
- code_commit_for_training: `498a465e6e5386d49e98f620d6418a9f57f12264`
- code_commit_for_final_eval: `498a465e6e5386d49e98f620d6418a9f57f12264`
- config_sha256: `dfd68a7e37a8f349d73a6f44b7c472d01bb32f2a9bb1f79a3ee007dc4960e37c`
- task_manifest_sha256: `1a75cf886043fda11e8bcbc9ad38b14f09e48976411ab12867ce66871270e1c9`
- selected_graph_sha256: `dde460290819f1c1271488560b7866f6708315f605a8ca10ef92d20c5dab8d60`
- train_sha256: `eb43507820dab41759c14f71a935060a729ad8ace1c4063957968e33146c68a7`
- test_sha256: `f3aa817cc1433c5d3d566f47c794c9c4daeb4c2360833797de045e147c8c9910`
- train_content_sha256: `ed34c697ccc4a7d50c233ecb6dbbd4cfad4fcf6faa21bfff5308d9ea5cf50a5e`
- test_content_sha256: `72be289c8982d76b110d256985efd10edbb8514e8c341be256f1a1559a49fbba`
- merge_back_to_main: `false`
- final_eval_freeze_sha256: `7a798b5a34b568142e546849b64531b8bf23aa2a886bab635c1bf5915e2ca7e5`

## Corrected Copy contract

- T=2048; source positions 0..1023; marker/readout positions 1024..2047; marker token 63.
- Target is not appended/injected/teacher-forced; loss is only marker positions 1024..2047.
- Identity integer encoding; vocab/output size 64; no integer shift.
- No padding; tensors are exactly [B, 2048].
- Learned absolute position embedding absent; nonlearnable RoPE rotates Q/K before every backend branch; position_parameter_count=0.
- Train split is old train; test split is old validation; old OOD test is discarded; no validation split is duplicated from test.
- Train mode records train_diagnostic only and did not read test; final-eval first opened test after checkpoint/config freeze.

## Reachability / structure gate

| method | target_in_1hop_rate | target_in_Lhop_rate | unreachable_rate | avg_shortest_path | notes |
|---|---:|---:|---:|---:|---|
| dense | 1.000000 | 1.000000 | 0.000000 | 1.000000 |  |
| local | 0.000000 | 0.000000 | 1.000000 | NA | structural negative control |
| zigzag_certified | 0.027344 | 1.000000 | 0.000000 | 2.143555 |  |
| random_regular | 0.033203 | 1.000000 | 0.000000 | 1.971680 | alignment max/mean=0/0.0 |

## Gates and final test results

| method | gate train token acc | gate train seq acc | gate train loss | final test token acc | final test seq acc | final test loss | checkpoint sha256 |
|---|---:|---:|---:|---:|---:|---:|---|
| dense | 1.000000000 | 1.000000000 | 0.005471816 | 1.000000000 | 1.000000000 | 0.005444592 | `54a0c197a9bde8a38779079c793fe7d7f01d849272119d1ca5d5977a161c7098` |
| local | 0.015502930 | 0.000000000 | 4.127619237 | 0.015905273 | 0.000000000 | 4.127435886 | `bafe0b9d9d68d17cfe4be5e9b936a4cf131c8615c10455598b8741ea57a400f8` |
| random_regular | 0.047119141 | 0.000000000 | 3.997054830 | 0.048803711 | 0.000000000 | 3.996641410 | `16ef630b82c184986cf309aa7e84b69b4c875c4621c10fb2f1560fd92f63fbb8` |
| zigzag_certified | 0.040344238 | 0.000000000 | 4.063384771 | 0.040809570 | 0.000000000 | 4.066061985 | `b7326f2441c9d97d496de0cf348dedd52b1cbabc756793e20ac44d2ca788efc0` |

Baselines from train marginal statistics: uniform64 accuracy `0.015625`, uniform64 NLL `4.1588830833596715`, empirical train marginal NLL `4.127130890358254`, global-mode token accuracy `0.016233203125`, position-wise mode token accuracy `0.01914599609375`.

Interpretation: dense reaches perfect corrected-copy generalization on the new test split. local remains at the structural negative-control baseline. random_regular and zigzag_certified are structurally reachable but, with this frozen 1-epoch/4-layer diagnostic budget, remain far from sequence-level copy and only slightly above marginal baselines; this should be reported as an optimization/architecture-budget negative diagnostic, not as evidence from the old invalid Copy setup.

## Final-eval isolation

- Checkpoint selection was frozen before test read in `outputs/copy_corrected_v01/final_eval_freeze.json`: latest/final `train_final_step625.pt` per method/seed; no test- or metric-based selection.
- First attempted final-eval failed for all methods before opening test because PyTorch default `torch.load(weights_only=True)` rejected checkpoint RNG state. The successful rerun used `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` only to load trusted in-run checkpoints; no config, checkpoint, model, data, graph, or training budget changed.
- dense first_test_read_at: `2026-06-18T17:00:01.937127+00:00`
- local first_test_read_at: `2026-06-18T17:00:21.809932+00:00`
- random_regular first_test_read_at: `2026-06-18T17:00:40.221475+00:00`
- zigzag_certified first_test_read_at: `2026-06-18T17:00:59.807059+00:00`

## Key artifact paths

- `configs/copy_corrected_v01.json`
- `configs/copy_corrected_v01_task_parameters.json`
- `outputs/copy_corrected_v01/final_eval_freeze.json`
- `outputs/copy_corrected_v01/graphs/copy/reachability.json`
- `outputs/copy_corrected_v01/runs/gate/dense/seed0/final_eval.json`
- `outputs/copy_corrected_v01/runs/gate/local/seed0/final_eval.json`
- `outputs/copy_corrected_v01/runs/gate/random_regular/seed0/final_eval.json`
- `outputs/copy_corrected_v01/runs/gate/zigzag_certified/seed0/final_eval.json`
