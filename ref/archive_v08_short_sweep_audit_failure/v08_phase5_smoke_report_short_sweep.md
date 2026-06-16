# v08 Phase 5 Smoke 测试报告

## 结论

Phase 5 smoke 测试通过。六个 v08 选定任务、三个 required method、seed 0 共 18 个 run 均完成，所有结果行的字段契约审计均为 `passed`，训练日志中未出现 non-finite loss 或 NaN。

## 参数说明

- `config`: `configs/probes_v08_smoke.json`
- `profile`: `smoke`
- `remote_host`: `huiwei`
- `remote_code_root`: `/home/huiwei/ysx/zigzag_attention`
- `remote_data_root`: `/home/huiwei/ysx/expander_bench/data/probes`
- `remote_env`: `ysx_base`
- `CUDA_VISIBLE_DEVICES`: `3`
- `tasks`: `copy`, `selective_copy`, `induction_associative_recall`, `niah_kv_retrieval`, `ruler`, `lra_listops`
- `methods`: `local`, `zigzag_certified`, `random_regular`
- `seed`: `0`
- `smoke_steps`: `3`
- `attention_contract`: `non_causal`
- `causal`: `false`
- `graph_directionality`: `directed`
- `deployed_git_commit`: `1e6182f3`

Smoke 只用于接口、训练循环、图 artifact、字段契约和远端环境的连通性验证，不解释为最终模型质量。

## 结果摘要

| task | method | status | steps | primary metric | value | field audit |
|---|---|---|---:|---|---:|---|
| copy | local | ok | 3 | copy_token_accuracy | 0.0147705078125 | passed |
| copy | zigzag_certified | ok | 3 | copy_token_accuracy | 0.01397705078125 | passed |
| copy | random_regular | ok | 3 | copy_token_accuracy | 0.0159912109375 | passed |
| selective_copy | local | ok | 3 | selective_copy_token_accuracy | 0.0625 | passed |
| selective_copy | zigzag_certified | ok | 3 | selective_copy_token_accuracy | 0.046875 | passed |
| selective_copy | random_regular | ok | 3 | selective_copy_token_accuracy | 0.0390625 | passed |
| induction_associative_recall | local | ok | 3 | retrieval_exact_match | 0.0 | passed |
| induction_associative_recall | zigzag_certified | ok | 3 | retrieval_exact_match | 0.0 | passed |
| induction_associative_recall | random_regular | ok | 3 | retrieval_exact_match | 0.0 | passed |
| niah_kv_retrieval | local | ok | 3 | retrieval_exact_match | 0.0 | passed |
| niah_kv_retrieval | zigzag_certified | ok | 3 | retrieval_exact_match | 0.0 | passed |
| niah_kv_retrieval | random_regular | ok | 3 | retrieval_exact_match | 0.0 | passed |
| ruler | local | ok | 3 | retrieval_exact_match | 0.0 | passed |
| ruler | zigzag_certified | ok | 3 | retrieval_exact_match | 0.0 | passed |
| ruler | random_regular | ok | 3 | retrieval_exact_match | 0.0 | passed |
| lra_listops | local | ok | 3 | listops_accuracy | 0.125 | passed |
| lra_listops | zigzag_certified | ok | 3 | listops_accuracy | 0.125 | passed |
| lra_listops | random_regular | ok | 3 | listops_accuracy | 0.0 | passed |

## 预跑失败与修复审计

1. 第一次远端 smoke 在读取数据时失败，原因是 Phase 4 manifest 中的 `version_path` 为本地绝对路径。已在 `scripts/run_probe_experiment.py` 中加入远端路径解析，优先读取 manifest 路径，若不存在则按 `PROBE_V08_DATA_ROOT` 或默认远端数据根重映射到同一数据版本。
2. 第二次远端 smoke 在结果字段审计时失败，原因是 `audit_result_row` 用 set 检查空列表导致 `TypeError: unhashable type: 'list'`。已在 `scripts/probe_common.py` 中改为逐项判断 `None`、空字符串和空列表。
3. 第三次远端 smoke 完整通过，聚合结果为 `completed_runs=18`、`expected_runs=18`、`result_field_audits_passed=18`。

## 产物

- `outputs/probes_v08_smoke/summary.json`
- `outputs/probes_v08_smoke/results.csv`
- `outputs/probes_v08_smoke/results.jsonl`
- `outputs/probes_v08_smoke/result_field_audit.json`
- `outputs/probes_v08_smoke/*/*/summary.json`
- `outputs/probes_v08_smoke/*/*/metrics.jsonl`
- `outputs/probes_v08_smoke/*/*/training_curves.png`
- `logs/probes_v08_smoke_20260616T075425Z.log`
- `logs/probes_v08_smoke_20260616T075559Z.log`
- `logs/probes_v08_smoke_20260616T075730Z.log`
