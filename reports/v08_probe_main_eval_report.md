# v08 Probe Main Train + Eval 报告

## 结论

Phase 6 主训练与评测完成。六个 v08 选定任务、三个 required method、seed 0 共 18 个 run 均完成，结果字段审计 18/18 通过，日志检查未发现 Traceback、RuntimeError、CUDA OOM、non-finite loss 或 NaN。所有主结果均满足 `attention_contract=non_causal`、`causal=false`、`graph_directionality=directed`。

## 参数说明

- `config`: `configs/probes_v08_main.json`
- `profile`: `main`
- `remote_host`: `huiwei`
- `remote_code_root`: `/home/huiwei/ysx/zigzag_attention`
- `remote_data_root`: `/home/huiwei/ysx/expander_bench/data/probes`
- `remote_env`: `ysx_base`
- `tasks`: `copy`, `selective_copy`, `induction_associative_recall`, `niah_kv_retrieval`, `ruler`, `lra_listops`
- `methods`: `local`, `zigzag_certified`, `random_regular`
- `seed`: `0`
- `main_steps`: copy/selective_copy/niah/ruler/listops 为 `80`，induction_associative_recall 为 `120`
- `optimizer`: `adamw`
- `lr_scheduler`: `cosine`
- `base_learning_rate`: `3e-4`
- `min_learning_rate`: `3e-5`
- `attention_contract`: `non_causal`
- `causal`: `false`
- `graph_directionality`: `directed`
- `deployed_git_commit`: `c90fe8df`

`lra_pathfinder` 与 `lra_pathx` 不在主评估任务列表中，未进入 `configs/probes_v08_main.json`，也未出现在主结果表。

## 主结果

| task | method | steps | primary metric | value | test loss | peak GB |
|---|---|---:|---|---:|---:|---:|
| copy | local | 80 | copy_token_accuracy | 0.016142578125 | 4.136092058181763 | 0.40009260177612305 |
| copy | zigzag_certified | 80 | copy_token_accuracy | 0.01611474609375 | 4.1357198905944825 | 2.3005528450012207 |
| copy | random_regular | 80 | copy_token_accuracy | 0.01621044921875 | 4.136468312263489 | 2.3211426734924316 |
| selective_copy | local | 80 | selective_copy_token_accuracy | 0.0724375 | 2.679329532623291 | 0.399871826171875 |
| selective_copy | zigzag_certified | 80 | selective_copy_token_accuracy | 0.0733125 | 2.679596015930176 | 1.9330315589904785 |
| selective_copy | random_regular | 80 | selective_copy_token_accuracy | 0.0723125 | 2.6832066264152528 | 1.9597420692443848 |
| induction_associative_recall | local | 120 | retrieval_exact_match | 0.0 | 8.661381112162273 | 0.9451808929443359 |
| induction_associative_recall | zigzag_certified | 120 | retrieval_exact_match | 0.0 | 8.662625177001953 | 3.020430088043213 |
| induction_associative_recall | random_regular | 120 | retrieval_exact_match | 0.0 | 8.674885482279459 | 3.051729202270508 |
| niah_kv_retrieval | local | 80 | retrieval_exact_match | 0.13 | 3.133462734222412 | 0.18604516983032227 |
| niah_kv_retrieval | zigzag_certified | 80 | retrieval_exact_match | 0.118 | 3.083429039955139 | 0.9574770927429199 |
| niah_kv_retrieval | random_regular | 80 | retrieval_exact_match | 0.122 | 3.1299538040161132 | 0.970271110534668 |
| ruler | local | 80 | retrieval_exact_match | 0.0 | 4.379728702885764 | 0.18605375289916992 |
| ruler | zigzag_certified | 80 | retrieval_exact_match | 0.0 | 4.408753556796483 | 0.9574770927429199 |
| ruler | random_regular | 80 | retrieval_exact_match | 0.0 | 4.397342068842479 | 0.970271110534668 |
| lra_listops | local | 80 | listops_accuracy | 0.154 | 2.2711171650886537 | 0.3907599449157715 |
| lra_listops | zigzag_certified | 80 | listops_accuracy | 0.1835 | 2.2759289982318878 | 2.2507596015930176 |
| lra_listops | random_regular | 80 | listops_accuracy | 0.1835 | 2.2555516557693482 | 2.2725515365600586 |

## 审计

- `completed_runs`: 18
- `expected_runs`: 18
- `result_field_audits_passed`: 18
- `result_field_audits_total`: 18
- `git_dirty`: 全部为 `false`
- `log_policy_satisfied`: 全部为 `true`
- `graph_certified`: 所有 `zigzag_certified` 与同图预算记录均保留 graph certificate
- `forbidden_tasks_in_main_results`: `[]`
- `failed_or_partial_rows`: `[]`
- `log_error_scan`: 未发现 Traceback、RuntimeError、CUDA OOM、non-finite loss 或 NaN

## 产物

- `outputs/probes_v08_main/summary.json`
- `outputs/probes_v08_main/results.csv`
- `outputs/probes_v08_main/results.jsonl`
- `outputs/probes_v08_main/result_field_audit.json`
- `outputs/probes_v08_main/*/*/summary.json`
- `outputs/probes_v08_main/*/*/metrics.jsonl`
- `outputs/probes_v08_main/*/*/training_curves.png`
- `outputs/probes_v08_main/*/*/attention_diagnostics.json`
- `outputs/probes_v08_main/*/*/artifacts/graph/selected_graph.json`
- `outputs/probes_v08_main/*/*/artifacts/graph/graph_certificate.json`
- `logs/probes_v08_main_copy_20260616T080234Z.log`
- `logs/probes_v08_main_selective_copy_20260616T080234Z.log`
- `logs/probes_v08_main_induction_associative_recall_20260616T080234Z.log`
- `logs/probes_v08_main_niah_kv_retrieval_20260616T080234Z.log`
- `logs/probes_v08_main_ruler_20260616T080555Z.log`
- `logs/probes_v08_main_lra_listops_20260616T080555Z.log`
