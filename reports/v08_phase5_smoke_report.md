# v08 Phase 5 Smoke Test 报告

## 结论

Phase 5 按 `configs/probes_v08_smoke.json` 对 6 个 task、3 个 required methods、seed 0 执行 smoke train + eval，共 18 个 run。聚合结果为 `status=ok`，`completed_runs=18`，`expected_runs=18`，`result_field_audits_passed=18`，`result_field_audits_total=18`。本阶段只验证入口、字段、graph/budget、metric、日志和曲线产物可用，不作为 Phase 6 主比较结论。

旧版一轮 main 与 1 x 1 曲线图产物已归档到 `ref/archive_v08_one_epoch_bad_curves/`；本报告对应重新生成后的 smoke 输出。

## Smoke 结果摘要

| task | local primary_metric_value | zigzag_certified primary_metric_value | random_regular primary_metric_value | train_steps | train_examples_seen |
|---|---:|---:|---:|---:|---:|
| copy | 0.014465 | 0.015747 | 0.012756 | 3 | 24 |
| selective_copy | 0.046875 | 0.070312 | 0.140625 | 3 | 24 |
| induction_associative_recall | 0.000000 | 0.000000 | 0.000000 | 3 | 48 |
| niah_kv_retrieval | 0.000000 | 0.000000 | 0.000000 | 3 | 24 |
| ruler | 0.000000 | 0.000000 | 0.000000 | 3 | 24 |
| lra_listops | 0.125000 | 0.000000 | 0.125000 | 3 | 24 |

## Gate 审计

| gate | 结果 |
|---|---|
| smoke 聚合状态 | `outputs/probes_v08_smoke/summary.json` 中 `status=ok` |
| 完成数量 | `completed_runs=18`，`expected_runs=18` |
| 字段审计 | `result_field_audits_passed=18`，`result_field_audits_total=18` |
| attention 合约 | 全部为 `attention_contract=non_causal`、`causal=false`、`graph_directionality=directed` |
| random budget 对齐 | 全部 `random_k_alignment_error_max=0`、`random_k_alignment_error_mean=0.0` |
| 曲线图 gate | 18 张 `training_curves.png` 均为 1080 x 960 PNG，最小文件大小 11198 bytes |
| NaN/非有限 loss | 未发现 `nan_detected=true` 或 `nonfinite_loss_detected=true` 的 run |

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| Phase 5 | v08 第 5 阶段 | smoke train + eval | v08 手册 | 区分 smoke 与 main | 无 |
| smoke | 轻量跑通配置 | profile 名称 | `configs/probes_v08_smoke.json` | 验证入口和字段，不做主结论 | main 阶段不用该 profile |
| task | probe 任务名 | 6 个任务之一 | Phase 4 manifest | 按任务分组审计和汇总 | 无 |
| method | attention 方法 | `local`、`zigzag_certified`、`random_regular` | Phase 4 manifest | 保证同任务内方法公平比较 | 无 |
| seed | 随机种子 | `0` | smoke config | 复现实验随机性 | 多 seed 不在本轮 smoke 中执行 |
| train_steps | 训练更新步数 | step 数 | resolved config | 检查 smoke 是否按配置执行 | 无 |
| train_examples_seen | 训练样本数 | example 数 | 运行时统计 | 确认实际训练量 | 无 |
| train_epochs | 训练等效 epoch | smoke 为 `not_applicable` | 运行结果 | smoke 只跑固定 3 steps，不声明 epoch | 报告中解释为不适用 |
| primary_metric_value | 主指标数值 | 0 到 1 或任务定义范围 | metric 计算 | smoke 确认 metric 能写出 | 不用于主结论 |
| completed_runs | 已完成 run 数 | run 数 | 聚合 summary | 验证所有 smoke 组合完成 | 无 |
| expected_runs | 预期 run 数 | run 数 | config 展开 | 对比是否漏跑 | 无 |
| result_field_audits_passed | 通过字段审计的 run 数 | run 数 | `result_field_audit.json` | 防止缺字段或空字段进入结果 | 无 |
| attention_contract | attention 合约 | `non_causal` | Phase 4 与运行结果 | 确认 v08 使用 non-causal 设定 | 无 |
| causal | 是否 causal mask | `false` | attention diagnostics | 与 `attention_contract` 互相校验 | 无 |
| graph_directionality | 图方向性 | `directed` | graph artifact | 确认使用有向图 | 无 |
| random_k_alignment_error_max | random 与 zigzag 每 query K 最大差 | K 的绝对差 | `random_budget.json` | 验证 random_regular 预算逐 query 对齐 | 无 |
| random_k_alignment_error_mean | random 与 zigzag 每 query K 平均差 | K 的平均绝对差 | `random_budget.json` | 防止平均预算漂移 | 无 |
| training_curves.png | 训练曲线图 | PNG 文件 | `scripts/probe_metrics.py` | 验证曲线真实生成，不允许 1 x 1 空图 | 无 |
| training_curves_png_satisfied | 曲线图 gate 是否通过 | 布尔值 | result field audit | 检查宽、高、大小阈值 | 不通过则该 run 审计失败 |
| nan_detected | 是否出现 NaN | 布尔值 | metrics 日志 | smoke 稳定性 gate | 无 |
| nonfinite_loss_detected | 是否出现非有限 loss | 布尔值 | metrics 日志 | 防止无效训练进入结果 | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 22 |
| unexplained_parameters | [] |
| english_only_sections | [] |
