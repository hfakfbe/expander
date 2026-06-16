# v08 Phase 6 Main Eval 总报告

## 结论

Phase 6 已按 `configs/probes_v08_main.json` 和 `configs/probes_v08_task_parameters.json` 重新执行完整 main train + eval。主实验覆盖 6 个 task、3 个 required methods、seed 0，共 18 个 run；聚合结果为 `status=ok`、`completed_runs=18`、`expected_runs=18`、`result_field_audits_passed=18`、`result_field_audits_total=18`。

这次 main 不再是一轮 full-train sweep，而是 Phase 4 冻结的 5 个 train-equivalent epochs。对应训练样本数为：copy 50000、selective_copy 50000、niah_kv_retrieval 50000、ruler 60000、induction_associative_recall 900000、lra_listops 480000。旧版一轮训练与 1 x 1 空白曲线图产物已归档到 `ref/archive_v08_one_epoch_bad_curves/`。

结果上，lra_listops 学到了可见信号，zigzag_certified 的 `listops_accuracy=0.383`；niah_kv_retrieval 有低水平 retrieval exact match；copy、induction_associative_recall 和 ruler 仍未学起来。该结论比上一版更可靠，因为训练预算、random budget 和曲线图 gate 都已通过审计。

## 主结果表

| task | method | primary_metric_name | primary_metric_value | test_loss | train_steps | train_examples_seen | train_epochs |
|---|---|---|---:|---:|---:|---:|---:|
| copy | local | copy_token_accuracy | 0.016165 | 4.127336 | 6250 | 50000 | 5 |
| copy | zigzag_certified | copy_token_accuracy | 0.016072 | 4.127310 | 6250 | 50000 | 5 |
| copy | random_regular | copy_token_accuracy | 0.016214 | 4.127317 | 6250 | 50000 | 5 |
| selective_copy | local | selective_copy_token_accuracy | 0.070500 | 2.641094 | 6250 | 50000 | 5 |
| selective_copy | zigzag_certified | selective_copy_token_accuracy | 0.068500 | 2.640435 | 6250 | 50000 | 5 |
| selective_copy | random_regular | selective_copy_token_accuracy | 0.068875 | 2.640640 | 6250 | 50000 | 5 |
| induction_associative_recall | local | retrieval_exact_match | 0.000000 | 8.334947 | 56250 | 900000 | 5 |
| induction_associative_recall | zigzag_certified | retrieval_exact_match | 0.000000 | 9.691010 | 56250 | 900000 | 5 |
| induction_associative_recall | random_regular | retrieval_exact_match | 0.000000 | 8.822923 | 56250 | 900000 | 5 |
| niah_kv_retrieval | local | retrieval_exact_match | 0.122000 | 2.196365 | 6250 | 50000 | 5 |
| niah_kv_retrieval | zigzag_certified | retrieval_exact_match | 0.116000 | 2.200174 | 6250 | 50000 | 5 |
| niah_kv_retrieval | random_regular | retrieval_exact_match | 0.104000 | 2.198175 | 6250 | 50000 | 5 |
| ruler | local | retrieval_exact_match | 0.000000 | 3.664145 | 7500 | 60000 | 5 |
| ruler | zigzag_certified | retrieval_exact_match | 0.000000 | 2.832849 | 7500 | 60000 | 5 |
| ruler | random_regular | retrieval_exact_match | 0.000000 | 2.893758 | 7500 | 60000 | 5 |
| lra_listops | local | listops_accuracy | 0.368000 | 1.599512 | 60000 | 480000 | 5 |
| lra_listops | zigzag_certified | listops_accuracy | 0.383000 | 1.581749 | 60000 | 480000 | 5 |
| lra_listops | random_regular | listops_accuracy | 0.366500 | 1.596271 | 60000 | 480000 | 5 |

## 审计结果

| gate | 结果 |
|---|---|
| 主实验完成度 | `outputs/probes_v08_main/summary.json` 中 `completed_runs=18`、`expected_runs=18` |
| 字段审计 | `result_field_audits_passed=18`、`result_field_audits_total=18` |
| attention 合约 | 全部为 `attention_contract=non_causal`、`causal=false`、`graph_directionality=directed` |
| random_regular 对齐 | 全部 `random_k_aligned_to_zigzag=true`、`random_k_alignment_error_max=0`、`random_k_alignment_error_mean=0.0` |
| 曲线图 gate | 18 张 `training_curves.png` 均为 1080 x 960 PNG，最小文件大小 15196 bytes，最大文件大小 36991 bytes |
| 训练日志覆盖 | `actual_logged_train_step_count` 范围为 65 到 601，均不小于对应 `min_logged_train_step_count` |
| 失败或 OOM | 无 failed run；`oom_fallback_applied=false`；未发现 NaN 或非有限 loss |

## 解释与限制

1. `train_epochs=5` 表示按 Phase 4 预算折算为 5 个训练集等效覆盖量。当前数据 loader 使用确定性采样和 step budget，因此它不是严格顺序读取 5 个 epoch 的声明。
2. 本轮仍是 seed 0 单种子实验；method 间小幅差异不能解释为统计显著差异。
3. copy、induction_associative_recall、ruler 的主指标仍低，说明 5 个 train-equivalent epochs 仍不足以让这些配置可靠学起任务，或任务头、优化器、读出方式仍需进一步诊断。
4. random_regular 已按 zigzag 的实际 non-causal per-query K 对齐，不再存在上一版 random budget 更高的问题。
5. 曲线图生成逻辑已移除 1 x 1 静默 fallback；即使 matplotlib 不可用，也会写出带坐标轴和曲线的 PNG，并由 result field audit 检查尺寸和大小。

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| Phase 6 | v08 第 6 阶段 | Full Train + Eval | v08 手册 | 标记主实验阶段 | 无 |
| main | 主实验 profile | profile 名称 | `configs/probes_v08_main.json` | 与 smoke 区分 | smoke 报告单独解释 |
| task | probe 任务名 | 6 个任务之一 | Phase 4 manifest | 按任务比较方法 | 无 |
| method | attention 方法 | `local`、`zigzag_certified`、`random_regular` | Phase 4 manifest | 同任务内公平比较 | 无 |
| local | 局部 attention baseline | method 名称 | 训练入口 | 稀疏 baseline | 无 |
| zigzag_certified | 证书化 zigzag attention 主方法 | method 名称 | graph artifact | v08 主方法 | 无 |
| random_regular | 同预算随机规则图 baseline | method 名称 | random budget | 与 zigzag 同 K 预算比较 | 无 |
| seed | 随机种子 | `0` | main config | 复现实验 | 多 seed 未执行时报告限制 |
| primary_metric_name | 主指标名称 | metric 名称 | Phase 4 metric 选择 | 说明表中数值含义 | 无 |
| primary_metric_value | 主指标数值 | 0 到 1 或任务定义范围 | test eval | 主比较数值 | 无 |
| test_loss | test split loss | loss 值 | eval 过程 | 辅助判断学习状态 | 无 |
| train_steps | 训练更新步数 | step 数 | Phase 4 resolved value | 确认主训练预算 | 无 |
| train_examples_seen | 实际训练样本数 | example 数 | 运行时统计 | 验证不是一轮 sweep | 无 |
| train_epochs | 训练等效 epoch | 本轮为 5 | Phase 4 resolved value | 说明训练预算强度 | smoke 中不适用 |
| train_epochs_planned | 计划训练等效 epoch | 本轮为 5 | Phase 4 manifest | 检查执行与计划一致 | 无 |
| train_epochs_completed | 完成训练等效 epoch | 本轮为 5 | 运行结果 | 检查训练未提前停止 | 无 |
| attention_contract | attention 合约 | `non_causal` | config 与 diagnostics | 确认 v08 非 causal 设定 | 无 |
| causal | 是否 causal mask | `false` | attention diagnostics | 与合约互相校验 | 无 |
| graph_directionality | 图方向性 | `directed` | graph artifact | 确认有向图 | 无 |
| random_k_aligned_to_zigzag | random 是否逐 query 对齐 zigzag K | 布尔值 | random budget | 保证 baseline 公平 | 无 |
| random_k_alignment_error_max | random 与 zigzag 每 query K 最大差 | K 的绝对差 | random budget | 发现预算污染 | 无 |
| random_k_alignment_error_mean | random 与 zigzag 每 query K 平均差 | K 的平均绝对差 | random budget | 发现平均预算漂移 | 无 |
| training_curves.png | 训练曲线图 | PNG 文件 | `scripts/probe_metrics.py` | 检查训练轨迹 | 无 |
| training_curves_png_satisfied | 曲线图 gate 是否通过 | 布尔值 | result field audit | 防止 1 x 1 空图通过 | 不通过则 run 审计失败 |
| actual_logged_train_step_count | 实际记录的训练日志点数 | 行数 | metrics 日志 | 验证曲线和日志覆盖 | 无 |
| min_logged_train_step_count | 最小要求训练日志点数 | 行数 | Phase 4 log policy | 审计日志是否足够 | 无 |
| result_field_audits_passed | 通过字段审计的 run 数 | run 数 | 聚合 summary | 防止缺字段结果进入主比较 | 无 |
| completed_runs | 已完成 run 数 | run 数 | 聚合 summary | 检查是否漏跑 | 无 |
| expected_runs | 预期 run 数 | run 数 | config 展开 | 对照完成度 | 无 |
| oom_fallback_applied | 是否触发 OOM 降级 | 布尔值 | 运行结果 | 检查是否临时改变参数 | 未触发时为 `false` |
| nonfinite_loss_detected | 是否出现非有限 loss | 布尔值 | metrics 日志 | 训练有效性 gate | 无 |
| nan_detected | 是否出现 NaN | 布尔值 | metrics 日志 | 训练稳定性 gate | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 32 |
| unexplained_parameters | [] |
| english_only_sections | [] |
