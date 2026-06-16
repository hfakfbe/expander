# v08 Phase 6 每任务报告

## 结论

本文件是 Phase 6 的每任务报告集合，补充 `reports/v08_probe_main_eval_report.md` 的总表。6 个 task 均按 Phase 4 冻结参数完成 `local`、`zigzag_certified`、`random_regular` 三个 required methods；每个 run 的 `attention_contract=non_causal`、`causal=false`、`graph_directionality=directed`，并通过字段审计和日志 gate。

## copy

| method | primary_metric_value | test_loss | validation_loss_final | final_train_loss | total_wall_time_sec | peak_allocated_gb | logged/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0.0162354 | 4.12795 | 4.12792 | 4.12797 | 310.675 | 2.382 | 15/13 |
| zigzag_certified | 0.0160576 | 4.12787 | 4.12788 | 4.1287 | 685.808 | 17.478 | 15/13 |
| random_regular | 0.0160093 | 4.12782 | 4.12785 | 4.12829 | 679.907 | 17.436 | 15/13 |

- 主指标: `copy_token_accuracy`。
- 训练预算: `steps=1250`, `train_examples=10000`, `effective_batch=8`。
- random 对齐: `zigzag_k_mean=120.578125`, `random_k_mean=120.578125`, `err_max=0`。
- 解释: 三个方法主指标都接近随机水平，本任务不能声称已学会 copy；但 full-train、日志和 budget gate 均通过。

## selective_copy

| method | primary_metric_value | test_loss | validation_loss_final | final_train_loss | total_wall_time_sec | peak_allocated_gb | logged/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0.07 | 2.64183 | 2.64047 | 2.64483 | 88.260 | 2.041 | 15/13 |
| zigzag_certified | 0.069 | 2.64122 | 2.64137 | 2.64364 | 415.282 | 14.261 | 15/13 |
| random_regular | 0.0746875 | 2.64113 | 2.64098 | 2.64797 | 411.913 | 14.187 | 15/13 |

- 主指标: `selective_copy_token_accuracy`。
- 训练预算: `steps=1250`, `train_examples=10000`, `effective_batch=8`。
- random 对齐: `zigzag_k_mean=120.578125`, `random_k_mean=120.578125`, `err_max=0`。
- 解释: random_regular 主指标略高，zigzag_certified 与 local 接近；本任务仍属于低准确率结果，应保守解释。

## induction_associative_recall

| method | primary_metric_value | test_loss | validation_loss_final | final_train_loss | total_wall_time_sec | peak_allocated_gb | logged/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0 | 8.32111 | 8.32133 | 8.31978 | 377.524 | 1.824 | 115/113 |
| zigzag_certified | 0 | 8.3209 | 8.32127 | 8.31736 | 1269.892 | 5.960 | 115/113 |
| random_regular | 0 | 8.32127 | 8.32083 | 8.32397 | 1290.768 | 5.958 | 115/113 |

- 主指标: `retrieval_exact_match`。
- 训练预算: `steps=11250`, `train_examples=180000`, `effective_batch=16`。
- random 对齐: `zigzag_k_mean=119.781250`, `random_k_mean=119.781250`, `err_max=0`。
- 解释: exact match 全部为 0；zigzag_certified 的 final train loss 略低，但不能据此声称检索成功。

## niah_kv_retrieval

| method | primary_metric_value | test_loss | validation_loss_final | final_train_loss | total_wall_time_sec | peak_allocated_gb | logged/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0.11 | 2.204 | 2.20343 | 2.1666 | 65.263 | 1.011 | 15/13 |
| zigzag_certified | 0.13 | 2.20392 | 2.20715 | 2.24809 | 212.515 | 7.150 | 15/13 |
| random_regular | 0.106 | 2.20789 | 2.20394 | 2.25163 | 210.415 | 7.129 | 15/13 |

- 主指标: `retrieval_exact_match`。
- 训练预算: `steps=1250`, `train_examples=10000`, `effective_batch=8`。
- random 对齐: `zigzag_k_mean=120.578125`, `random_k_mean=120.578125`, `err_max=0`。
- 解释: zigzag_certified 的 exact match 最高，为 0.13；single-seed 结果只能作为本轮主表观测。

## ruler

| method | primary_metric_value | test_loss | validation_loss_final | final_train_loss | total_wall_time_sec | peak_allocated_gb | logged/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0 | 3.69104 | 3.70954 | 3.58209 | 88.162 | 1.011 | 16/15 |
| zigzag_certified | 0 | 2.68691 | 2.77103 | 2.53056 | 279.876 | 7.150 | 16/15 |
| random_regular | 0 | 2.66503 | 2.69139 | 2.65037 | 276.652 | 7.129 | 16/15 |

- 主指标: `retrieval_exact_match`。
- 训练预算: `steps=1500`, `train_examples=12000`, `effective_batch=8`。
- random 对齐: `zigzag_k_mean=120.578125`, `random_k_mean=120.578125`, `err_max=0`。
- 解释: exact match 全部为 0；zigzag_certified 和 random_regular 的 loss 明显低于 local，但主指标没有转化为成功检索。

## lra_listops

| method | primary_metric_value | test_loss | validation_loss_final | final_train_loss | total_wall_time_sec | peak_allocated_gb | logged/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0.3315 | 1.8855 | 1.8976 | 1.53766 | 597.567 | 2.343 | 121/120 |
| zigzag_certified | 0.367 | 1.71408 | 1.75234 | 1.71986 | 4160.844 | 17.120 | 121/120 |
| random_regular | 0.355 | 1.70251 | 1.73009 | 1.61176 | 4398.060 | 17.080 | 121/120 |

- 主指标: `listops_accuracy`。
- 训练预算: `steps=12000`, `train_examples=96000`, `effective_batch=8`。
- random 对齐: `zigzag_k_mean=120.578117`, `random_k_mean=120.578117`, `err_max=0`。
- 解释: zigzag_certified 的主指标最高，为 0.367；random_regular 的 test loss 最低，但主指标低于 zigzag_certified。

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| `task` | probe 任务名 | 6 个任务名 | `configs/probes_v08_main.json` | 标识每任务报告对象 | 无 |
| `method` | attention 方法 | `local`, `zigzag_certified`, `random_regular` | run 输出 | 比较 required methods | 无 |
| `primary_metric_value` | 主指标值 | 0 到 1 | run 输出 | 每任务主比较值 | 无 |
| `copy_token_accuracy` | copy token 准确率 | 0 到 1 | copy run 输出 | copy 主指标 | 仅 copy 适用 |
| `selective_copy_token_accuracy` | selective copy token 准确率 | 0 到 1 | selective_copy run 输出 | selective_copy 主指标 | 仅 selective_copy 适用 |
| `retrieval_exact_match` | 检索完全匹配率 | 0 到 1 | induction/niah/ruler run 输出 | 检索类任务主指标 | 非检索任务不适用 |
| `listops_accuracy` | ListOps 分类准确率 | 0 到 1 | lra_listops run 输出 | listops 主指标 | 仅 lra_listops 适用 |
| `test_loss` | final test loss | 交叉熵类 loss | run 输出 | 补充主指标 | 无 |
| `validation_loss_final` | final validation loss | 交叉熵类 loss | run 输出 | 检查验证集末态 | 无 |
| `final_train_loss` | 最后训练 step loss | 交叉熵类 loss | run 输出 | 检查训练末态 | 无 |
| `total_wall_time_sec` | run 总耗时 | 秒 | run 输出 | 记录成本和吞吐背景 | 无 |
| `peak_allocated_gb` | GPU 峰值 allocated 显存 | GB | run 输出 | 记录显存占用 | 无 |
| `logged/min` | 实际训练日志条数/最低要求条数 | 行数 | `actual_logged_train_step_count` 和 `min_logged_train_step_count` | 验证日志 gate | 无 |
| `steps` | 实际训练步数 | optimizer steps | run 输出 | 验证 full-train budget | 无 |
| `train_examples` | 训练样本数 | examples | Phase 4 参数清单 | 计算训练预算 | 无 |
| `effective_batch` | 有效 batch size | examples | Phase 4 参数清单 | 计算 steps | 无 |
| `zigzag_k_mean` | zigzag non-causal 每 query K 平均值 | key 数 | run 输出 | 审计 sparse budget | 无 |
| `random_k_mean` | random non-causal 每 query K 平均值 | key 数 | run 输出 | 验证 random 同预算 | 仅 random 对齐说明使用 |
| `err_max` | random 与 zigzag 每 query K 最大误差 | key 数 | run 输出 `random_k_alignment_error_max` | 确认逐 query 对齐 | 非 random 方法不用于主表 |
| `attention_contract` | 注意力合同 | `non_causal` | run 输出 | v08 理论对齐 gate | 不满足则失败 |
| `causal` | 是否 causal mask | `false` | run 输出 | 防止 LM 口径混入 | 不满足则失败 |
| `graph_directionality` | 图方向性 | `directed` | run 输出 | directed zigzag gate | 不满足则失败 |
| `field audit` | 字段完整性审计 | passed/failed | `result_field_audit.json` | 检查输出字段完整性 | 无 |
| `log gate` | 日志覆盖审计 | passed/failed | `metrics.jsonl` 与审计文件 | 证明 final step 和 1% 覆盖 | 无 |
| `single-seed` | 单随机种子限制 | `seed=0` | config | 限定统计解释 | 多 seed 未运行时不可给置信区间 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 25 |
| unexplained_parameters | [] |
| english_only_sections | [] |
