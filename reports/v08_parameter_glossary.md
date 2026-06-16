# v08 参数术语表

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| attention_contract | 注意力合同 | non_causal | v08 手册 | 判定是否进入主评测 | 不满足则失败 |
| causal | 是否使用 causal mask | false | v08 手册 | 防止 next-token LM 混入 | 不满足则失败 |
| graph_directionality | 图方向性 | directed | v08 手册 | 理论对齐要求 | 不满足则失败 |
| resolved_graph_max_parallel_edges_per_block_pair | G 图 block-pair 并行边上限 | not_capped | Phase 4 | 小 q 任务无法满足固定 2 上限，故不人为裁剪；由证书审计实际图性质 | 无 |
| required_methods | 必跑方法集合 | local/zigzag_certified/random_regular | Phase 4 | 定义主比较 | 无 |
| optional_methods | 可选方法集合 | dense | Phase 4 | 记录未优先运行的参考方法 | 未运行时写 not_applicable |
| train_budget_policy | 训练预算表达方式 | step_budget | Phase 4 | 复现训练长度 | 无 |
| log_every | 日志间隔 | steps | Phase 4 logging gate | 审计 metrics.jsonl 覆盖率 | 无 |
| actual_logged_train_step_count | 实际训练日志行数 | 行数 | run 后 metrics.jsonl | 验证 1% gate | 无 |
| primary_metric_value | 主指标值 | task-specific | eval | 排序和比较主结果 | 无 |
| random_target_k_source | random budget 对齐来源 | zigzag_actual_noncausal_per_query_unique_k | Phase 4/运行入口 | 验证 random_regular 同预算 | 非 random run 写 not_applicable |
| random_k_alignment_error_max | random 与 zigzag 每 query K 最大误差 | key 数 | run 后 budget 诊断 | 必须为 0 才算对齐 | 非 random run 写 not_applicable |
| resolved_train_examples | 训练样本总数 | 样本数 | Phase 1 audit | 计算 full-train sweep | 无 |
| resolved_effective_batch_size | 有效 batch | 样本数 | batch_size * gradient_accumulation_steps | 计算训练步数和吞吐 | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 14 |
| unexplained_parameters | [] |
| english_only_sections | [] |
