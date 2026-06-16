# v08 参数术语表

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

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 10 |
| unexplained_parameters | [] |
| english_only_sections | [] |
