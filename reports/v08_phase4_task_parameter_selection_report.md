# v08 Phase 4 Task Parameter Selection 报告

## 结论

Phase 4 已在 Phase 1 数据审计、Phase 2 dry-run 和 Phase 3 A100 80GB 远端 readiness 基础上重新冻结 6 个 probe task 的参数，并生成 smoke/main 配置、字段契约、编码器和每任务 directed non-causal graph artifact。required methods 为 `local, zigzag_certified, random_regular`；`dense` 暂列 optional，原因是 4k/8k 长上下文 dense reference 会显著挤占本轮主方法验证预算，v08 主线优先完成 theory-aligned zigzag_certified 与同预算 local/random_regular 对照。

| task | padded T | B/d | 模型 | effective batch | main steps | primary metric |
|---|---:|---:|---:|---:|---:|---|
| copy | 6144 | 64/8 | 4x128 | 8 | 1250 | copy_token_accuracy |
| selective_copy | 8256 | 64/8 | 3x96 | 8 | 1250 | selective_copy_token_accuracy |
| induction_associative_recall | 1024 | 64/8 | 4x128 | 16 | 11250 | retrieval_exact_match |
| niah_kv_retrieval | 4160 | 64/8 | 3x96 | 8 | 1250 | retrieval_exact_match |
| ruler | 4160 | 64/8 | 3x96 | 8 | 1500 | retrieval_exact_match |
| lra_listops | 6016 | 64/8 | 4x128 | 8 | 12000 | listops_accuracy |

## 选择依据

1. 长度策略：`resolved_padded_sequence_length` 保留 Phase 1 记录的最大输入/读出长度并按 `resolved_graph_block_size=64` 补齐；copy、selective_copy、niah、ruler、listops 均不把 validation/test 合同裁短。
2. 训练预算：`resolved_train_budget_value` 统一按 `ceil(resolved_train_examples / resolved_effective_batch_size)` 解析，表示每个 required method 至少看完一个完整 train split 的 full-train sweep，而不是 smoke 式固定步数。
3. zigzag 主方法效果：`resolved_layers`、`resolved_d_model` 和 `resolved_effective_batch_size` 按任务长度与 A100 80GB 显存选择，目标是在不降长度的前提下给 `zigzag_certified` 更多参数容量和足够更新步数。
4. 公平性：同一 task 内 `local`、`zigzag_certified`、`random_regular` 使用相同长度、模型、batch、optimizer、learning rate、seed、steps、validation budget 和 test budget。
5. random 对齐：`random_regular` 在运行入口按 `zigzag_actual_noncausal_per_query_unique_k` 生成 per-query 对齐的随机 remote rows；`random_k_alignment_error_max` 必须为 0，才能进入 main comparison。
6. 日志 gate：`resolved_log_every` 由 `resolved_train_budget_value` 和 1% logging gate 反推，`resolved_planned_logged_train_step_count` 不低于 `resolved_min_logged_train_step_count`，final step 必须记录。

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| resolved_padded_sequence_length | 图和位置编码使用的补齐后长度 | token 数 | Phase 1 长度与 B 补齐 | 决定 attention mask 尺寸 | 无 |
| resolved_graph_block_size | zigzag/local 的 block 大小 | token 数 | Phase 4 选择 | 控制局部预算和 q | 无 |
| resolved_graph_degree_or_budget | zigzag H 图 degree | 整数 | Phase 4 选择 | 控制稀疏远程边预算 | 无 |
| resolved_graph_max_parallel_edges_per_block_pair | G 图 block-pair 并行边上限 | not_capped | Phase 4 选择 | 避免对小 q 大 B 任务施加数学上不可满足的人为上限；实际重复率由 graph certificate 记录 | 无 |
| resolved_layers | Transformer 层数 | 层 | Phase 4 选择 | 记录模型容量 | 无 |
| resolved_d_model | hidden 维度 | 维度 | Phase 4 选择 | 记录模型容量 | 无 |
| resolved_effective_batch_size | 梯度累积后的有效 batch | 样本数 | batch_size * gradient_accumulation_steps | 保证 method 间公平 | 无 |
| resolved_train_examples | 训练集样本数 | 样本数 | Phase 1 audit | 计算 full-train sweep 步数 | 无 |
| resolved_train_budget_value | main 训练步数 | steps | ceil(train examples/effective batch) | 复现实验预算并证明不是短扫 | smoke 使用 smoke.steps 并记录 |
| resolved_log_every | 训练日志间隔 | steps | logging gate 反推 | 满足 1% 日志覆盖 | 无 |
| resolved_planned_logged_train_step_count | 计划记录的训练日志点数 | 行数 | Phase 4 公式 | 预先证明 logging gate 可满足 | 无 |
| resolved_min_logged_train_step_count | 最少训练日志点数 | 行数 | v08 手册 1% gate | 审计 metrics.jsonl 覆盖率 | 无 |
| random_k_alignment_error_max | random 与 zigzag 每 query K 最大误差 | token/key 数 | run 后 budget 诊断 | 验证 random_regular 同预算公平性 | 非 random run 写 not_applicable |
| random_target_k_source | random 对齐目标来源 | zigzag_actual_noncausal_per_query_unique_k | Phase 4/运行入口 | 说明 random budget 对齐口径 | 非 random run 写 not_applicable |
| attention_contract | 注意力合同 | non_causal | v08 手册 | 保证理论对齐 | 不满足则不得进入主结果 |
| causal | 是否 causal mask | false | v08 手册 | 防止 LM 化 | 不满足则不得进入主结果 |
| graph_directionality | 图方向性 | directed | v08 手册和 graph artifact | 对齐 directed expander | 无 |
| primary_metric | 每个 task 主指标 | metric 名称 | Phase 1 schema 与手册 | 主比较字段 | 无 |
| required_methods | 必跑方法集合 | local/zigzag_certified/random_regular | v08 手册和 Phase 4 | 定义 main comparison | 无 |
| optional_methods | 可选方法集合 | dense | Phase 4 资源取舍 | 说明未纳入主比较的方法 | 未运行写 not_applicable |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 20 |
| unexplained_parameters | [] |
| english_only_sections | [] |
