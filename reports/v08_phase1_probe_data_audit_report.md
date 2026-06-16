# v08 Phase 1 Probe Data Contract Audit 报告

## 结论

本阶段按 `ref/zigzag_experiment_execution_manual_v08.md` 只审计 6 个 `validated` 且 `can_enter_main_eval=true` 的 probe 数据版本。审计状态：`validated`。`lra_pathfinder` 与 `lra_pathx` 未进入本阶段清单。

| task | 状态 | train/validation/test 行数 | input 长度 min/mean/max | target 长度 min/mean/max | 推荐主指标 |
|---|---|---:|---:|---:|---|
| copy | validated | 10000/1000/1000 | 2048/2218.7/4096 | 1024/1109.3/2048 | copy_token_accuracy |
| selective_copy | validated | 10000/1000/1000 | 4128/4469.3/8224 | 16/16.0/16 | selective_copy_token_accuracy |
| induction_associative_recall | validated | 180000/3000/4000 | 64/141.3/1024 | 4/17.7/256 | retrieval_exact_match |
| niah_kv_retrieval | validated | 10000/500/500 | 3487/4055.1/4096 | 1/1.0/1 | retrieval_exact_match |
| ruler | validated | 12000/3000/3000 | 2561/3983.0/4096 | 1/3.5/10 | retrieval_exact_match |
| lra_listops | validated | 96000/2000/2000 | 1501/3095.7/5995 | 1/1.0/1 | listops_accuracy |

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| attention_contract | 注意力合同 | non_causal | dataset_card.non_causal_contract 与 v08 手册 | 确认主实验不是 causal LM | 不适用时不得进入主评测 |
| causal | 是否使用 causal mask | false | v08 手册 | 防止误用 next-token 任务 | 不适用时不得进入主评测 |
| checksum_status | sha256 校验状态 | ok/failed | checksums.sha256 | 确认数据内容可追溯 | failed 时阻止后续 phase |
| train_rows | 训练集行数 | 样本数 | JSONL 行数 | 和部署合同核对 | 无 |
| validation_rows | 验证集行数 | 样本数 | JSONL 行数 | 和部署合同核对 | 无 |
| test_rows | 测试集行数 | 样本数 | JSONL 行数 | 和部署合同核对 | 无 |
| input_schema | input 字段类型分布 | JSON 字符串 | JSONL 扫描 | 决定编码器和模型输入 | 无 |
| target_schema | target 字段类型分布 | JSON 字符串 | JSONL 扫描 | 决定 loss 和 metric | 无 |
| input_length_min/mean/max | 输入长度统计 | token/字符合同长度 | JSONL 与 metadata | Phase 4 选择长度和图规模 | 无 |
| target_length_min/mean/max | 目标长度统计 | token/标签数 | JSONL 与 metadata | Phase 4 选择 readout 和 loss | 无 |
| recommended_metric | 推荐主指标 | metric 名称 | v08 手册和数据 schema | Phase 4 冻结主指标 | schema 不匹配时在 Phase 4 修正 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 13 |
| unexplained_parameters | [] |
| english_only_sections | [] |
