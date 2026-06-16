from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from graph_diagnostics import certificate_for_artifact
from graph_structures import build_graph_artifact
from probe_common import (
    ATTENTION_CONTRACT,
    EXPERIMENT_VERSION,
    FIELD_CONTRACT,
    GRAPH_DIRECTIONALITY,
    SELECTED_PROBES,
    command_string,
    file_sha256,
    read_json,
    selected_probe_path,
    stats,
    utc_now,
    write_command,
    write_csv,
    write_json,
    write_jsonl,
)
from probe_tasks import byte_encoder, build_listops_encoder, integer_encoder, padded_length


REQUIRED_METHODS = ["local", "zigzag_certified", "random_regular"]
OPTIONAL_METHODS = ["dense"]


def _numeric_max(version_dir: Path) -> int:
    max_value = 0
    for split in ["train", "validation", "test"]:
        with (version_dir / f"{split}.jsonl").open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                row = json.loads(line)
                values = []
                if isinstance(row.get("input"), list):
                    values.extend(item for item in row["input"] if isinstance(item, int))
                target = row.get("target")
                if isinstance(target, list):
                    for item in target:
                        if isinstance(item, int):
                            values.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("value"), int):
                            values.append(item["value"])
                elif isinstance(target, int):
                    values.append(target)
                if values:
                    max_value = max(max_value, max(int(item) for item in values))
    return max_value


def _report_field(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _select_model(task: str, max_seq: int) -> dict:
    if task in {"copy", "induction_associative_recall", "lra_listops"}:
        return {"layers": 4, "d_model": 128, "heads": 4, "ffn_dim": 512, "dropout": 0.1}
    if task == "selective_copy":
        return {"layers": 3, "d_model": 96, "heads": 4, "ffn_dim": 384, "dropout": 0.1}
    return {"layers": 3, "d_model": 96, "heads": 4, "ffn_dim": 384, "dropout": 0.1}


def _select_batch(task: str, max_seq: int) -> dict:
    if max_seq >= 4096:
        return {"batch_size": 8, "gradient_accumulation_steps": 1, "eval_batch_size": 8}
    if task == "induction_associative_recall":
        return {"batch_size": 16, "gradient_accumulation_steps": 1, "eval_batch_size": 32}
    return {"batch_size": 4, "gradient_accumulation_steps": 2, "eval_batch_size": 8}


def _select_steps(train_rows: int, effective_batch: int) -> int:
    return max(1, math.ceil(int(train_rows) / max(int(effective_batch), 1)))


def _select_log_every(steps: int) -> int:
    if steps < 100:
        return 1
    min_logs = math.ceil(steps * 0.01)
    return max(1, steps // min_logs)


def _task_lengths(audit_row: dict, version_dir: Path, loss_type: str) -> tuple[int, int, int]:
    max_input = int(float(audit_row["input_length_max"]))
    max_target = int(float(audit_row["target_length_max"]))
    if loss_type == "mqar_position_cross_entropy":
        raw = max_input
        readout_start = max_input
    elif loss_type == "classification_cross_entropy":
        raw = max_input
        readout_start = max_input
    else:
        raw = max_input + max_target
        readout_start = max_input
    return raw, max_input, readout_start


def _loss_type(task: str) -> str:
    if task in {"copy", "selective_copy"}:
        return "sequence_cross_entropy"
    if task == "induction_associative_recall":
        return "mqar_position_cross_entropy"
    if task in {"niah_kv_retrieval", "ruler"}:
        return "retrieval_sequence_cross_entropy"
    if task == "lra_listops":
        return "classification_cross_entropy"
    raise ValueError(task)


def _encoder(task: str, version_dir: Path, output_dir: Path) -> dict:
    encoder_path = output_dir / "encoders" / task / "encoder.json"
    if task in {"niah_kv_retrieval", "ruler"}:
        payload = byte_encoder(encoder_path)
    elif task == "lra_listops":
        payload = build_listops_encoder(version_dir / "train.jsonl", encoder_path)
    else:
        payload = integer_encoder(_numeric_max(version_dir), encoder_path)
    payload["path"] = str(encoder_path)
    payload["sha256"] = file_sha256(encoder_path)
    return payload


def _write_graph(task: str, raw_length: int, B: int, d: int, seed: int, output_dir: Path) -> dict:
    graph_dir = output_dir / "graphs" / task
    graph_dir.mkdir(parents=True, exist_ok=True)
    g_config = {"max_parallel_edges_per_block_pair": None}
    artifact = build_graph_artifact(
        N_task=raw_length,
        T_raw=raw_length,
        block_size=B,
        degree=d,
        graph_seed=seed,
        g_config=g_config,
        version=EXPERIMENT_VERSION,
    )
    artifact["allow_multiedges"] = True
    artifact["preserve_multiplicity"] = True
    artifact["graph_generation_algorithm"] = "zigzag_v08_task_parameter_selection"
    cert = certificate_for_artifact(
        artifact,
        {
            "acceptance": {
                "rho_bound_lt": 1.0,
                "max_remote_local_overlap_mean": 0.5,
            }
        },
    )
    artifact["certificate"] = cert
    write_json(graph_dir / "selected_graph.json", artifact)
    write_json(graph_dir / "graph_certificate.json", cert)
    sha = file_sha256(graph_dir / "selected_graph.json")
    (graph_dir / "graph_artifact.sha256").write_text(sha + "  selected_graph.json\n", encoding="utf-8")
    generation = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "command": command_string(),
        "graph_generation_algorithm": artifact["graph_generation_algorithm"],
        "graph_seed": int(seed),
        "N_task": int(raw_length),
        "T_raw": int(raw_length),
        "T": int(artifact["T"]),
        "q": int(artifact["q"]),
        "B": int(B),
        "d": int(d),
        "max_parallel_edges_per_block_pair": g_config["max_parallel_edges_per_block_pair"],
        "generation_attempts": 1,
        "canonical_graph_artifact_sha256": sha,
        "selected_graph_path": str(graph_dir / "selected_graph.json"),
        "graph_certificate_path": str(graph_dir / "graph_certificate.json"),
    }
    write_json(graph_dir / "graph_generation.json", generation)
    return {
        "graph_dir": str(graph_dir),
        "selected_graph_path": str(graph_dir / "selected_graph.json"),
        "graph_certificate_path": str(graph_dir / "graph_certificate.json"),
        "graph_generation_path": str(graph_dir / "graph_generation.json"),
        "graph_artifact_sha256_path": str(graph_dir / "graph_artifact.sha256"),
        "selected_graph_sha256": sha,
        "certificate": cert,
        "artifact": artifact,
    }


def build_task_record(task: str, audit_row: dict, output_dir: Path) -> dict:
    version_dir = selected_probe_path(task)
    card = read_json(version_dir / "dataset_card.json")
    loss_type = _loss_type(task)
    raw_length, input_limit, readout_start = _task_lengths(audit_row, version_dir, loss_type)
    B = 64 if raw_length >= 1024 else 32
    d = 8
    T = padded_length(raw_length, B)
    encoder = _encoder(task, version_dir, output_dir)
    model = _select_model(task, T)
    batch = _select_batch(task, T)
    train_rows = int(audit_row["train_rows"])
    validation_rows = int(audit_row["validation_rows"])
    test_rows = int(audit_row["test_rows"])
    effective_batch = batch["batch_size"] * batch["gradient_accumulation_steps"]
    main_steps = _select_steps(train_rows, effective_batch)
    smoke_steps = 3
    log_every = _select_log_every(main_steps)
    smoke_log_every = 1
    graph = _write_graph(task, raw_length, B, d, seed=0, output_dir=output_dir)
    param_count_estimate = (
        encoder["vocab_size"] * model["d_model"]
        + T * model["d_model"]
        + model["layers"] * (12 * model["d_model"] * model["d_model"] + 2 * model["d_model"] * model["ffn_dim"])
    )
    secondary = {
        "copy": ["copy_sequence_accuracy", "copy_eos_accuracy"],
        "selective_copy": ["selective_copy_sequence_accuracy"],
        "induction_associative_recall": ["retrieval_token_accuracy"],
        "niah_kv_retrieval": ["retrieval_token_accuracy"],
        "ruler": ["retrieval_token_accuracy", "ruler_subtask_exact_match", "ruler_subtask_token_accuracy"],
        "lra_listops": ["listops_macro_accuracy"],
    }[task]
    record = {
        "task": task,
        "version_path": str(version_dir),
        "attention_contract": ATTENTION_CONTRACT,
        "causal": False,
        "graph_directionality": GRAPH_DIRECTIONALITY,
        "input_schema": audit_row["input_schema"],
        "target_schema": audit_row["target_schema"],
        "primary_metric": SELECTED_PROBES[task]["primary_metric"],
        "secondary_metrics": secondary,
        "input_length_min_mean_max": [audit_row["input_length_min"], audit_row["input_length_mean"], audit_row["input_length_max"]],
        "target_length_min_mean_max": [audit_row["target_length_min"], audit_row["target_length_mean"], audit_row["target_length_max"]],
        "chosen_train_length_policy": "use_full_declared_train_input_length_and_task_readout",
        "chosen_eval_length_policy": "support_max_validation_test_length_from_phase1",
        "encoder_or_tokenizer": encoder["encoder_type"],
        "label_or_value_space": "classes_0_9" if task == "lra_listops" else f"vocab_size_{encoder['vocab_size']}",
        "loss_type": loss_type,
        "model_family": "probe_transformer_encoder_readout",
        "model_capacity": model,
        "graph_block_policy": "padded_sequence_blocks",
        "graph_degree_or_budget_policy": "B64_d8_for_long_tasks_or_B32_d8_for_short_tasks_with_no_artificial_parallel_edge_cap",
        "required_methods": REQUIRED_METHODS,
        "optional_methods": OPTIONAL_METHODS,
        "seed_policy": "single_seed_0_for_v08_first_complete_sweep",
        "train_split_policy": "deterministic_random_sampling_from_full_train_jsonl",
        "validation_split_policy": "full_validation_for_main_limited_validation_for_logs",
        "test_split_policy": "full_test_for_main",
        "effective_batch_policy": "batch_size_times_gradient_accumulation",
        "optimizer_policy": "adamw",
        "lr_schedule_policy": "cosine_with_warmup",
        "train_budget_policy": "step_budget",
        "validation_eval_policy": "main_full_validation_final_eval_plus_small_log_eval",
        "test_eval_policy": "main_full_test_final_eval",
        "logging_policy": "log_every_satisfies_v08_1_percent_gate_and_final_step",
        "checkpoint_policy": "no_tensor_checkpoint_for_v08_first_sweep",
        "oom_or_runtime_fallback_policy": "drop_optional_dense_first_then_reduce_batch_for_all_required_methods",
        "selection_reason": (
            "该配置在 non-causal directed contract 下保留完整 Phase 1 最大长度，不截断 validation/test 合同；"
            f"main 训练预算解析为 ceil(train_rows/effective_batch)={main_steps} steps，"
            f"每个 required method 至少覆盖 {train_rows} 个训练样本的一轮 full-train sweep；"
            "batch 和模型容量按 Phase 3 A100 80GB 资源、Phase 5 smoke 可行性和长序列显存成本选择，"
            "优先保证 zigzag_certified 有足够更新步数学习主指标；local/random_regular 使用同一 task 级长度、模型、batch、optimizer、seed 和 steps。"
        ),
        "resolved_input_length_policy": "max_phase1_input_length",
        "resolved_target_length_policy": "max_phase1_target_length_or_position_targets",
        "resolved_sequence_length_min": int(float(audit_row["input_length_min"])),
        "resolved_sequence_length_mean": float(audit_row["input_length_mean"]),
        "resolved_sequence_length_p95": float(audit_row["input_length_p95"]),
        "resolved_sequence_length_max": int(float(audit_row["input_length_max"])),
        "resolved_target_length_min": int(float(audit_row["target_length_min"])),
        "resolved_target_length_mean": float(audit_row["target_length_mean"]),
        "resolved_target_length_p95": float(audit_row["target_length_p95"]),
        "resolved_target_length_max": int(float(audit_row["target_length_max"])),
        "resolved_runtime_input_length": input_limit,
        "resolved_runtime_target_length": int(float(audit_row["target_length_max"])),
        "resolved_readout_start": readout_start,
        "resolved_raw_sequence_length": raw_length,
        "resolved_padded_sequence_length": T,
        "resolved_train_examples": train_rows,
        "resolved_validation_examples": validation_rows,
        "resolved_test_examples": test_rows,
        "resolved_train_split_sha256": file_sha256(version_dir / "train.jsonl"),
        "resolved_validation_split_sha256": file_sha256(version_dir / "validation.jsonl"),
        "resolved_test_split_sha256": file_sha256(version_dir / "test.jsonl"),
        "resolved_encoder_type": encoder["encoder_type"],
        "resolved_tokenizer_or_encoder_path": encoder["path"],
        "resolved_tokenizer_or_encoder_sha256": encoder["sha256"],
        "resolved_vocab_or_value_space_size": encoder["vocab_size"],
        "resolved_label_space": "0..9" if task == "lra_listops" else "not_applicable",
        "resolved_loss_type": loss_type,
        "resolved_model_family": "probe_transformer_encoder_readout",
        "resolved_layers": model["layers"],
        "resolved_d_model": model["d_model"],
        "resolved_heads": model["heads"],
        "resolved_ffn_dim": model["ffn_dim"],
        "resolved_dropout": model["dropout"],
        "resolved_parameter_count": int(param_count_estimate),
        "resolved_attention_backend": "auto_split",
        "resolved_graph_id": graph["artifact"]["graph_id"],
        "resolved_graph_seed": 0,
        "resolved_graph_generation_algorithm": "zigzag_v08_task_parameter_selection",
        "resolved_graph_block_size": B,
        "resolved_graph_num_blocks_or_nodes": int(T // B),
        "resolved_graph_degree_or_budget": d,
        "resolved_graph_max_parallel_edges_per_block_pair": "not_capped",
        "resolved_B_alias_if_applicable": B,
        "resolved_q_alias_if_applicable": int(T // B),
        "resolved_d_alias_if_applicable": d,
        "resolved_required_methods": REQUIRED_METHODS,
        "resolved_optional_methods": OPTIONAL_METHODS,
        "resolved_seeds": [0],
        "resolved_optimizer": "adamw",
        "resolved_learning_rate": 3e-4,
        "resolved_base_learning_rate": 3e-4,
        "resolved_lr_scheduler": "cosine",
        "resolved_warmup_ratio": 0.05,
        "resolved_warmup_steps": max(1, int(round(main_steps * 0.05))),
        "resolved_min_lr_ratio": 0.1,
        "resolved_min_learning_rate": 3e-5,
        "resolved_weight_decay": 0.01,
        "resolved_grad_clip_norm": 1.0,
        "resolved_batch_size": batch["batch_size"],
        "resolved_gradient_accumulation_steps": batch["gradient_accumulation_steps"],
        "resolved_effective_batch_size": effective_batch,
        "resolved_eval_batch_size": batch["eval_batch_size"],
        "resolved_train_budget_unit": "steps",
        "resolved_train_budget_value": main_steps,
        "resolved_steps_planned_if_step_budget": main_steps,
        "resolved_epochs_planned_if_epoch_budget": "not_applicable",
        "resolved_log_every": log_every,
        "resolved_log_step_policy": "step_1_every_log_every_and_final_step",
        "resolved_min_logged_train_step_count": main_steps if main_steps < 100 else math.ceil(main_steps * 0.01),
        "resolved_planned_logged_train_step_count": len({1, main_steps, *range(log_every, main_steps + 1, log_every)}),
        "resolved_log_coverage_ratio_min": 1.0 if main_steps < 100 else 0.01,
        "resolved_eval_every": max(1, main_steps // 4),
        "resolved_checkpoint_every": 0,
        "resolved_checkpoint_policy": "manifest_only_no_tensor_checkpoint",
        "resolved_validation_eval_budget": validation_rows,
        "resolved_test_eval_budget": test_rows,
        "resolved_oom_fallback_sequence": ["skip_optional_dense", "halve_batch_all_required_methods", "halve_steps_all_required_methods"],
        "smoke": {
            "steps": smoke_steps,
            "log_every": smoke_log_every,
            "eval_every": 1,
            "train_examples": min(train_rows, max(16, batch["batch_size"] * smoke_steps)),
            "validation_examples": min(validation_rows, 8),
            "test_examples": min(test_rows, 8),
        },
        "main": {
            "steps": main_steps,
            "log_every": log_every,
            "eval_every": max(1, main_steps // 4),
            "train_examples": train_rows,
            "validation_examples": validation_rows,
            "test_examples": test_rows,
            "log_eval_examples": min(validation_rows, 16),
        },
        "graph_artifacts": graph,
        "dataset_card_sha256": file_sha256(version_dir / "dataset_card.json"),
        "deployment_status_sha256": file_sha256(version_dir / "deployment_status.yaml"),
        "source_lock_sha256": file_sha256(version_dir / "source.lock"),
        "checksums_sha256": file_sha256(version_dir / "checksums.sha256"),
        "dataset_source": card.get("source", {}).get("source_url", "not_applicable"),
        "dataset_revision_or_hash": card.get("source", {}).get("commit_or_release", "not_applicable"),
    }
    return record


def write_reports(rows: list[dict], output_dir: Path) -> None:
    report = Path("reports/v08_phase4_task_parameter_selection_report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for row in rows:
        table_rows.append(
            "| {task} | {resolved_padded_sequence_length} | {resolved_graph_block_size}/{resolved_graph_degree_or_budget} | {resolved_layers}x{resolved_d_model} | {resolved_effective_batch_size} | {resolved_train_budget_value} | {primary_metric} |".format(**row)
        )
    report.write_text(
        f"""# v08 Phase 4 Task Parameter Selection 报告

## 结论

Phase 4 已在 Phase 1 数据审计、Phase 2 dry-run 和 Phase 3 A100 80GB 远端 readiness 基础上重新冻结 6 个 probe task 的参数，并生成 smoke/main 配置、字段契约、编码器和每任务 directed non-causal graph artifact。required methods 为 `{', '.join(REQUIRED_METHODS)}`；`dense` 暂列 optional，原因是 4k/8k 长上下文 dense reference 会显著挤占本轮主方法验证预算，v08 主线优先完成 theory-aligned zigzag_certified 与同预算 local/random_regular 对照。

| task | padded T | B/d | 模型 | effective batch | main steps | primary metric |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(table_rows)}

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
""",
        encoding="utf-8",
    )
    glossary = Path("reports/v08_parameter_glossary.md")
    glossary.write_text(
        """# v08 参数术语表

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
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-summary", type=Path, default=Path("outputs/probes_v08_data_audit/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probes_v08_parameter_selection"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = read_json(args.audit_summary)
    audit_rows = {row["task"]: row for row in audit["rows"]}
    rows = [build_task_record(task, audit_rows[task], args.output_dir) for task in SELECTED_PROBES]
    manifest = {
        "version": EXPERIMENT_VERSION,
        "phase": "phase4_task_parameter_selection",
        "timestamp_utc": utc_now(),
        "attention_contract": ATTENTION_CONTRACT,
        "causal": False,
        "graph_directionality": GRAPH_DIRECTIONALITY,
        "required_methods": REQUIRED_METHODS,
        "optional_methods": OPTIONAL_METHODS,
        "tasks": rows,
    }
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    write_json(configs_dir / "probes_v08_task_parameters.json", manifest)
    write_json(
        configs_dir / "probes_v08_smoke.json",
        {
            "version": EXPERIMENT_VERSION,
            "phase": "phase5_smoke",
            "profile": "smoke",
            "task_parameter_manifest": "configs/probes_v08_task_parameters.json",
            "output_root": "outputs/probes_v08_smoke",
            "tasks": [row["task"] for row in rows],
            "methods": REQUIRED_METHODS,
            "seeds": [0],
        },
    )
    write_json(
        configs_dir / "probes_v08_main.json",
        {
            "version": EXPERIMENT_VERSION,
            "phase": "phase6_main",
            "profile": "main",
            "task_parameter_manifest": "configs/probes_v08_task_parameters.json",
            "output_root": "outputs/probes_v08_main",
            "tasks": [row["task"] for row in rows],
            "methods": REQUIRED_METHODS,
            "seeds": [0],
        },
    )
    write_json(
        configs_dir / "probes_v08_result_field_contract.json",
        {
            "version": EXPERIMENT_VERSION,
            "field_count": len(FIELD_CONTRACT),
            "fields": FIELD_CONTRACT,
            "forbidden_spellings": ["noncaual", "non-casual"],
        },
    )
    flat_rows = [{key: _report_field(value) for key, value in row.items() if key != "graph_artifacts"} for row in rows]
    write_json(args.output_dir / "summary.json", {"status": "ok", "tasks": rows, "command": command_string()})
    write_csv(args.output_dir / "task_parameters.csv", flat_rows)
    write_jsonl(args.output_dir / "task_parameters.jsonl", flat_rows)
    write_command(args.output_dir / "command.sh")
    write_reports(rows, args.output_dir)


if __name__ == "__main__":
    main()
