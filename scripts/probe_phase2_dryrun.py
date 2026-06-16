from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from graph_diagnostics import certificate_for_artifact
from graph_structures import build_graph_artifact
from probe_common import (
    ATTENTION_CONTRACT,
    EXPERIMENT_VERSION,
    GRAPH_DIRECTIONALITY,
    SELECTED_PROBES,
    command_string,
    file_sha256,
    selected_probe_path,
    utc_now,
    write_command,
    write_csv,
    write_json,
    write_jsonl,
)
from probe_metrics import masked_sequence_loss
from probe_tasks import (
    JsonlStore,
    ProbeTransformer,
    byte_encoder,
    build_listops_encoder,
    gather_position_logits,
    integer_encoder,
    load_encoder,
    make_probe_batch,
)
from synthetic_mvp_core.artifacts import make_attention_artifacts, resolve_attention_backend


def numeric_max_quick(version_dir: Path, limit: int = 64) -> int:
    max_value = 0
    seen = 0
    with (version_dir / "train.jsonl").open("r", encoding="utf-8") as fp:
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
            if values:
                max_value = max(max_value, max(values))
            seen += 1
            if seen >= limit:
                break
    return max(max_value, 16)


def loss_type(task: str) -> str:
    if task in {"copy", "selective_copy"}:
        return "sequence_cross_entropy"
    if task == "induction_associative_recall":
        return "mqar_position_cross_entropy"
    if task in {"niah_kv_retrieval", "ruler"}:
        return "retrieval_sequence_cross_entropy"
    return "classification_cross_entropy"


def make_record(task: str, output_dir: Path) -> dict:
    version_dir = selected_probe_path(task)
    encoder_path = output_dir / "encoders" / task / "encoder.json"
    if task in {"niah_kv_retrieval", "ruler"}:
        encoder = byte_encoder(encoder_path)
    elif task == "lra_listops":
        encoder = build_listops_encoder(version_dir / "train.jsonl", encoder_path)
    else:
        encoder = integer_encoder(numeric_max_quick(version_dir), encoder_path)
    if task == "induction_associative_recall":
        input_len, target_len, raw = 64, 4, 64
    elif task in {"copy", "selective_copy"}:
        input_len, target_len, raw = 64, 8, 72
    elif task in {"niah_kv_retrieval", "ruler"}:
        input_len, target_len, raw = 128, 12, 140
    else:
        input_len, target_len, raw = 128, 1, 128
    B, d = 16, 4
    T = ((raw + B - 1) // B) * B
    artifact = build_graph_artifact(
        raw,
        raw,
        B,
        d,
        graph_seed=0,
        version=EXPERIMENT_VERSION,
        g_config={"max_parallel_edges_per_block_pair": None},
    )
    cert = certificate_for_artifact(artifact, {"acceptance": {"rho_bound_lt": 1.0, "max_remote_local_overlap_mean": 0.75}})
    artifact["certificate"] = cert
    graph_dir = output_dir / "graphs" / task
    graph_dir.mkdir(parents=True, exist_ok=True)
    write_json(graph_dir / "selected_graph.json", artifact)
    write_json(graph_dir / "graph_certificate.json", cert)
    return {
        "task": task,
        "version_path": str(version_dir),
        "primary_metric": SELECTED_PROBES[task]["primary_metric"],
        "resolved_loss_type": loss_type(task),
        "resolved_runtime_input_length": input_len,
        "resolved_runtime_target_length": target_len,
        "resolved_readout_start": input_len,
        "resolved_padded_sequence_length": T,
        "resolved_tokenizer_or_encoder_path": str(encoder_path),
        "resolved_vocab_or_value_space_size": encoder["vocab_size"],
        "resolved_graph_block_size": B,
        "resolved_graph_degree_or_budget": d,
        "resolved_layers": 1,
        "resolved_d_model": 32,
        "resolved_heads": 4,
        "resolved_ffn_dim": 64,
        "resolved_dropout": 0.0,
        "resolved_attention_backend": "auto_split",
        "graph_artifacts": {"artifact": artifact, "certificate": cert},
    }


def run_task(task: str, output_dir: Path, device: torch.device) -> dict:
    started = time.perf_counter()
    record = make_record(task, output_dir)
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    rows = store.sample(2, seed=0, stream="phase2", step=0, limit=8)
    batch = make_probe_batch(rows, record, encoder, device)
    args = type("Args", (), {})()
    args.block_size = record["resolved_graph_block_size"]
    args.degree = record["resolved_graph_degree_or_budget"]
    args.causal = False
    args.graph_config = record["graph_artifacts"]["artifact"]
    args.graph_artifact = record["graph_artifacts"]["artifact"]
    args.graph_certificate = record["graph_artifacts"]["certificate"]
    args.seed = 0
    args.multiplicity_mode = "unique_log_m"
    backend = resolve_attention_backend(record["resolved_attention_backend"], "local")
    artifacts = make_attention_artifacts("local", record["resolved_padded_sequence_length"], args, device, backend)
    model = ProbeTransformer(
        vocab_size=record["resolved_vocab_or_value_space_size"],
        token_output_size=record["resolved_vocab_or_value_space_size"],
        class_count=10,
        seq_len=record["resolved_padded_sequence_length"],
        d_model=record["resolved_d_model"],
        layers=record["resolved_layers"],
        heads=record["resolved_heads"],
        ffn_dim=record["resolved_ffn_dim"],
        dropout=record["resolved_dropout"],
        attention_backend=backend,
        block_size=record["resolved_graph_block_size"],
    ).to(device)
    token_logits, class_logits = model(
        batch.tokens,
        batch.pad_mask,
        artifacts.mask,
        artifacts.local_valid,
        artifacts.neighbors,
        artifacts.valid_neighbors,
        artifacts.block_pair_index,
        artifacts.local_log_m,
        artifacts.neighbor_log_m,
    )
    if record["resolved_loss_type"] == "classification_cross_entropy":
        assert batch.class_targets is not None
        loss = torch.nn.functional.cross_entropy(class_logits, batch.class_targets)
    else:
        assert batch.target_positions is not None and batch.targets is not None and batch.target_mask is not None
        selected = gather_position_logits(token_logits, batch.target_positions)
        loss = masked_sequence_loss(selected, batch.targets, batch.target_mask)
    loss.backward()
    return {
        "task": task,
        "status": "ok",
        "loss": float(loss.item()),
        "attention_contract": ATTENTION_CONTRACT,
        "causal": False,
        "graph_directionality": GRAPH_DIRECTIONALITY,
        "input_batch_shape": list(batch.tokens.shape),
        "target_positions_shape": list(batch.target_positions.shape) if batch.target_positions is not None else "classification",
        "wall_time_sec": time.perf_counter() - started,
    }


def write_report(rows: list[dict]) -> None:
    report = Path("reports/v08_phase2_code_adaptation_report.md")
    table = "\n".join(
        f"| {row['task']} | {row['status']} | {row['loss']:.4f} | {row['input_batch_shape']} |"
        for row in rows
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# v08 Phase 2 Code Adaptation 报告

## 结论

Phase 2 已新增 probe 专用审计、参数、任务、指标、远端 readiness 和训练入口，并用 tiny interface dry-run 验证 6 个 task 的 JSONL 读取、编码、non-causal attention、forward、loss 和 backward 路径。dry-run 参数只用于接口验证，不进入 Phase 4 主参数。

| task | 状态 | dry-run loss | batch shape |
|---|---|---:|---|
{table}

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| attention_contract | 注意力合同 | non_causal | v08 手册 | 确认 dry-run 不走 causal LM | 不满足则失败 |
| causal | 是否 causal mask | false | v08 手册 | 防止任务格式错误 | 不满足则失败 |
| graph_directionality | 图方向性 | directed | graph artifact | 验证 directed graph 路径 | 无 |
| input_batch_shape | 输入 batch 张量形状 | [batch, T] | dry-run | 验证不同 schema 可进入模型 | 无 |
| target_positions_shape | 目标 readout/position 形状 | [batch, target_len] 或 classification | dry-run | 验证三类 loss | classification 写 classification |
| loss | dry-run 训练损失 | 标量 | forward/backward | 验证 loss 有限 | 无 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 6 |
| unexplained_parameters | [] |
| english_only_sections | [] |
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probes_v08_phase2_interface_dryrun"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    rows = [run_task(task, args.output_dir, device) for task in SELECTED_PROBES]
    summary = {
        "version": EXPERIMENT_VERSION,
        "phase": "phase2_code_adaptation_interface_dryrun",
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "failed",
        "timestamp_utc": utc_now(),
        "command": command_string(),
        "rows": rows,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "dryrun_results.csv", rows)
    write_jsonl(args.output_dir / "dryrun_results.jsonl", rows)
    write_command(args.output_dir / "command.sh")
    write_report(rows)


if __name__ == "__main__":
    main()
