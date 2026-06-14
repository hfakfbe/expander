from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from graph_diagnostics import CERTIFICATE_FIELDS, compute_shortcut_stats
from graph_structures import load_graph_artifact
from synthetic_mvp_core.artifacts import RESULT_FIELDS, method_certification_fields
from synthetic_mvp_core.io_utils import write_csv, write_json, write_jsonl


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_certificate(certificate: dict) -> dict:
    normalized = dict(certificate)
    if "certified" not in normalized and "rho_zigzag_certified" in normalized:
        normalized["certified"] = bool(normalized["rho_zigzag_certified"])
    if "rho_bound" not in normalized and "rho_zigzag_bound" in normalized:
        normalized["rho_bound"] = normalized["rho_zigzag_bound"]
    return normalized


def csv_fields(rows: list[dict], preferred: list[str] | None = None) -> list[str]:
    fields = list(preferred or [])
    seen = set(fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def graph_certified(certificate: dict) -> bool:
    if "certified" in certificate:
        return bool(certificate["certified"])
    return bool(certificate.get("rho_zigzag_certified", False))


def resolved_config(raw_config: dict, graph_artifact: dict, certificate: dict, seed: int | None) -> dict:
    resolved = copy.deepcopy(raw_config)
    attention = resolved.setdefault("attention", {})
    attention["block_size"] = int(graph_artifact["B"])
    attention["degree"] = int(graph_artifact["d"])
    attention["runtime_graph"] = {
        "graph_artifact": attention.get("graph_artifact", ""),
        "graph_id": graph_artifact.get("graph_id", ""),
        "graph_seed": graph_artifact.get("graph_seed", ""),
        "B": int(graph_artifact["B"]),
        "d": int(graph_artifact["d"]),
        "T": int(graph_artifact["T"]),
        "q": int(graph_artifact["q"]),
        "G_type": graph_artifact.get("G", {}).get("type", ""),
        "H_type": graph_artifact.get("H", {}).get("type", ""),
        "graph_certified": graph_certified(certificate),
    }
    if "rho_zigzag_bound" in certificate:
        attention["runtime_graph"]["rho_zigzag_bound"] = certificate["rho_zigzag_bound"]
    elif "rho_bound" in certificate:
        attention["runtime_graph"]["rho_bound"] = certificate["rho_bound"]
    if seed is not None:
        resolved.setdefault("train", {})["seeds"] = [int(seed)]
    return resolved


def copy_source_length(row: dict) -> int:
    if row.get("copy_source_length") not in {"", None}:
        return int(row["copy_source_length"])
    if row.get("N_train") not in {"", None}:
        return int(row["N_train"])
    if row.get("N_total") not in {"", None}:
        return (int(row["N_total"]) - 2) // 2
    raise KeyError("cannot infer copy source length from row; expected copy_source_length, N_train, or N_total")


def update_result_row(row: dict, shortcut_summary: dict, certificate: dict) -> dict:
    method = str(row.get("method", ""))
    cert_fields = method_certification_fields(
        method,
        certificate,
        str(row.get("multiplicity_mode", "boolean")),
    )
    updated = dict(row)
    updated.update(
        {
            "certified": cert_fields["certified"],
            "graph_certified": cert_fields["graph_certified"],
            "implementation_certified": cert_fields["implementation_certified"],
            "theory_aligned_method": cert_fields["theory_aligned_method"],
            "target_in_1hop_rate": shortcut_summary.get("target_in_1hop_rate", ""),
            "target_in_2hop_rate": shortcut_summary.get("target_in_2hop_rate", ""),
            "target_in_Lhop_rate": shortcut_summary.get("target_in_Lhop_rate", ""),
            "average_shortest_path": shortcut_summary.get("average_shortest_path", ""),
            "unreachable_rate": shortcut_summary.get("unreachable_rate", ""),
        }
    )
    if "rho_zigzag_bound" in certificate:
        updated["rho_zigzag_bound"] = certificate["rho_zigzag_bound"]
        updated["rho_zigzag_certified"] = graph_certified(certificate)
    return updated


def repair_run(
    run_dir: Path,
    raw_config: dict,
    graph_artifact: dict,
    certificate: dict,
    recompute_shortcuts: bool,
) -> tuple[list[dict], list[dict], dict | None]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return [], [], None
    summary = read_json(summary_path)
    rows = list(summary.get("results", []))
    if not rows:
        return [], [], summary.get("result")
    first = rows[0]
    if first.get("status") != "ok":
        return rows, read_jsonl(run_dir / "shortcut_diagnostics.jsonl"), summary.get("result")

    shortcut_rows = read_jsonl(run_dir / "shortcut_diagnostics.jsonl")
    shortcut_summary = shortcut_rows[0] if shortcut_rows else {}
    if recompute_shortcuts:
        method = str(first["method"])
        seed = int(first["seed"])
        source_len = copy_source_length(first)
        T = int(first.get("T") or first.get("N_total") or graph_artifact["T"])
        B = int(first.get("B") or graph_artifact["B"])
        d = int(first.get("d") or graph_artifact["d"])
        layers = int(first["layers"])
        shortcut_rows = compute_shortcut_stats(
            method=method,
            N=source_len,
            T=T,
            B=B,
            d=d,
            seed=seed,
            layers=layers,
            graph_artifact=graph_artifact,
        )
        write_csv(run_dir / "shortcut_diagnostics.csv", shortcut_rows, csv_fields(shortcut_rows))
        write_jsonl(run_dir / "shortcut_diagnostics.jsonl", shortcut_rows)
        shortcut_summary = next(
            (row for row in shortcut_rows if row.get("mask_scope") == "causal_effective"),
            shortcut_rows[0],
        )

    updated_rows = [update_result_row(row, shortcut_summary, certificate) for row in rows]
    write_csv(run_dir / "results.csv", updated_rows, csv_fields(updated_rows, RESULT_FIELDS))
    write_jsonl(run_dir / "results.jsonl", updated_rows)

    run_raw_config = copy.deepcopy(raw_config)
    run_resolved_config = resolved_config(run_raw_config, graph_artifact, certificate, int(first["seed"]))
    write_json(run_dir / "raw_config_snapshot.json", run_raw_config)
    write_json(run_dir / "resolved_config_snapshot.json", run_resolved_config)
    write_json(run_dir / "config_snapshot.json", run_resolved_config)

    result = summary.get("result")
    if isinstance(result, dict):
        result["shortcut_diagnostics"] = shortcut_rows
        result["evals"] = updated_rows
    summary["results"] = updated_rows
    summary["result"] = result
    summary.setdefault("repair_notes", []).append("repaired by scripts/repair_copy_outputs.py")
    write_json(summary_path, summary)
    return updated_rows, shortcut_rows, result


def repair_root(
    output_dir: Path,
    config_path: Path,
    graph_artifact_path: Path,
    main_seeds: set[int],
    recompute_shortcuts: bool,
) -> None:
    raw_config_path = output_dir / "raw_config_snapshot.json"
    resolved_snapshot_path = output_dir / "resolved_config_snapshot.json"
    existing_snapshot_path = output_dir / "config_snapshot.json"
    if raw_config_path.exists():
        raw_config = read_json(raw_config_path)
    elif existing_snapshot_path.exists():
        raw_config = read_json(existing_snapshot_path)
    elif resolved_snapshot_path.exists():
        raw_config = read_json(resolved_snapshot_path)
    else:
        raw_config = read_json(config_path)

    graph_artifact = load_graph_artifact(graph_artifact_path)
    certificate = normalized_certificate(dict(graph_artifact.get("certificate", {})))
    all_records: list[dict] = []
    all_shortcuts: list[dict] = []
    method_results: list[dict] = []
    extra_run_ids: list[str] = []

    for run_dir in sorted(path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("train_")):
        rows, shortcut_rows, result = repair_run(
            run_dir,
            raw_config,
            graph_artifact,
            certificate,
            recompute_shortcuts=recompute_shortcuts,
        )
        if not rows:
            continue
        seed = int(rows[0].get("seed", -1))
        if seed in main_seeds:
            all_records.extend(rows)
            all_shortcuts.extend(shortcut_rows)
            if result is not None:
                method_results.append(result)
        else:
            extra_run_ids.append(run_dir.name)

    main_raw_config = copy.deepcopy(read_json(config_path))
    main_raw_config.setdefault("train", {})["seeds"] = sorted(main_seeds)
    main_resolved_config = resolved_config(main_raw_config, graph_artifact, certificate, None)
    write_json(output_dir / "raw_config_snapshot.json", raw_config)
    write_json(output_dir / "resolved_config_snapshot.json", main_resolved_config)
    write_json(output_dir / "config_snapshot.json", main_resolved_config)
    write_csv(output_dir / "results.csv", all_records, csv_fields(all_records, RESULT_FIELDS))
    write_jsonl(output_dir / "results.jsonl", all_records)
    write_csv(output_dir / "phase5_results.csv", all_records, csv_fields(all_records, RESULT_FIELDS))
    write_jsonl(output_dir / "phase5_results.jsonl", all_records)
    write_csv(output_dir / "phase3_results.csv", all_records, csv_fields(all_records, RESULT_FIELDS))
    write_jsonl(output_dir / "phase3_results.jsonl", all_records)
    if all_shortcuts:
        write_csv(output_dir / "shortcut_diagnostics.csv", all_shortcuts, csv_fields(all_shortcuts))
        write_jsonl(output_dir / "shortcut_diagnostics.jsonl", all_shortcuts)
    write_csv(output_dir / "graph_diagnostics.csv", [certificate], csv_fields([certificate], CERTIFICATE_FIELDS))
    write_json(output_dir / "graph_certificate.json", certificate)

    metrics_lines: list[str] = []
    for row in all_records:
        metrics_path = Path(str(row.get("metrics_path", "")))
        if metrics_path.exists():
            metrics_lines.extend(metrics_path.read_text(encoding="utf-8").splitlines())
    if metrics_lines:
        (output_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    write_json(
        output_dir / "summary.json",
        {
            "status": "ok" if all_records and all(row.get("status") == "ok" for row in all_records) else "failed",
            "config": main_resolved_config,
            "raw_config_snapshot_path": str(output_dir / "raw_config_snapshot.json"),
            "resolved_config_snapshot_path": str(output_dir / "resolved_config_snapshot.json"),
            "results": all_records,
            "method_results": method_results,
            "extra_runs_preserved_not_in_main_table": extra_run_ids,
            "repair_notes": [
                "shortcut diagnostics recomputed with within-L-hop semantics"
                if recompute_shortcuts
                else "shortcut diagnostics preserved from existing files",
                "root main result table restricted to requested main seeds",
                "certified is a backward-compatible alias for theory_aligned_method",
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair completed copy experiment outputs without rerunning training."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graph-artifact", type=Path, required=True)
    parser.add_argument("--main-seeds", default="0")
    parser.add_argument(
        "--skip-shortcuts",
        action="store_true",
        help="Preserve existing shortcut diagnostics instead of recomputing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_seeds = {int(seed.strip()) for seed in args.main_seeds.split(",") if seed.strip()}
    repair_root(
        args.output_dir,
        args.config,
        args.graph_artifact,
        main_seeds,
        recompute_shortcuts=not args.skip_shortcuts,
    )


if __name__ == "__main__":
    main()
