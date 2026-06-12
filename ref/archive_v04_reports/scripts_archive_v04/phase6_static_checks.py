import ast
import json
from pathlib import Path


def functions_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def classes_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def main() -> None:
    root = Path.cwd()
    synthetic = root / "scripts" / "synthetic_mvp.py"
    run_model = root / "code" / "lra-benchmarks" / "run_model.py"
    datasets = root / "code" / "lra-benchmarks" / "lra_datasets.py"
    checks = {
        "synthetic_functions": {
            name: name in functions_in_file(synthetic)
            for name in [
                "build_attention_mask",
                "build_cross_mask",
                "local_cross_attention",
                "local_blockpair_attention",
                "run_mask_tests",
            ]
        },
        "base_run_model_functions": {
            name: name in functions_in_file(run_model)
            for name in ["get_model", "transformers_collator", "accuracy_score"]
        },
        "base_dataset_classes": {
            name: name in classes_in_file(datasets)
            for name in ["ListOpsDataset", "ImdbDataset"]
        },
        "base_attention_smoke_classes": {
            name: name in classes_in_file(root / "scripts" / "base_attention_smoke.py")
            for name in ["BertSparseSelfAttention"]
        },
    }
    status = "ok" if all(all(group.values()) for group in checks.values()) else "failed"
    result = {"status": status, "checks": checks}
    output = root / "outputs" / "phase6_static_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
