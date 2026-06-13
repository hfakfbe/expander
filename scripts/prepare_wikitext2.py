from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

from wikitext2_utils import ByteTokenizer, file_sha256, write_command, write_json, write_jsonl


SPLITS = ["train", "validation", "test"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPS_DIR = PROJECT_ROOT / ".deps"
if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_candidates(config: dict) -> list[dict]:
    dataset = config["dataset"]
    candidates = [dict(dataset["preferred_source"])]
    fallback = dataset.get("fallback_source")
    if fallback:
        candidates.append(dict(fallback))
    return candidates


def dataset_revision(path: str) -> str:
    try:
        response = requests.get(f"https://huggingface.co/api/datasets/{path}", timeout=20)
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("sha") or payload.get("lastModified") or "")
    except Exception:
        return ""


def fetch_rows(source: dict, split: str, text_field: str) -> tuple[list[dict], int]:
    rows: list[dict] = []
    total = None
    offset = 0
    page_size = 100
    while total is None or offset < total:
        params = {
            "dataset": source["path"],
            "split": split,
            "offset": offset,
            "length": page_size,
        }
        if source.get("name"):
            params["config"] = source["name"]
        response = requests.get("https://datasets-server.huggingface.co/rows", params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        total = int(payload["num_rows_total"])
        for row in payload["rows"]:
            value = row["row"].get(text_field)
            if value is None:
                raise ValueError(f"missing text field {text_field!r} in split {split!r}")
            rows.append({"row_idx": int(row["row_idx"]), text_field: str(value)})
        offset += int(payload.get("num_rows_per_page") or page_size)
    return rows, int(total or len(rows))


def parquet_url(source: dict, split: str) -> str | None:
    if source.get("path") != "Salesforce/wikitext" or source.get("name") != "wikitext-2-raw-v1":
        return None
    return (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    )


def fetch_parquet_rows(source: dict, split: str, text_field: str, output_dir: Path) -> list[dict]:
    url = parquet_url(source, split)
    if url is None:
        raise ValueError("no parquet URL known for source")
    import pyarrow.parquet as pq

    parquet_dir = output_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{split}.parquet"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    parquet_path.write_bytes(response.content)
    table = pq.read_table(parquet_path)
    if text_field not in table.column_names:
        raise ValueError(f"missing text field {text_field!r} in parquet split {split!r}")
    values = table[text_field].to_pylist()
    return [{"row_idx": idx, text_field: str(value)} for idx, value in enumerate(values)]


def fetch_source_split(source: dict, split: str, text_field: str, output_dir: Path) -> list[dict]:
    try:
        return fetch_parquet_rows(source, split, text_field, output_dir)
    except Exception:
        rows, expected_total = fetch_rows(source, split, text_field)
        if len(rows) != expected_total:
            raise ValueError(f"{split} fetched {len(rows)} rows, expected {expected_total}")
        return rows


def split_stats(rows: list[dict], text_field: str, tokenizer: ByteTokenizer) -> dict:
    lengths = [len(str(row[text_field])) for row in rows]
    nonempty = [str(row[text_field]) for row in rows if str(row[text_field]).strip()]
    token_lengths = [len(tokenizer.encode(text, add_eos=True)) for text in nonempty]
    return {
        "rows": len(rows),
        "nonempty_rows": len(nonempty),
        "empty_rows": len(rows) - len(nonempty),
        "empty_line_rate": (len(rows) - len(nonempty)) / max(len(rows), 1),
        "longest_text_length": max(lengths) if lengths else 0,
        "tokenized_length_min": min(token_lengths) if token_lengths else 0,
        "tokenized_length_mean": statistics.fmean(token_lengths) if token_lengths else 0.0,
        "tokenized_length_max": max(token_lengths) if token_lengths else 0,
    }


def prepare(config: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    text_field = config["dataset"].get("text_field", "text")
    errors: list[str] = []
    selected_source = None
    split_rows: dict[str, list[dict]] = {}

    for source in source_candidates(config):
        try:
            candidate_rows = {}
            for split in SPLITS:
                candidate_rows[split] = fetch_source_split(source, split, text_field, output_dir)
            selected_source = source
            split_rows = candidate_rows
            break
        except Exception as exc:
            errors.append(f"{source.get('path')}:{source.get('name', '')}: {exc!r}")

    if selected_source is None:
        raise RuntimeError("all WikiText2 sources failed: " + "; ".join(errors))

    split_summaries = {}
    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        split_summaries[split] = split_stats(rows, text_field, tokenizer)
        split_summaries[split]["sha256"] = file_sha256(output_dir / f"{split}.jsonl")

    readiness = {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "wikitext2",
        "dataset_source": selected_source.get("path", ""),
        "dataset_config": selected_source.get("name", ""),
        "dataset_revision_or_hash": dataset_revision(selected_source.get("path", "")),
        "text_field": text_field,
        "required_splits": SPLITS,
        "splits": split_summaries,
        "source_errors_before_success": errors,
        "tokenizer": {
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
    }
    validation = config.get("validation", {})
    for split in validation.get("required_splits", SPLITS):
        if split not in split_summaries:
            raise ValueError(f"missing split {split}")
        min_rows = int(validation.get("min_rows", {}).get(split, 1))
        if split_summaries[split]["rows"] < min_rows:
            raise ValueError(f"split {split} has too few rows")
        max_empty = float(validation.get("max_empty_line_rate", 1.0))
        if split_summaries[split]["empty_line_rate"] > max_empty:
            raise ValueError(f"split {split} empty line rate too high")

    nonempty_train = [row[text_field] for row in split_rows["train"] if row[text_field].strip()]
    tokenized_smoke = {
        "tokenizer": readiness["tokenizer"],
        "examples": [
            {
                "text_preview": text[:120],
                "token_ids_prefix": tokenizer.encode(text, add_eos=True)[:32],
                "tokenized_length": len(tokenizer.encode(text, add_eos=True)),
            }
            for text in nonempty_train[:5]
        ],
    }
    dataset_info = {
        "dataset": "wikitext2",
        "variant": "wikitext-2-raw-v1",
        "source": readiness["dataset_source"],
        "config": readiness["dataset_config"],
        "revision_or_hash": readiness["dataset_revision_or_hash"],
        "files": {split: f"{split}.jsonl" for split in SPLITS},
    }
    write_json(output_dir / "dataset_info.json", dataset_info)
    write_json(output_dir / "data_readiness.json", readiness)
    write_json(output_dir / "tokenized_smoke.json", tokenized_smoke)
    return readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    output_dir = args.output_dir or Path(config["dataset"]["output_dir"])
    write_json(output_dir / "config_snapshot.json", config)
    write_command(output_dir / "command.sh")
    readiness = prepare(config, output_dir)
    print(json.dumps({"status": readiness["status"], "source": readiness["dataset_source"]}), flush=True)


if __name__ == "__main__":
    main()
