from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Iterable

import requests

from v07_artifacts import file_sha256, git_commit, utc_now, write_json
from wikitext2_utils import write_command, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPS_DIR = PROJECT_ROOT / ".deps"
if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_lines(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def dataset_revision(path: str) -> str:
    try:
        response = requests.get(f"https://huggingface.co/api/datasets/{path}", timeout=20)
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("sha") or payload.get("lastModified") or "")
    except Exception:
        return ""


def source_candidates(config: dict) -> list[dict]:
    task = config.get("task", {})
    candidates = [
        {
            "path": task.get("preferred_source", "Salesforce/wikitext"),
            "name": task.get("dataset", "wikitext-103-raw-v1"),
        }
    ]
    fallback = task.get("fallback_source")
    if fallback:
        if isinstance(fallback, dict):
            candidates.append(dict(fallback))
        else:
            candidates.append({"path": str(fallback), "name": ""})
    return candidates


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
        response = requests.get("https://datasets-server.huggingface.co/rows", params=params, timeout=90)
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


def fetch_rows_with_datasets(source: dict, split: str, text_field: str, cache_dir: Path) -> tuple[list[dict], int]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is not available") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    args = [source["path"]]
    if source.get("name"):
        args.append(source["name"])
    dataset = load_dataset(*args, split=split, cache_dir=str(cache_dir))
    rows: list[dict] = []
    for row_idx, row in enumerate(dataset):
        value = row.get(text_field)
        if value is None:
            raise ValueError(f"missing text field {text_field!r} in split {split!r}")
        rows.append({"row_idx": int(row_idx), text_field: str(value)})
    return rows, len(rows)


def fetch_required_splits(config: dict, raw_dir: Path) -> tuple[dict, dict, list[str]]:
    task = config["task"]
    text_field = task.get("text_field", "text")
    splits = [task.get("train_split", "train"), task.get("test_split", "test")]
    errors: list[str] = []
    for source in source_candidates(config):
        try:
            out = {}
            totals = {}
            for split in splits:
                try:
                    rows, expected = fetch_rows_with_datasets(
                        source,
                        split,
                        text_field,
                        raw_dir / ".hf_cache",
                    )
                except Exception as datasets_exc:
                    errors.append(
                        f"{source.get('path')}:{source.get('name', '')}:{split}:"
                        f" datasets={datasets_exc!r}; falling back to rows API"
                    )
                    rows, expected = fetch_rows(source, split, text_field)
                out[split] = rows
                totals[split] = expected
            return {"source": source, "rows": out, "totals": totals}, {
                "text_field": text_field,
                "splits": splits,
            }, errors
        except Exception as exc:
            errors.append(f"{source.get('path')}:{source.get('name', '')}: {exc!r}")
    raise RuntimeError("all WikiText sources failed: " + "; ".join(errors))


def train_byte_level_bpe(config: dict, texts: list[str], tokenizer_dir: Path) -> dict:
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except Exception as exc:
        raise RuntimeError(
            "tokenizers package is required for v07 byte_level_bpe tokenizer training"
        ) from exc

    tok_cfg = config["tokenizer"]
    tokenizer = Tokenizer(BPE(unk_token=tok_cfg.get("unk_token", "<unk>")))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    special_tokens = list(tok_cfg.get("special_tokens", ["<pad>", "<eos>", "<unk>"]))
    trainer = BpeTrainer(
        vocab_size=int(tok_cfg.get("vocab_size", 32000)),
        min_frequency=int(tok_cfg.get("min_frequency", 2)),
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    eos_token = tok_cfg.get("eos_token", "<eos>")
    eos_id = tokenizer.token_to_id(eos_token)
    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos_token}",
        special_tokens=[(eos_token, int(eos_id))],
    )
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config_payload = {
        "tokenizer_algorithm": "byte_level_bpe",
        "tokenizer_train_split": tok_cfg.get("train_from_split", "train"),
        "tokenizer_vocab_size": int(tok_cfg.get("vocab_size", 32000)),
        "tokenizer_min_frequency": int(tok_cfg.get("min_frequency", 2)),
        "special_tokens": special_tokens,
        "pad_token": tok_cfg.get("pad_token", "<pad>"),
        "eos_token": eos_token,
        "unk_token": tok_cfg.get("unk_token", "<unk>"),
        "pad_token_id": tokenizer.token_to_id(tok_cfg.get("pad_token", "<pad>")),
        "eos_token_id": eos_id,
        "unk_token_id": tokenizer.token_to_id(tok_cfg.get("unk_token", "<unk>")),
        "vocab_size": tokenizer.get_vocab_size(),
    }
    write_json(tokenizer_dir / "tokenizer_config.json", config_payload)
    training_payload = {
        "timestamp_utc": utc_now(),
        "train_split": tok_cfg.get("train_from_split", "train"),
        "used_test_split": False,
        "input_text_count": len(texts),
        "algorithm": "byte_level_bpe",
    }
    write_json(tokenizer_dir / "tokenizer_training.json", training_payload)
    return {
        "tokenizer": tokenizer,
        "tokenizer_path": tokenizer_path,
        "tokenizer_config": config_payload,
        "tokenizer_training": training_payload,
    }


def encode_blocks(tokenizer, texts: list[str], sequence_length: int, append_eos: bool) -> tuple[list[list[int]], int]:
    token_ids: list[int] = []
    eos_id = tokenizer.token_to_id("<eos>")
    for text in texts:
        if not text.strip():
            continue
        encoded = tokenizer.encode(text).ids
        if not append_eos and encoded and eos_id is not None and encoded[-1] == eos_id:
            encoded = encoded[:-1]
        token_ids.extend(int(v) for v in encoded)
    usable = (len(token_ids) // int(sequence_length)) * int(sequence_length)
    blocks = [
        token_ids[idx : idx + int(sequence_length)]
        for idx in range(0, usable, int(sequence_length))
    ]
    return blocks, len(token_ids)


def prepare(config: dict, output_dir: Path) -> dict:
    total_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(config["task"].get("raw_dataset_dir", "datasets/wikitext_v07_raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "raw_config_snapshot.json", config)

    download_start = time.perf_counter()
    fetched, fetch_meta, source_errors = fetch_required_splits(config, raw_dir)
    download_wall = time.perf_counter() - download_start
    source = fetched["source"]
    rows_by_split = fetched["rows"]
    text_field = fetch_meta["text_field"]
    train_split = config["task"].get("train_split", "train")
    test_split = config["task"].get("test_split", "test")
    raw_files = {}
    split_stats = {}
    for split, rows in rows_by_split.items():
        raw_path = raw_dir / f"{split}.jsonl"
        write_text_lines(raw_path, rows)
        raw_files[split] = raw_path
        nonempty = [row[text_field] for row in rows if str(row[text_field]).strip()]
        split_stats[split] = {
            "rows": len(rows),
            "nonempty_rows": len(nonempty),
            "sha256": file_sha256(raw_path),
            "path": str(raw_path),
        }

    train_texts = [row[text_field] for row in rows_by_split[train_split] if str(row[text_field]).strip()]
    tokenizer_start = time.perf_counter()
    tokenizer_dir = output_dir / config["tokenizer"].get("output_subdir", "artifacts/tokenizer")
    tokenizer_bundle = train_byte_level_bpe(config, train_texts, tokenizer_dir)
    tokenizer_wall = time.perf_counter() - tokenizer_start
    tokenizer = tokenizer_bundle["tokenizer"]
    tokenizer_path = tokenizer_bundle["tokenizer_path"]
    tokenizer_sha = file_sha256(tokenizer_path)

    tokenization_start = time.perf_counter()
    tokenized_dir = output_dir / config.get("tokenize", {}).get("output_subdir", "tokenized")
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    sequence_length = int(config["task"].get("sequence_length", 1024))
    append_eos = bool(config.get("tokenize", {}).get("append_eos", True))
    train_blocks, train_token_count = encode_blocks(tokenizer, train_texts, sequence_length, append_eos)
    test_texts = [row[text_field] for row in rows_by_split[test_split] if str(row[text_field]).strip()]
    test_blocks, test_token_count = encode_blocks(tokenizer, test_texts, sequence_length, append_eos)
    train_path = tokenized_dir / "train_blocks.jsonl"
    test_path = tokenized_dir / "test_blocks.jsonl"
    write_jsonl(train_path, [{"block_id": idx, "input_ids": block} for idx, block in enumerate(train_blocks)])
    write_jsonl(test_path, [{"block_id": idx, "input_ids": block} for idx, block in enumerate(test_blocks)])
    tokenization_wall = time.perf_counter() - tokenization_start

    if not train_blocks or not test_blocks:
        raise RuntimeError("tokenized train/test blocks must both be non-empty")

    revision = dataset_revision(source.get("path", ""))
    summary_common = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "dataset": config["task"].get("dataset", "wikitext-103-raw-v1"),
        "dataset_source": source.get("path", ""),
        "dataset_config": source.get("name", ""),
        "dataset_revision_or_hash": revision,
        "dataset_cache_or_local_path": str(raw_dir),
        "train_nonempty_rows": split_stats[train_split]["nonempty_rows"],
        "test_nonempty_rows": split_stats[test_split]["nonempty_rows"],
        "tokenizer_algorithm": "byte_level_bpe",
        "tokenizer_train_split": "train",
        "tokenizer_vocab_size": int(config["tokenizer"].get("vocab_size", 32000)),
        "tokenizer_min_frequency": int(config["tokenizer"].get("min_frequency", 2)),
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_path": str(tokenizer_path),
        "tokenized_train_path": str(train_path),
        "tokenized_train_sha256": file_sha256(train_path),
        "tokenized_test_path": str(test_path),
        "tokenized_test_sha256": file_sha256(test_path),
        "train_token_count": train_token_count,
        "train_block_count": len(train_blocks),
        "test_token_count": test_token_count,
        "test_block_count": len(test_blocks),
        "sequence_length": sequence_length,
        "data_download_wall_time_sec": download_wall,
        "tokenizer_train_wall_time_sec": tokenizer_wall,
        "tokenization_wall_time_sec": tokenization_wall,
        "total_wall_time_sec": time.perf_counter() - total_start,
        "source_errors_before_success": source_errors,
        "git_commit": git_commit(PROJECT_ROOT),
        "command": shlex.join([sys.executable, *sys.argv]),
    }
    readiness = {
        **summary_common,
        "splits": split_stats,
        "tokenizer": tokenizer_bundle["tokenizer_config"],
    }
    tokenization_summary = {
        **summary_common,
        "tokenizer_config": tokenizer_bundle["tokenizer_config"],
        "tokenizer_training": tokenizer_bundle["tokenizer_training"],
    }
    resolved = dict(config)
    resolved["resolved_outputs"] = summary_common
    write_json(output_dir / "resolved_config_snapshot.json", resolved)
    write_json(output_dir / "config_snapshot.json", resolved)
    write_json(output_dir / "data_readiness.json", readiness)
    write_json(output_dir / "tokenization_summary.json", tokenization_summary)
    write_json(output_dir / "summary.json", summary_common)
    write_json(output_dir / "dataset_info.json", summary_common)
    return summary_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    output_dir = args.output_dir or Path(config["output"]["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_command(output_dir / "command.sh")
    try:
        summary = prepare(config, output_dir)
    except Exception as exc:
        error = {
            "status": "failed",
            "timestamp_utc": utc_now(),
            "failure_reason": repr(exc),
            "command": shlex.join([sys.executable, *sys.argv]),
        }
        write_json(output_dir / "summary.json", error)
        (output_dir / "error.log").write_text(repr(exc) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"status": summary["status"], "source": summary["dataset_source"]}), flush=True)


if __name__ == "__main__":
    main()
