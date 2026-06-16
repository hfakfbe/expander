from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from probe_common import (
    EXPERIMENT_VERSION,
    REQUIRED_VERSION_FILES,
    SELECTED_PROBES,
    command_string,
    count_lines,
    file_sha256,
    selected_probe_path,
    utc_now,
    write_command,
    write_csv,
    write_json,
)


def local_rows() -> list[dict]:
    rows = []
    for task in SELECTED_PROBES:
        version_dir = selected_probe_path(task)
        file_count = sum(1 for name in REQUIRED_VERSION_FILES if (version_dir / name).exists())
        rows.append(
            {
                "task": task,
                "version_path": str(version_dir),
                "file_count": file_count,
                "train_rows": count_lines(version_dir / "train.jsonl"),
                "validation_rows": count_lines(version_dir / "validation.jsonl"),
                "test_rows": count_lines(version_dir / "test.jsonl"),
                "train_sha256": file_sha256(version_dir / "train.jsonl"),
                "validation_sha256": file_sha256(version_dir / "validation.jsonl"),
                "test_sha256": file_sha256(version_dir / "test.jsonl"),
                "dataset_card_sha256": file_sha256(version_dir / "dataset_card.json"),
            }
        )
    return rows


def remote_probe_script(remote_data_root: str) -> str:
    rels = {
        task: str(Path(info["version_path"]).relative_to("../expander_bench/data/probes"))
        for task, info in SELECTED_PROBES.items()
    }
    return f"""
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path({remote_data_root!r})
rels = {json.dumps(rels)}
required = {json.dumps(REQUIRED_VERSION_FILES)}
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
def lines(path):
    c=0
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            c += chunk.count(b'\\n')
    return c
rows=[]
for task, rel in rels.items():
    d=root / rel
    rows.append({{
        'task': task,
        'remote_version_path': str(d),
        'exists': d.exists(),
        'file_count': sum(1 for name in required if (d/name).exists()),
        'train_rows': lines(d/'train.jsonl') if (d/'train.jsonl').exists() else -1,
        'validation_rows': lines(d/'validation.jsonl') if (d/'validation.jsonl').exists() else -1,
        'test_rows': lines(d/'test.jsonl') if (d/'test.jsonl').exists() else -1,
        'train_sha256': sha(d/'train.jsonl') if (d/'train.jsonl').exists() else '',
        'validation_sha256': sha(d/'validation.jsonl') if (d/'validation.jsonl').exists() else '',
        'test_sha256': sha(d/'test.jsonl') if (d/'test.jsonl').exists() else '',
        'dataset_card_sha256': sha(d/'dataset_card.json') if (d/'dataset_card.json').exists() else '',
    }})
env = {{}}
for cmd, key in [
    ([sys.executable, '-V'], 'python_version'),
    ([sys.executable, '-c', "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"], 'torch_cuda'),
    ([sys.executable, '-m', 'pip', 'freeze'], 'pip_freeze'),
]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    env[key] = p.stdout.strip()
gpu = subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(json.dumps({{'rows': rows, 'env': env, 'gpu': gpu.stdout}}, sort_keys=True))
"""


def remote_python_command(env_name: str) -> str:
    body = f"""
set -e
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  for f in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$f" ]; then
      . "$f"
      break
    fi
  done
fi
conda activate {shlex.quote(env_name)}
python -
""".strip()
    return "bash -lc " + shlex.quote(body)


def run_remote(host: str, remote_data_root: str, env_name: str) -> dict:
    proc = subprocess.run(
        ["ssh", host, remote_python_command(env_name)],
        input=remote_probe_script(remote_data_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="huiwei")
    parser.add_argument("--remote-data-root", default="/home/huiwei/ysx/expander_bench/data/probes")
    parser.add_argument("--remote-env", default="ysx_base")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probes_v08_remote_readiness"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local = local_rows()
    remote = run_remote(args.host, args.remote_data_root, args.remote_env)
    env_snapshot = {key: value for key, value in remote["env"].items() if key != "pip_freeze"}
    remote_by_task = {row["task"]: row for row in remote["rows"]}
    rows = []
    checksum_rows = []
    all_ok = True
    for local_row in local:
        remote_row = remote_by_task.get(local_row["task"], {})
        ok = (
            bool(remote_row.get("exists"))
            and local_row["file_count"] == remote_row.get("file_count")
            and local_row["train_rows"] == remote_row.get("train_rows")
            and local_row["validation_rows"] == remote_row.get("validation_rows")
            and local_row["test_rows"] == remote_row.get("test_rows")
            and local_row["train_sha256"] == remote_row.get("train_sha256")
            and local_row["validation_sha256"] == remote_row.get("validation_sha256")
            and local_row["test_sha256"] == remote_row.get("test_sha256")
            and local_row["dataset_card_sha256"] == remote_row.get("dataset_card_sha256")
        )
        all_ok = all_ok and ok
        rows.append({**local_row, **remote_row, "status": "ok" if ok else "failed"})
        for key in ["train_sha256", "validation_sha256", "test_sha256", "dataset_card_sha256"]:
            checksum_rows.append(
                {
                    "task": local_row["task"],
                    "field": key,
                    "local": local_row[key],
                    "remote": remote_row.get(key, ""),
                    "status": "ok" if local_row[key] == remote_row.get(key, "") else "failed",
                }
            )
    summary = {
        "version": EXPERIMENT_VERSION,
        "phase": "phase3_remote_readiness",
        "timestamp_utc": utc_now(),
        "status": "ok" if all_ok else "failed",
        "remote_host": args.host,
        "remote_data_root": args.remote_data_root,
        "remote_env": args.remote_env,
        "gpu_status": remote["gpu"],
        "env": env_snapshot,
        "requirements_snapshot_path": str(args.output_dir / "requirements_snapshot.txt"),
        "command": command_string(),
        "rows": rows,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "remote_file_counts.csv", rows)
    write_json(args.output_dir / "remote_checksums_verification.json", checksum_rows)
    (args.output_dir / "env_snapshot.txt").write_text(json.dumps(env_snapshot, indent=2) + "\n\n" + remote["gpu"], encoding="utf-8")
    (args.output_dir / "requirements_snapshot.txt").write_text(remote["env"].get("pip_freeze", "") + "\n", encoding="utf-8")
    write_command(args.output_dir / "command.sh")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
