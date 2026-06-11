import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path


LISTOPS_URL = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_if_needed(url: str, path: Path, timeout: int, force: bool = False) -> dict:
    if path.exists() and path.stat().st_size > 0 and not force:
        return {"status": "exists", "path": str(path), "bytes": path.stat().st_size}
    if force and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--show-error",
            "--silent",
            "--max-time",
            str(timeout),
            "--output",
            str(path),
            url,
        ],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
        return {
            "status": "blocked",
            "url": url,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    return {"status": "downloaded", "path": str(path), "bytes": path.stat().st_size}


def gzip_ok(path: Path) -> tuple[bool, str | None]:
    try:
        with gzip.open(path, "rb") as fp:
            for _ in iter(lambda: fp.read(1024 * 1024), b""):
                pass
        return True, None
    except Exception as exc:
        return False, repr(exc)


def extract_tar_gz(archive: Path, dest_dir: Path, members_prefix: str | None = None) -> dict:
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if members_prefix is not None:
                members = [member for member in members if member.name.startswith(members_prefix)]
            if not members:
                return {"status": "blocked", "error": f"no members matched {members_prefix!r}"}
            tar.extractall(path=dest_dir, members=members)
    except Exception as exc:
        return {"status": "blocked", "error": repr(exc)}
    return {"status": "ok", "members": len(members)}


def inspect_listops_split(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    labels: Counter[str] = Counter()
    max_tokens = 0
    rows = 0
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            rows += 1
            labels[str(row.get("Target", ""))] += 1
            max_tokens = max(max_tokens, len(str(row.get("Source", "")).split()))
    return {
        "exists": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "label_distribution": dict(sorted(labels.items())),
        "max_tokens": max_tokens,
    }


def inspect_imdb_split(path: Path) -> dict:
    neg = sorted((path / "neg").glob("*.txt"))
    pos = sorted((path / "pos").glob("*.txt"))
    max_chars = 0
    for sample in (neg[:100] + pos[:100]):
        max_chars = max(max_chars, len(sample.read_text(encoding="utf-8", errors="replace")))
    return {
        "exists": path.exists(),
        "path": str(path),
        "neg_count": len(neg),
        "pos_count": len(pos),
        "max_chars_sampled": max_chars,
    }


def listops_source_scope(repo_dir: Path) -> str:
    train = repo_dir / "datasets" / "lra_release" / "listops-1000" / "basic_train.tsv"
    if not train.exists():
        return "missing"
    rows = inspect_listops_split(train)["rows"]
    if rows < 10000:
        return "pipeline_only_generated_or_incomplete"
    return "official_candidate_requires_provenance"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path("code/lra-benchmarks"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/phase6_data_readiness/readiness.json"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    datasets_dir = repo_dir / "datasets"
    archives_dir = datasets_dir / "_archives"
    result = {
        "status": "blocked",
        "repo_dir": str(repo_dir),
        "downloads": {},
        "listops": {"source_scope": listops_source_scope(repo_dir), "splits": {}},
        "imdb": {"splits": {}},
    }

    if args.download:
        listops_archive = archives_dir / "lra_release.gz"
        imdb_archive = archives_dir / "aclImdb_v1.tar.gz"
        if listops_archive.exists():
            ok, _ = gzip_ok(listops_archive)
            if not ok:
                listops_archive.unlink()
        result["downloads"]["listops"] = download_if_needed(LISTOPS_URL, listops_archive, args.timeout)
        if listops_archive.exists():
            ok, error = gzip_ok(listops_archive)
            result["downloads"]["listops"].update(
                {
                    "gzip_ok": ok,
                    "gzip_error": error,
                    "sha256": sha256_file(listops_archive) if ok else None,
                }
            )
            if ok:
                result["downloads"]["listops_extract"] = extract_tar_gz(
                    listops_archive, datasets_dir, "lra_release/listops-1000"
                )
        if imdb_archive.exists():
            ok, _ = gzip_ok(imdb_archive)
            if not ok:
                imdb_archive.unlink()
        result["downloads"]["imdb"] = download_if_needed(IMDB_URL, imdb_archive, args.timeout)
        if imdb_archive.exists():
            ok, error = gzip_ok(imdb_archive)
            result["downloads"]["imdb"].update(
                {
                    "gzip_ok": ok,
                    "gzip_error": error,
                    "sha256": sha256_file(imdb_archive) if ok else None,
                }
            )
            if ok:
                result["downloads"]["imdb_extract"] = extract_tar_gz(imdb_archive, datasets_dir)

    listops_dir = datasets_dir / "lra_release" / "listops-1000"
    for split in ("basic_train.tsv", "basic_val.tsv", "basic_test.tsv"):
        result["listops"]["splits"][split] = inspect_listops_split(listops_dir / split)

    imdb_dir = datasets_dir / "aclImdb"
    for split in ("train", "test"):
        result["imdb"]["splits"][split] = inspect_imdb_split(imdb_dir / split)

    listops_ready = (
        result["listops"]["source_scope"] == "official_candidate_requires_provenance"
        and result["listops"]["splits"]["basic_train.tsv"]["rows"] > 0
        and result["listops"]["splits"]["basic_val.tsv"]["rows"] > 0
    )
    imdb_ready = (
        result["imdb"]["splits"]["train"]["neg_count"] > 0
        and result["imdb"]["splits"]["train"]["pos_count"] > 0
        and result["imdb"]["splits"]["test"]["neg_count"] > 0
        and result["imdb"]["splits"]["test"]["pos_count"] > 0
    )
    result["listops"]["ready_for_official_claims"] = listops_ready
    result["imdb"]["ready"] = imdb_ready
    result["status"] = "ready" if listops_ready and imdb_ready else "blocked"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
