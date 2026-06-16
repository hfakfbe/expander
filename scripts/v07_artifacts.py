from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph_structures import load_graph_artifact


V07_GRAPH_GENERATION_ALGORITHM = "zigzag_v07_fixed_N1024_q32_B32_d8"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(cwd: Path | None = None) -> str:
    env_commit = os.environ.get("GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit
    root = cwd or Path.cwd()
    for parent in [root, *root.parents]:
        deployed = parent / ".deployed_git_commit_v07"
        if deployed.exists():
            value = deployed.read_text(encoding="utf-8").strip()
            if value:
                return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def normalize_certificate(certificate: dict) -> dict:
    out = dict(certificate or {})
    if "rho_zigzag_bound" not in out and "rho_bound" in out:
        out["rho_zigzag_bound"] = out["rho_bound"]
    if "rho_zigzag_exact" not in out and "rho_exact" in out:
        out["rho_zigzag_exact"] = out["rho_exact"]
    if "rho_zigzag_certified" not in out:
        out["rho_zigzag_certified"] = bool(out.get("certified", False))
    if "certified" not in out:
        out["certified"] = bool(out.get("rho_zigzag_certified", False))
    if "rho_bound" not in out and "rho_zigzag_bound" in out:
        out["rho_bound"] = out["rho_zigzag_bound"]
    if "rho_exact" not in out and "rho_zigzag_exact" in out:
        out["rho_exact"] = out["rho_zigzag_exact"]
    return out


@dataclass
class GraphMaterialization:
    artifact: dict
    certificate: dict
    graph_generation: dict
    artifact_dir: Path
    selected_graph_path: Path
    certificate_path: Path
    generation_path: Path
    sha256_path: Path
    graph_artifact_sha256: str
    graph_certificate_sha256: str
    graph_generation_sha256: str
    canonical_graph_dir: Path
    canonical_graph_artifact_path: Path
    canonical_graph_artifact_sha256: str
    canonical_graph_seed: int | str
    canonical_graph_generation_algorithm: str
    sha256_matches_canonical: bool

    def as_dict(self) -> dict:
        return {
            "artifact_dir": str(self.artifact_dir),
            "graph_artifact_path": str(self.selected_graph_path),
            "graph_certificate_path": str(self.certificate_path),
            "graph_generation_path": str(self.generation_path),
            "graph_artifact_sha256": self.graph_artifact_sha256,
            "graph_certificate_sha256": self.graph_certificate_sha256,
            "graph_generation_sha256": self.graph_generation_sha256,
            "canonical_graph_dir": str(self.canonical_graph_dir),
            "canonical_graph_artifact_path": str(self.canonical_graph_artifact_path),
            "canonical_graph_artifact_sha256": self.canonical_graph_artifact_sha256,
            "canonical_graph_seed": self.canonical_graph_seed,
            "canonical_graph_generation_algorithm": self.canonical_graph_generation_algorithm,
            "graph_artifact_sha256_matches_canonical": self.sha256_matches_canonical,
        }


def _config_graph(config: dict) -> dict:
    graph = dict(config.get("graph", {}))
    if graph:
        return graph
    attention = config.get("attention", {})
    artifact = attention.get("graph_artifact")
    if artifact:
        return {
            "source_dir": str(Path(artifact).parent),
            "selected_graph_filename": Path(artifact).name,
            "copy_to_subdir": "artifacts/graph",
            "require_sha256_match": False,
        }
    return {}


def _source_paths(graph_cfg: dict) -> tuple[Path, Path, Path, Path | None]:
    source_dir = Path(graph_cfg.get("source_dir", ""))
    selected_name = graph_cfg.get("selected_graph_filename", "selected_graph.json")
    selected = source_dir / selected_name
    certificate = source_dir / "graph_certificate.json"
    if not certificate.exists():
        certificate = source_dir / "selected_graph_certificate.json"
    generation = source_dir / "graph_generation.json"
    sha_path = source_dir / "graph_artifact.sha256"
    return selected, certificate, generation, sha_path if sha_path.exists() else None


def materialize_graph_artifact(
    config: dict,
    output_dir: Path,
    *,
    require: bool = True,
) -> GraphMaterialization | None:
    graph_cfg = _config_graph(config)
    if not graph_cfg:
        if require:
            raise ValueError("missing graph config/source_dir")
        return None
    if bool(graph_cfg.get("generate", False)):
        if require:
            raise ValueError("training configs must point to an existing canonical graph source_dir")
        return None

    source_selected, source_cert, source_generation, source_sha_path = _source_paths(graph_cfg)
    if not source_selected.exists():
        if require:
            raise FileNotFoundError(f"canonical selected_graph.json not found: {source_selected}")
        return None

    copy_subdir = graph_cfg.get("copy_to_subdir", "artifacts/graph")
    artifact_dir = output_dir / copy_subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    selected_path = artifact_dir / "selected_graph.json"
    certificate_path = artifact_dir / "graph_certificate.json"
    generation_path = artifact_dir / "graph_generation.json"
    sha256_path = artifact_dir / "graph_artifact.sha256"

    if source_selected.resolve() != selected_path.resolve():
        shutil.copyfile(source_selected, selected_path)

    artifact = load_graph_artifact(selected_path)
    if source_cert.exists():
        source_certificate = normalize_certificate(read_json(source_cert))
    elif artifact.get("certificate"):
        source_certificate = normalize_certificate(dict(artifact["certificate"]))
    else:
        source_certificate = normalize_certificate({})
    write_json(certificate_path, source_certificate)

    canonical_sha = (
        source_sha_path.read_text(encoding="utf-8").split()[0].strip()
        if source_sha_path is not None
        else file_sha256(source_selected)
    )
    actual_sha = file_sha256(selected_path)
    expected_sha = str(graph_cfg.get("expected_graph_artifact_sha256") or "").strip()
    if expected_sha and actual_sha != expected_sha and bool(graph_cfg.get("require_sha256_match", True)):
        raise ValueError(
            f"graph artifact sha256 mismatch: actual={actual_sha} expected={expected_sha}"
        )
    if actual_sha != canonical_sha and bool(graph_cfg.get("require_sha256_match", True)):
        raise ValueError(
            f"graph artifact sha256 does not match canonical: actual={actual_sha} canonical={canonical_sha}"
        )

    source_generation_payload = read_json(source_generation) if source_generation.exists() else {}
    graph_generation_algorithm = str(
        graph_cfg.get("graph_generation_algorithm")
        or source_generation_payload.get("graph_generation_algorithm")
        or artifact.get("graph_generation_algorithm")
        or V07_GRAPH_GENERATION_ALGORITHM
    )
    graph_generation = dict(source_generation_payload)
    graph_generation.update(
        {
            "canonical_source_path": str(source_selected),
            "canonical_graph_dir": str(source_selected.parent),
            "canonical_graph_artifact_path": str(source_selected),
            "copy_timestamp_utc": utc_now(),
            "graph_seed": artifact.get("graph_seed", graph_cfg.get("graph_seed", "")),
            "graph_generation_algorithm": graph_generation_algorithm,
            "sha256": actual_sha,
            "canonical_graph_artifact_sha256": canonical_sha,
            "expected_graph_artifact_sha256": expected_sha,
            "canonical_sha256_verified": actual_sha == canonical_sha,
            "allow_multiedges": bool(graph_cfg.get("allow_multiedges", artifact.get("allow_multiedges", True))),
            "preserve_multiplicity": bool(
                graph_cfg.get("preserve_multiplicity", artifact.get("preserve_multiplicity", True))
            ),
        }
    )
    write_json(generation_path, graph_generation)
    sha256_path.write_text(actual_sha + "  selected_graph.json\n", encoding="utf-8")

    certificate_sha = file_sha256(certificate_path)
    generation_sha = file_sha256(generation_path)
    return GraphMaterialization(
        artifact=artifact,
        certificate=source_certificate,
        graph_generation=graph_generation,
        artifact_dir=artifact_dir,
        selected_graph_path=selected_path,
        certificate_path=certificate_path,
        generation_path=generation_path,
        sha256_path=sha256_path,
        graph_artifact_sha256=actual_sha,
        graph_certificate_sha256=certificate_sha,
        graph_generation_sha256=generation_sha,
        canonical_graph_dir=source_selected.parent,
        canonical_graph_artifact_path=source_selected,
        canonical_graph_artifact_sha256=canonical_sha,
        canonical_graph_seed=artifact.get("graph_seed", graph_cfg.get("graph_seed", "")),
        canonical_graph_generation_algorithm=graph_generation_algorithm,
        sha256_matches_canonical=actual_sha == canonical_sha,
    )
