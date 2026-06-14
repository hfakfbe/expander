from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from graph_structures import (
    DEFAULT_GRAPH_CONFIG,
    build_attention_mask,
    build_graph_artifact,
    build_local_mask,
    build_random_regular_cross_edges,
    build_zigzag_multiplicity,
    canonical_method,
    counts_to_mask,
    expected_raw_k,
    load_graph_artifact,
    padded_length,
)
from v07_artifacts import (
    V07_GRAPH_GENERATION_ALGORITHM,
    file_sha256,
    git_commit,
    normalize_certificate,
)


CERTIFICATE_FIELDS = [
    "version",
    "graph_id",
    "graph_seed",
    "T_raw",
    "T",
    "N_task",
    "B",
    "d",
    "q",
    "G_type",
    "H_type",
    "rot_g_is_bijection",
    "P_G_row_stochastic_error",
    "P_G_col_stochastic_error",
    "P_H_row_stochastic_error",
    "P_H_col_stochastic_error",
    "lambda_G",
    "mu_H",
    "rho_bound",
    "rho_zigzag_bound",
    "rho_zigzag_certified",
    "rho_exact",
    "rho_zigzag_exact",
    "simple_condition_lhs",
    "certified",
    "resample_reason",
    "remote_labelled_K",
    "local_labelled_K",
    "raw_K",
    "remote_unique_K_min",
    "remote_unique_K_mean",
    "remote_unique_K_max",
    "collision_count_min",
    "collision_count_mean",
    "collision_count_max",
    "remote_local_overlap_min",
    "remote_local_overlap_mean",
    "remote_local_overlap_max",
    "unique_total_K_min",
    "unique_total_K_mean",
    "unique_total_K_max",
    "row_degree_min_boolean",
    "row_degree_max_boolean",
    "col_degree_min_boolean",
    "col_degree_max_boolean",
    "stationary_l2_to_uniform_boolean",
]


DEFAULT_CONFIG = {
    "version": "v06",
    "N_task": 512,
    "T_raw": 1026,
    "candidate_block_sizes": [16, 32, 64],
    "candidate_degrees": [4, 6, 8],
    "graph_seeds": [0, 1, 2, 3, 4],
    "G": {
        "type": "permutation_regular",
        "require_derangement": True,
        "max_parallel_edges_per_block_pair": 2,
    },
    "H": {
        "type": "permutation_regular",
        "allow_self_port": False,
    },
    "acceptance": {
        "rho_bound_lt": 1.0,
        "prefer_simple_sufficient_condition": True,
        "max_remote_local_overlap_mean": 0.25,
    },
    "output": {"root": "outputs/copy_v06_graph_search"},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return deep_merge(DEFAULT_CONFIG, loaded)


def transition_from_permutations(permutations: list[list[int]], n: int) -> np.ndarray:
    P = np.zeros((n, n), dtype=np.float64)
    weight = 1.0 / len(permutations)
    for perm in permutations:
        for src, dst in enumerate(perm):
            P[int(dst), int(src)] += weight
    return P


def check_doubly_stochastic(P: np.ndarray) -> tuple[float, float]:
    row_error = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    col_error = float(np.max(np.abs(P.sum(axis=0) - 1.0)))
    return row_error, col_error


def spectral_norm(A: np.ndarray, exact: bool = False, max_iter: int = 250) -> float:
    if exact or A.shape[0] <= 256:
        return float(np.linalg.svd(A, compute_uv=False)[0])
    rng = np.random.default_rng(0)
    v = rng.normal(size=A.shape[1])
    v /= max(np.linalg.norm(v), 1e-12)
    sigma = 0.0
    for _ in range(max_iter):
        u = A @ v
        u_norm = np.linalg.norm(u)
        if u_norm <= 1e-15:
            return 0.0
        u /= u_norm
        v_next = A.T @ u
        v_norm = np.linalg.norm(v_next)
        if v_norm <= 1e-15:
            return 0.0
        v = v_next / v_norm
        sigma = float(v_norm)
    return sigma


def compute_lambda_g(P_G: np.ndarray) -> float:
    q = P_G.shape[0]
    return spectral_norm(P_G - np.ones((q, q), dtype=np.float64) / q, exact=True)


def compute_mu_h(P_H: np.ndarray) -> float:
    B = P_H.shape[0]
    return spectral_norm(P_H - np.ones((B, B), dtype=np.float64) / B, exact=True)


def compute_rho_bound(lambda_G: float, mu_H: float) -> float:
    inside = lambda_G**2 + 2 * mu_H**2 - (lambda_G**2) * (mu_H**2)
    return float(math.sqrt(max(inside, 0.0)))


def check_rot_g_bijection(artifact: dict) -> bool:
    q = int(artifact["q"])
    B = int(artifact["B"])
    seen: set[tuple[int, int]] = set()
    for port, perm in enumerate(artifact["G"]["permutations"]):
        for src_block, dst_block in enumerate(perm):
            src = (src_block, port)
            dst = (int(dst_block), port)
            if src[0] < 0 or src[0] >= q or dst[0] < 0 or dst[0] >= q:
                return False
            if src[1] < 0 or src[1] >= B or dst[1] < 0 or dst[1] >= B:
                return False
            seen.add(dst)
    return len(seen) == q * B


def build_pzz_matrix(artifact: dict) -> np.ndarray:
    T = int(artifact["T"])
    B = int(artifact["B"])
    d = int(artifact["d"])
    rows = build_zigzag_multiplicity(T, B, d, artifact, include_local=False)
    P = np.zeros((T, T), dtype=np.float64)
    denom = float(d * d)
    for src, counts in enumerate(rows):
        for dst, count in counts.items():
            P[int(dst), int(src)] += float(count) / denom
    return P


def compute_rho_exact(artifact: dict) -> float:
    P = build_pzz_matrix(artifact)
    T = P.shape[0]
    return spectral_norm(P - np.ones((T, T), dtype=np.float64) / T)


def _stats(values: list[float | int], prefix: str) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_min": float(arr.min()) if arr.size else 0.0,
        f"{prefix}_mean": float(arr.mean()) if arr.size else 0.0,
        f"{prefix}_max": float(arr.max()) if arr.size else 0.0,
    }


def compute_collision_overlap_stats(artifact: dict) -> dict:
    T = int(artifact["T"])
    B = int(artifact["B"])
    d = int(artifact["d"])
    remote_rows = build_zigzag_multiplicity(T, B, d, artifact, include_local=False)
    remote_unique: list[int] = []
    collisions: list[int] = []
    overlaps: list[int] = []
    unique_total: list[int] = []
    local = build_local_mask(T, B, torch.device("cpu"))
    boolean = torch.zeros((T, T), dtype=torch.bool)
    for src, counts in enumerate(remote_rows):
        remote_keys = set(counts.keys())
        local_keys = set(range((src // B) * B, (src // B + 1) * B))
        total_keys = local_keys | remote_keys
        remote_unique.append(len(remote_keys))
        collisions.append(d * d - len(remote_keys))
        overlaps.append(len(remote_keys & local_keys))
        unique_total.append(len(total_keys))
        if total_keys:
            boolean[src, torch.tensor(sorted(total_keys), dtype=torch.long)] = True
    row_degree = boolean.sum(dim=1).numpy()
    col_degree = boolean.sum(dim=0).numpy()
    transition = boolean.double().numpy()
    transition = transition / np.maximum(transition.sum(axis=1, keepdims=True), 1.0)
    pi = np.ones(T, dtype=np.float64) / T
    for _ in range(500):
        pi = pi @ transition
    uniform = np.ones(T, dtype=np.float64) / T
    out = {}
    out.update(_stats(remote_unique, "remote_unique_K"))
    out.update(_stats(collisions, "collision_count"))
    out.update(_stats(overlaps, "remote_local_overlap"))
    out.update(_stats(unique_total, "unique_total_K"))
    out.update(
        {
            "row_degree_min_boolean": int(row_degree.min()),
            "row_degree_max_boolean": int(row_degree.max()),
            "col_degree_min_boolean": int(col_degree.min()),
            "col_degree_max_boolean": int(col_degree.max()),
            "stationary_l2_to_uniform_boolean": float(np.linalg.norm(pi - uniform)),
            "local_labelled_K": B,
            "remote_labelled_K": d * d,
            "raw_K": B + d * d,
        }
    )
    # Keep the local tensor used above alive only through this point; this line also
    # guards against accidentally deleting the local component from the boolean graph.
    assert bool(local.any())
    return out


def compute_boolean_ablation_stats(artifact: dict) -> dict:
    return compute_collision_overlap_stats(artifact)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_graph_certificate(path: Path, certificate: dict) -> None:
    write_json(path, certificate)


def certificate_for_artifact(artifact: dict, config: dict, resample_reason: str = "") -> dict:
    q = int(artifact["q"])
    B = int(artifact["B"])
    d = int(artifact["d"])
    P_G = transition_from_permutations(artifact["G"]["permutations"], q)
    P_H = transition_from_permutations(artifact["H"]["permutations"], B)
    g_row_err, g_col_err = check_doubly_stochastic(P_G)
    h_row_err, h_col_err = check_doubly_stochastic(P_H)
    lambda_G = compute_lambda_g(P_G)
    mu_H = compute_mu_h(P_H)
    rho_bound = compute_rho_bound(lambda_G, mu_H)
    simple_lhs = lambda_G**2 + 2 * mu_H**2
    rho_exact = compute_rho_exact(artifact)
    acceptance = config.get("acceptance", {})
    certified = (
        check_rot_g_bijection(artifact)
        and g_row_err < 1e-10
        and g_col_err < 1e-10
        and h_row_err < 1e-10
        and h_col_err < 1e-10
        and rho_bound < float(acceptance.get("rho_bound_lt", 1.0))
        and rho_exact <= rho_bound + 1e-6
    )
    if not certified and not resample_reason:
        if rho_bound >= float(acceptance.get("rho_bound_lt", 1.0)):
            resample_reason = "rho_bound_not_lt_threshold"
        elif rho_exact > rho_bound + 1e-6:
            resample_reason = "rho_exact_exceeds_bound"
        else:
            resample_reason = "certificate_checks_failed"
    stats = compute_collision_overlap_stats(artifact)
    max_overlap = acceptance.get("max_remote_local_overlap_mean")
    if max_overlap is not None and stats["remote_local_overlap_mean"] > float(max_overlap):
        certified = False
        resample_reason = resample_reason or "remote_local_overlap_mean_too_high"
    cert = {
        "version": artifact["version"],
        "graph_id": artifact["graph_id"],
        "graph_seed": artifact["graph_seed"],
        "T_raw": artifact["T_raw"],
        "T": artifact["T"],
        "N_task": artifact["N_task"],
        "B": B,
        "d": d,
        "q": q,
        "G_type": artifact["G"]["type"],
        "H_type": artifact["H"]["type"],
        "rot_g_is_bijection": check_rot_g_bijection(artifact),
        "P_G_row_stochastic_error": g_row_err,
        "P_G_col_stochastic_error": g_col_err,
        "P_H_row_stochastic_error": h_row_err,
        "P_H_col_stochastic_error": h_col_err,
        "lambda_G": lambda_G,
        "mu_H": mu_H,
        "rho_bound": rho_bound,
        "rho_zigzag_bound": rho_bound,
        "rho_exact": rho_exact,
        "rho_zigzag_exact": rho_exact,
        "rho_zigzag_certified": bool(certified),
        "simple_condition_lhs": simple_lhs,
        "certified": bool(certified),
        "resample_reason": resample_reason,
    }
    cert.update(stats)
    return cert


def failed_certificate(config: dict, B: int, d: int, graph_seed: int, reason: str) -> dict:
    T_raw = int(config["T_raw"])
    T = padded_length(T_raw, B)
    row = {field: "" for field in CERTIFICATE_FIELDS}
    row.update(
        {
            "version": config.get("version", "v06"),
            "graph_id": f"failed_B{B}_d{d}_s{graph_seed}",
            "graph_seed": graph_seed,
            "T_raw": T_raw,
            "T": T,
            "N_task": int(config["N_task"]),
            "B": B,
            "d": d,
            "q": T // B,
            "G_type": config.get("G", {}).get("type", ""),
            "H_type": config.get("H", {}).get("type", ""),
            "rot_g_is_bijection": False,
            "certified": False,
            "resample_reason": reason,
            "remote_labelled_K": d * d,
            "local_labelled_K": B,
            "raw_K": B + d * d,
        }
    )
    return row


def select_certificate(rows: list[dict]) -> dict | None:
    candidates = [row for row in rows if bool(row.get("certified"))]
    if not candidates:
        return None

    def key(row: dict):
        return (
            float(row["rho_bound"]),
            float(row["simple_condition_lhs"]),
            float(row["remote_local_overlap_mean"]),
            abs(float(row["unique_total_K_mean"]) - float(row["raw_K"])),
            int(row["B"]) * int(row["d"]),
        )

    return sorted(candidates, key=key)[0]


def run_graph_search(config: dict, output_dir: Path) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    artifacts: dict[str, dict] = {}
    for B in [int(v) for v in config["candidate_block_sizes"]]:
        for d in [int(v) for v in config["candidate_degrees"]]:
            if d >= B:
                for graph_seed in config["graph_seeds"]:
                    rows.append(failed_certificate(config, B, d, int(graph_seed), "degree_not_lt_B"))
                continue
            for graph_seed in [int(v) for v in config["graph_seeds"]]:
                try:
                    graph_cfg = config.get("graph", {})
                    artifact = build_graph_artifact(
                        N_task=int(config["N_task"]),
                        T_raw=int(config["T_raw"]),
                        block_size=B,
                        degree=d,
                        graph_seed=graph_seed,
                        g_config=config.get("G", {}),
                        h_config=config.get("H", {}),
                        version=config.get("version", "v06"),
                    )
                    if str(config.get("version", "")).lower() == "v07":
                        artifact["N_total"] = int(config.get("N_total", config["N_task"]))
                        artifact["allow_multiedges"] = bool(
                            graph_cfg.get("allow_multiedges", config.get("allow_multiedges", True))
                        )
                        artifact["preserve_multiplicity"] = bool(
                            graph_cfg.get(
                                "preserve_multiplicity",
                                config.get("preserve_multiplicity", True),
                            )
                        )
                        artifact["graph_generation_algorithm"] = str(
                            graph_cfg.get(
                                "graph_generation_algorithm",
                                V07_GRAPH_GENERATION_ALGORITHM,
                            )
                        )
                    cert = certificate_for_artifact(artifact, config)
                    artifact["certificate"] = normalize_certificate(cert)
                    artifacts[artifact["graph_id"]] = artifact
                    write_json(graph_dir / f"{artifact['graph_id']}.json", artifact)
                    rows.append(cert)
                    print(json.dumps({"graph_id": artifact["graph_id"], "certified": cert["certified"], "rho_bound": cert["rho_bound"]}), flush=True)
                except Exception as exc:
                    rows.append(failed_certificate(config, B, d, graph_seed, repr(exc)))
                    print(json.dumps({"failed_graph": f"B{B}_d{d}_s{graph_seed}", "error": repr(exc)}), flush=True)
    write_csv(output_dir / "graph_certificates.csv", rows, CERTIFICATE_FIELDS)
    write_jsonl(output_dir / "graph_certificates.jsonl", rows)
    selected = select_certificate(rows)
    if selected is not None:
        artifact = artifacts[selected["graph_id"]]
        selected = normalize_certificate(selected)
        artifact["certificate"] = selected
        write_json(output_dir / "selected_graph.json", artifact)
        write_json(output_dir / "selected_graph_certificate.json", selected)
        if str(config.get("version", "")).lower() == "v07":
            write_json(output_dir / "graph_certificate.json", selected)
            artifact_sha = file_sha256(output_dir / "selected_graph.json")
            (output_dir / "graph_artifact.sha256").write_text(
                artifact_sha + "  selected_graph.json\n", encoding="utf-8"
            )
            graph_cfg = config.get("graph", {})
            generation = {
                "status": "ok",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "git_commit": git_commit(Path.cwd()),
                "graph_generation_algorithm": str(
                    graph_cfg.get("graph_generation_algorithm", V07_GRAPH_GENERATION_ALGORITHM)
                ),
                "graph_seed": int(artifact["graph_seed"]),
                "N_total": int(config.get("N_total", artifact.get("N_total", artifact["N_task"]))),
                "N_task": int(artifact["N_task"]),
                "T_raw": int(artifact["T_raw"]),
                "T": int(artifact["T"]),
                "q": int(artifact["q"]),
                "B": int(artifact["B"]),
                "d": int(artifact["d"]),
                "allow_multiedges": bool(artifact.get("allow_multiedges", True)),
                "preserve_multiplicity": bool(artifact.get("preserve_multiplicity", True)),
                "canonical_graph_artifact_sha256": artifact_sha,
                "selected_graph_path": str(output_dir / "selected_graph.json"),
                "graph_certificate_path": str(output_dir / "graph_certificate.json"),
                "graph_artifact_sha256_path": str(output_dir / "graph_artifact.sha256"),
                "generation_attempts": len(rows),
            }
            write_json(output_dir / "graph_generation.json", generation)
    summary = {
        "status": "ok" if selected is not None else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_candidates": len(rows),
        "num_certified": sum(1 for row in rows if bool(row.get("certified"))),
        "selected_graph_id": selected["graph_id"] if selected is not None else "",
        "selected_certificate": selected,
        "total_wall_time_sec": time.perf_counter() - started,
    }
    if selected is not None and str(config.get("version", "")).lower() == "v07":
        artifact_sha = file_sha256(output_dir / "selected_graph.json")
        summary.update(
            {
                "canonical_graph_dir": str(output_dir),
                "canonical_graph_artifact_path": str(output_dir / "selected_graph.json"),
                "canonical_graph_artifact_sha256": artifact_sha,
                "canonical_graph_seed": artifacts[selected["graph_id"]].get("graph_seed", ""),
                "canonical_graph_generation_algorithm": artifacts[selected["graph_id"]].get(
                    "graph_generation_algorithm",
                    V07_GRAPH_GENERATION_ALGORITHM,
                ),
                "graph_seed": artifacts[selected["graph_id"]].get("graph_seed", ""),
                "graph_generation_algorithm": artifacts[selected["graph_id"]].get(
                    "graph_generation_algorithm",
                    V07_GRAPH_GENERATION_ALGORITHM,
                ),
                "N_total": int(config.get("N_total", artifacts[selected["graph_id"]].get("N_task", 0))),
                "T": int(artifacts[selected["graph_id"]]["T"]),
                "q": int(artifacts[selected["graph_id"]]["q"]),
                "B": int(artifacts[selected["graph_id"]]["B"]),
                "d": int(artifacts[selected["graph_id"]]["d"]),
                "allow_multiedges": bool(artifacts[selected["graph_id"]].get("allow_multiedges", True)),
                "preserve_multiplicity": bool(
                    artifacts[selected["graph_id"]].get("preserve_multiplicity", True)
                ),
            }
        )
    write_json(output_dir / "summary.json", summary)
    if selected is None:
        raise SystemExit("no certified graph found; stop before training")
    return summary


def _method_mask(method: str, T: int, B: int, d: int, seed: int, graph_artifact: dict | None, causal: bool) -> torch.Tensor:
    device = torch.device("cpu")
    method = canonical_method(method)
    graph_config = graph_artifact if method in {"zigzag_certified", "zigzag_boolean"} else DEFAULT_GRAPH_CONFIG
    if method == "random_regular":
        rows = [Counter() for _ in range(T)]
        for src, dst in build_random_regular_cross_edges(T, B, d, seed):
            rows[src][dst] += 1
        local = build_local_mask(T, B, device)
        mask = local | counts_to_mask(rows, T, device)
    else:
        mask = build_attention_mask(method, T, B, d, device, seed, graph_config)
    if causal:
        mask = mask & torch.ones((T, T), dtype=torch.bool).tril()
    return mask


def _shortcut_one_mask(mask: torch.Tensor, N: int, layers: int) -> dict:
    T = int(mask.shape[0])
    pairs = [(N + i, i) for i in range(N)]
    pairs.append((2 * N, min(2 * N + 1, T - 1)))
    one = 0
    two = 0
    lhop = 0
    shortest: list[int] = []
    unreachable = 0
    adjacency = [torch.nonzero(mask[row], as_tuple=False).flatten().tolist() for row in range(T)]
    for query, target in pairs:
        if query >= T or target >= T:
            unreachable += 1
            continue
        distance = 0 if query == target else None
        visited = {query}
        frontier = {query}
        depth = 0
        while distance is None and frontier:
            depth += 1
            nxt: set[int] = set()
            for node in frontier:
                for dst in adjacency[node]:
                    if dst == target:
                        distance = depth
                        break
                    if dst not in visited:
                        visited.add(dst)
                        nxt.add(dst)
                if distance is not None:
                    break
            frontier = nxt
        if distance is None:
            unreachable += 1
        else:
            shortest.append(distance)
            if distance <= 1:
                one += 1
            if distance <= 2:
                two += 1
            if distance <= layers:
                lhop += 1
    denom = float(len(pairs))
    return {
        "target_in_1hop_rate": one / denom,
        "target_in_2hop_rate": two / denom,
        "target_in_Lhop_rate": lhop / denom,
        "average_shortest_path": float(np.mean(shortest)) if shortest else math.inf,
        "unreachable_rate": unreachable / denom,
    }


def compute_shortcut_stats(
    method: str,
    N: int,
    T: int,
    B: int,
    d: int,
    seed: int,
    layers: int,
    graph_artifact: dict | None,
) -> list[dict]:
    structural = _method_mask(method, T, B, d, seed, graph_artifact, causal=False)
    causal = _method_mask(method, T, B, d, seed, graph_artifact, causal=True)
    rows = []
    for scope, mask in [("structural", structural), ("causal_effective", causal)]:
        stats = _shortcut_one_mask(mask, N=N, layers=layers)
        stats.update(
            {
                "method": method,
                "N": N,
                "T": T,
                "B": B,
                "d": d,
                "seed": seed,
                "layers": layers,
                "mask_scope": scope,
                "attention_pair_count": int(mask.sum().item()),
                "raw_K": expected_raw_k(method, T, B, d),
            }
        )
        rows.append(stats)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    output_dir = cli.output_dir or Path(config["output"]["root"])
    write_json(output_dir / "config_snapshot.json", config)
    run_graph_search(config, output_dir)


if __name__ == "__main__":
    main()
