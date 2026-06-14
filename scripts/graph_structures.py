from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_GRAPH_CONFIG = {
    "G": {"type": "cyclic"},
    "H": {"type": "cycle"},
}

V06_GRAPH_TYPE = "permutation_regular"


def _graph_section(graph_config: dict | None, name: str) -> dict:
    if graph_config is None:
        graph_config = DEFAULT_GRAPH_CONFIG
    return dict(graph_config.get(name, DEFAULT_GRAPH_CONFIG[name]))


def _stable_id(payload: dict, prefix: str = "graph") -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def padded_length(raw_length: int, block_size: int) -> int:
    return int(math.ceil(int(raw_length) / int(block_size)) * int(block_size))


def load_graph_artifact(path: str | Path) -> dict:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_graph_artifact(artifact)
    return artifact


def validate_graph_artifact(artifact: dict) -> None:
    required = ["version", "graph_id", "T", "B", "d", "q", "G", "H"]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"graph artifact missing keys: {missing}")
    if artifact["G"].get("type") != V06_GRAPH_TYPE:
        raise ValueError(f"unsupported artifact G type: {artifact['G'].get('type')}")
    if artifact["H"].get("type") != V06_GRAPH_TYPE:
        raise ValueError(f"unsupported artifact H type: {artifact['H'].get('type')}")
    q = int(artifact["q"])
    block_size = int(artifact["B"])
    degree = int(artifact["d"])
    g_perms = artifact["G"].get("permutations", [])
    h_perms = artifact["H"].get("permutations", [])
    if len(g_perms) != block_size:
        raise ValueError(f"G must contain B={block_size} permutations, got {len(g_perms)}")
    if len(h_perms) != degree:
        raise ValueError(f"H must contain d={degree} permutations, got {len(h_perms)}")
    for idx, perm in enumerate(g_perms):
        if sorted(int(v) for v in perm) != list(range(q)):
            raise ValueError(f"G permutation {idx} is not a permutation of [q]")
    for idx, perm in enumerate(h_perms):
        if sorted(int(v) for v in perm) != list(range(block_size)):
            raise ValueError(f"H permutation {idx} is not a permutation of [B]")
    if int(artifact["T"]) != q * block_size:
        raise ValueError("artifact T must equal q * B")


def validate_graph_config(
    seq_len: int,
    block_size: int,
    degree: int,
    graph_config: dict | None = None,
) -> dict:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    if degree <= 0:
        raise ValueError("degree must be positive")
    if degree >= block_size:
        raise ValueError("degree must be smaller than block_size")

    if graph_config is None:
        graph_config = DEFAULT_GRAPH_CONFIG
    graph_config = dict(graph_config)
    g_config = _graph_section(graph_config, "G")
    h_config = _graph_section(graph_config, "H")
    g_type = g_config.get("type")
    h_type = h_config.get("type")
    if g_type == "cyclic" and h_type == "cycle":
        return {"G": g_config, "H": h_config}
    if g_type == V06_GRAPH_TYPE and h_type == V06_GRAPH_TYPE:
        validate_graph_artifact(
            {
                "version": graph_config.get("version", "v06"),
                "graph_id": graph_config.get("graph_id", "inline"),
                "T": seq_len,
                "B": block_size,
                "d": degree,
                "q": seq_len // block_size,
                "G": g_config,
                "H": h_config,
            }
        )
        return graph_config
    raise ValueError(f"unsupported graph config G={g_type!r}, H={h_type!r}")


def _derangement(n: int, rng: random.Random, max_attempts: int = 10_000) -> list[int]:
    if n <= 1:
        raise ValueError("derangement requires n > 1")
    values = list(range(n))
    for _ in range(max_attempts):
        perm = values[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    shift = rng.randrange(1, n)
    return [(i + shift) % n for i in range(n)]


def _permutation(n: int, rng: random.Random, forbid_fixed_points: bool) -> list[int]:
    if forbid_fixed_points:
        return _derangement(n, rng)
    values = list(range(n))
    rng.shuffle(values)
    return values


def build_permutation_regular_h(
    block_size: int,
    degree: int,
    seed: int,
    allow_self_port: bool = False,
) -> dict:
    rng = random.Random(f"H|{seed}|{block_size}|{degree}|{int(allow_self_port)}")
    permutations = [
        _permutation(block_size, rng, forbid_fixed_points=not allow_self_port)
        for _ in range(degree)
    ]
    return {
        "type": V06_GRAPH_TYPE,
        "allow_self_port": bool(allow_self_port),
        "permutations": permutations,
    }


def _offset_permutations_for_g(
    num_blocks: int,
    block_size: int,
    seed: int,
    require_derangement: bool,
    max_parallel_edges_per_block_pair: int | None,
) -> list[list[int]]:
    if require_derangement and num_blocks <= 1:
        raise ValueError("deranged block permutations require q > 1")
    available_offsets = list(range(1 if require_derangement else 0, num_blocks))
    if not available_offsets:
        raise ValueError("no legal block offsets for G")
    if (
        max_parallel_edges_per_block_pair is not None
        and require_derangement
        and block_size > max_parallel_edges_per_block_pair * (num_blocks - 1)
    ):
        raise ValueError(
            "max_parallel_edges_per_block_pair is impossible for this B and q"
        )
    rng = random.Random(
        f"G_offsets|{seed}|{num_blocks}|{block_size}|{int(require_derangement)}"
    )
    offsets: list[int] = []
    while len(offsets) < block_size:
        shuffled = available_offsets[:]
        rng.shuffle(shuffled)
        offsets.extend(shuffled)
    offsets = offsets[:block_size]
    labels = list(range(num_blocks))
    rng.shuffle(labels)
    inverse_labels = [0] * num_blocks
    for idx, label in enumerate(labels):
        inverse_labels[label] = idx
    permutations: list[list[int]] = []
    for offset in offsets:
        perm = [
            inverse_labels[(labels[v] + offset) % num_blocks]
            for v in range(num_blocks)
        ]
        permutations.append(perm)
    return permutations


def _check_parallel_limit(permutations: list[list[int]], max_parallel: int | None) -> None:
    if max_parallel is None:
        return
    if not permutations:
        return
    num_blocks = len(permutations[0])
    counts = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]
    for perm in permutations:
        for src, dst in enumerate(perm):
            counts[src][dst] += 1
            if counts[src][dst] > max_parallel:
                raise ValueError("parallel edge limit exceeded")


def build_permutation_regular_g(
    num_blocks: int,
    block_size: int,
    seed: int,
    require_derangement: bool = True,
    max_parallel_edges_per_block_pair: int | None = 2,
) -> dict:
    permutations = _offset_permutations_for_g(
        num_blocks=num_blocks,
        block_size=block_size,
        seed=seed,
        require_derangement=require_derangement,
        max_parallel_edges_per_block_pair=max_parallel_edges_per_block_pair,
    )
    _check_parallel_limit(permutations, max_parallel_edges_per_block_pair)
    return {
        "type": V06_GRAPH_TYPE,
        "require_derangement": bool(require_derangement),
        "max_parallel_edges_per_block_pair": max_parallel_edges_per_block_pair,
        "permutations": permutations,
        "arrival_port": "identity",
    }


def build_graph_artifact(
    N_task: int,
    T_raw: int,
    block_size: int,
    degree: int,
    graph_seed: int,
    g_config: dict | None = None,
    h_config: dict | None = None,
    version: str = "v06",
) -> dict:
    T = padded_length(T_raw, block_size)
    q = T // block_size
    g_config = dict(g_config or {})
    h_config = dict(h_config or {})
    G = build_permutation_regular_g(
        num_blocks=q,
        block_size=block_size,
        seed=graph_seed,
        require_derangement=bool(g_config.get("require_derangement", True)),
        max_parallel_edges_per_block_pair=g_config.get("max_parallel_edges_per_block_pair", 2),
    )
    H = build_permutation_regular_h(
        block_size=block_size,
        degree=degree,
        seed=graph_seed,
        allow_self_port=bool(h_config.get("allow_self_port", False)),
    )
    payload = {
        "version": version,
        "graph_seed": int(graph_seed),
        "N_task": int(N_task),
        "T_raw": int(T_raw),
        "T": int(T),
        "B": int(block_size),
        "d": int(degree),
        "q": int(q),
        "G": G,
        "H": H,
    }
    payload["graph_id"] = _stable_id(
        payload,
        prefix=f"{version}_B{block_size}_d{degree}_s{graph_seed}",
    )
    validate_graph_artifact(payload)
    return payload


def build_rot_g(graph_artifact: dict) -> list[tuple[int, int]]:
    validate_graph_artifact(graph_artifact)
    q = int(graph_artifact["q"])
    block_size = int(graph_artifact["B"])
    permutations = graph_artifact["G"]["permutations"]
    rot: list[tuple[int, int]] = []
    for v in range(q):
        for port in range(block_size):
            rot.append((int(permutations[port][v]), port))
    return rot


def build_h_graph(
    block_size: int,
    degree: int,
    graph_config: dict | None = None,
) -> dict[int, list[int]]:
    validate_graph_config(block_size, block_size, degree, graph_config)
    return {
        port: h_neighbors(port, block_size, degree, graph_config)
        for port in range(block_size)
    }


def h_neighbors(
    port: int,
    block_size: int,
    degree: int,
    graph_config: dict | None = None,
) -> list[int]:
    h_config = _graph_section(graph_config, "H")
    h_type = h_config.get("type")
    if h_type == V06_GRAPH_TYPE:
        permutations = h_config["permutations"]
        return [int(permutations[a][port]) for a in range(degree)]
    if h_type != "cycle":
        raise ValueError(f"unsupported H graph type: {h_type}")
    if degree >= block_size:
        raise ValueError("degree must be smaller than block_size")
    offsets: list[int] = []
    step = 1
    while len(offsets) < degree:
        offsets.append(step)
        if len(offsets) < degree:
            offsets.append(-step)
        step += 1
    return [int((port + off) % block_size) for off in offsets[:degree]]


def rot_g(
    block: int,
    port: int,
    num_blocks: int,
    block_size: int,
    graph_config: dict | None = None,
) -> tuple[int, int]:
    g_config = _graph_section(graph_config, "G")
    g_type = g_config.get("type")
    if g_type == V06_GRAPH_TYPE:
        return int(g_config["permutations"][port][block]), port
    if g_type != "cyclic":
        raise ValueError(f"unsupported G graph type: {g_type}")
    max_offset = max(1, num_blocks // 2)
    offset = (port // 2) % max_offset + 1
    if port % 2 == 0:
        return (block + offset) % num_blocks, port ^ 1
    return (block - offset) % num_blocks, port ^ 1


def rot_g_cyclic(
    block: int,
    port: int,
    num_blocks: int,
    block_size: int,
) -> tuple[int, int]:
    return rot_g(block, port, num_blocks, block_size, DEFAULT_GRAPH_CONFIG)


def build_local_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    return (idx[:, None] // block_size) == (idx[None, :] // block_size)


def local_key_range(src: int, block_size: int) -> range:
    start = (src // block_size) * block_size
    return range(start, start + block_size)


def build_zigzag_multiplicity(
    seq_len: int,
    block_size: int,
    degree: int,
    graph_config: dict | None,
    include_local: bool = True,
) -> list[Counter[int]]:
    graph_config = validate_graph_config(seq_len, block_size, degree, graph_config)
    num_blocks = seq_len // block_size
    rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    for v in range(num_blocks):
        for i in range(block_size):
            src = v * block_size + i
            if include_local:
                for dst in local_key_range(src, block_size):
                    rows[src][dst] += 1
            for i_prime in h_neighbors(i, block_size, degree, graph_config):
                w, j_prime = rot_g(v, i_prime, num_blocks, block_size, graph_config)
                for j in h_neighbors(j_prime, block_size, degree, graph_config):
                    dst = w * block_size + j
                    rows[src][dst] += 1
    return rows


def build_zigzag_cross_edges(
    seq_len: int,
    block_size: int,
    degree: int,
    graph_config: dict | None = None,
) -> list[tuple[int, int]]:
    rows = build_zigzag_multiplicity(
        seq_len=seq_len,
        block_size=block_size,
        degree=degree,
        graph_config=graph_config,
        include_local=False,
    )
    edges: list[tuple[int, int]] = []
    for src, counts in enumerate(rows):
        for dst, multiplicity in counts.items():
            edges.extend((src, dst) for _ in range(int(multiplicity)))
    return edges


def build_random_regular_cross_edges(
    seq_len: int,
    block_size: int,
    degree: int,
    seed: int,
) -> list[tuple[int, int]]:
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    num_blocks = seq_len // block_size
    if num_blocks <= 1:
        raise ValueError("random regular cross edges require at least two blocks")
    k = int(degree) * int(degree)
    rng = random.Random(f"random_regular|{seed}|{seq_len}|{block_size}|{degree}")
    edges: list[tuple[int, int]] = []
    for _slot in range(k):
        block_perm = _derangement(num_blocks, rng)
        port_perm = list(range(block_size))
        rng.shuffle(port_perm)
        for src_block, dst_block in enumerate(block_perm):
            for src_port, dst_port in enumerate(port_perm):
                src = src_block * block_size + src_port
                dst = dst_block * block_size + dst_port
                edges.append((src, dst))
    return edges


def build_random_cross_edges(
    seq_len: int,
    block_size: int,
    degree: int,
    seed: int,
    graph_config: dict | None = None,
) -> list[tuple[int, int]]:
    validate_graph_config(seq_len, block_size, degree, graph_config)
    return build_random_regular_cross_edges(seq_len, block_size, degree, seed)


def edges_to_mask(
    edges: Iterable[tuple[int, int]],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    edge_list = list(edges)
    if not edge_list:
        return mask
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device)
    mask[edge_index[:, 0], edge_index[:, 1]] = True
    return mask


def build_zigzag_cross(
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    graph_config: dict | None = None,
) -> torch.Tensor:
    return edges_to_mask(
        build_zigzag_cross_edges(seq_len, block_size, degree, graph_config),
        seq_len,
        device,
    )


def build_random_cross(
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    seed: int,
    graph_config: dict | None = None,
) -> torch.Tensor:
    return edges_to_mask(
        build_random_regular_cross_edges(seq_len, block_size, degree, seed),
        seq_len,
        device,
    )


def build_boolean_ablation_mask(
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    graph_config: dict | None,
) -> torch.Tensor:
    local = build_local_mask(seq_len, block_size, device)
    remote = build_zigzag_cross(seq_len, block_size, degree, device, graph_config)
    return local | remote


def canonical_method(method: str) -> str:
    aliases = {
        "random": "random_regular",
        "zigzag": "zigzag_cycle",
        "zigzag_certified_cosine": "zigzag_certified_cosine",
    }
    return aliases.get(method, method)


def build_attention_mask(
    method: str,
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    seed: int,
    graph_config: dict | None = None,
) -> torch.Tensor:
    method = canonical_method(method)
    if method == "dense":
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    if method == "zigzag_cycle":
        graph_config = DEFAULT_GRAPH_CONFIG
    else:
        validate_graph_config(seq_len, block_size, degree, graph_config)
    local = build_local_mask(seq_len, block_size, device)
    if method == "local":
        return local
    if method == "random_regular":
        return local | build_random_cross(seq_len, block_size, degree, device, seed, graph_config)
    if method == "zigzag_certified_cosine":
        method = "zigzag_certified"
    if method in {"zigzag_certified", "zigzag_boolean", "zigzag_cycle"}:
        return local | build_zigzag_cross(seq_len, block_size, degree, device, graph_config)
    raise ValueError(f"unknown method: {method}")


def build_cross_mask(
    method: str,
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    seed: int,
    graph_config: dict | None = None,
) -> torch.Tensor:
    method = canonical_method(method)
    local = build_local_mask(seq_len, block_size, device)
    if method in {"dense", "local"}:
        return torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    if method == "random_regular":
        return build_random_cross(seq_len, block_size, degree, device, seed, graph_config) & ~local
    if method == "zigzag_cycle":
        graph_config = DEFAULT_GRAPH_CONFIG
    if method == "zigzag_certified_cosine":
        method = "zigzag_certified"
    if method in {"zigzag_certified", "zigzag_boolean", "zigzag_cycle"}:
        return build_zigzag_cross(seq_len, block_size, degree, device, graph_config) & ~local
    raise ValueError(f"unknown method: {method}")


def expected_raw_k(method: str, seq_len: int, block_size: int, degree: int) -> int:
    method = canonical_method(method)
    if method == "dense":
        return seq_len
    if method == "local":
        return block_size
    return block_size + degree * degree


def counts_to_mask(rows: list[Counter[int]], seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for src, counts in enumerate(rows):
        if counts:
            dst = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
            mask[src, dst] = True
    return mask


def mask_metrics(mask: torch.Tensor, method: str, block_size: int, degree: int) -> dict:
    seq_len = mask.shape[0]
    raw_k = expected_raw_k(method, seq_len, block_size, degree)
    effective = mask.sum(dim=-1).float()
    duplicate_rate = max(0.0, (raw_k - float(effective.mean().item())) / raw_k)
    return {
        "raw_k": raw_k,
        "unique_k_mean": float(effective.mean().item()),
        "unique_k_min": int(effective.min().item()),
        "unique_k_max": int(effective.max().item()),
        "effective_k_mean": float(effective.mean().item()),
        "effective_k_min": int(effective.min().item()),
        "effective_k_max": int(effective.max().item()),
        "duplicate_rate": duplicate_rate,
        "duplicate_rate_estimate": duplicate_rate,
        "attention_pair_count": int(mask.sum().item()),
        "self_loop_rate": float(torch.diag(mask).float().mean().item()),
    }
