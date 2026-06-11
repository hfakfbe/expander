from __future__ import annotations

import torch


DEFAULT_GRAPH_CONFIG = {
    "G": {"type": "cyclic"},
    "H": {"type": "cycle"},
}


def _graph_section(graph_config: dict | None, name: str) -> dict:
    if graph_config is None:
        graph_config = DEFAULT_GRAPH_CONFIG
    return dict(graph_config.get(name, DEFAULT_GRAPH_CONFIG[name]))


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

    g_config = _graph_section(graph_config, "G")
    h_config = _graph_section(graph_config, "H")
    if g_config.get("type") != "cyclic":
        raise ValueError(f"unsupported G graph type: {g_config.get('type')}")
    if h_config.get("type") != "cycle":
        raise ValueError(f"unsupported H graph type: {h_config.get('type')}")
    return {"G": g_config, "H": h_config}


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
    if h_config.get("type") != "cycle":
        raise ValueError(f"unsupported H graph type: {h_config.get('type')}")
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
    if g_config.get("type") != "cyclic":
        raise ValueError(f"unsupported G graph type: {g_config.get('type')}")
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


def build_zigzag_cross_edges(
    seq_len: int,
    block_size: int,
    degree: int,
    graph_config: dict | None = None,
) -> list[tuple[int, int]]:
    graph_config = validate_graph_config(seq_len, block_size, degree, graph_config)
    num_blocks = seq_len // block_size
    edges: list[tuple[int, int]] = []
    for v in range(num_blocks):
        for i in range(block_size):
            src = v * block_size + i
            for i_prime in h_neighbors(i, block_size, degree, graph_config):
                w, j_prime = rot_g(v, i_prime, num_blocks, block_size, graph_config)
                for j in h_neighbors(j_prime, block_size, degree, graph_config):
                    dst = w * block_size + j
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
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    edges: list[tuple[int, int]] = []
    k = degree * degree
    all_idx = torch.arange(seq_len)
    for src in range(seq_len):
        block = src // block_size
        nonlocal_idx = all_idx[all_idx // block_size != block]
        perm = torch.randperm(len(nonlocal_idx), generator=gen)[:k]
        edges.extend((src, int(dst)) for dst in nonlocal_idx[perm].tolist())
    return edges


def edges_to_mask(
    edges: list[tuple[int, int]],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    if not edges:
        return mask
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
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
        build_random_cross_edges(seq_len, block_size, degree, seed, graph_config),
        seq_len,
        device,
    )


def build_attention_mask(
    method: str,
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    seed: int,
    graph_config: dict | None = None,
) -> torch.Tensor:
    validate_graph_config(seq_len, block_size, degree, graph_config)
    if method == "dense":
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    local = build_local_mask(seq_len, block_size, device)
    if method == "local":
        return local
    if method == "random":
        return local | build_random_cross(seq_len, block_size, degree, device, seed, graph_config)
    if method == "zigzag":
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
    local = build_local_mask(seq_len, block_size, device)
    if method in {"dense", "local"}:
        return torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    if method == "random":
        return build_random_cross(seq_len, block_size, degree, device, seed, graph_config) & ~local
    if method == "zigzag":
        return build_zigzag_cross(seq_len, block_size, degree, device, graph_config) & ~local
    raise ValueError(f"unknown method: {method}")


def expected_raw_k(method: str, seq_len: int, block_size: int, degree: int) -> int:
    if method == "dense":
        return seq_len
    if method == "local":
        return block_size
    return block_size + degree * degree


def mask_metrics(mask: torch.Tensor, method: str, block_size: int, degree: int) -> dict:
    seq_len = mask.shape[0]
    raw_k = expected_raw_k(method, seq_len, block_size, degree)
    effective = mask.sum(dim=-1).float()
    duplicate_rate = max(0.0, (raw_k - float(effective.mean().item())) / raw_k)
    return {
        "raw_k": raw_k,
        "effective_k_mean": float(effective.mean().item()),
        "effective_k_min": int(effective.min().item()),
        "effective_k_max": int(effective.max().item()),
        "duplicate_rate": duplicate_rate,
        "duplicate_rate_estimate": duplicate_rate,
        "attention_pair_count": int(mask.sum().item()),
        "self_loop_rate": float(torch.diag(mask).float().mean().item()),
    }
