from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from probe_metrics import aggregate_metric_rows
from probe_tasks import JsonlStore, ProbeTransformer, load_encoder, make_probe_batch
from run_copy_corrected import deterministic_permutation, epoch_coverage
from run_probe_experiment import forward_loss_and_metrics
from synthetic_mvp_core.artifacts import make_attention_artifacts
from synthetic_mvp_core.artifacts import build_pure_random_rows_for_actual_mask_density
from synthetic_mvp_core.artifacts import build_random_remote_rows_for_multihop_copy_route
from synthetic_mvp_core.attention import dense_attention, local_blockpair_attention, local_cross_attention, neighbor_attention_from_table
from synthetic_mvp_core.model import MaskedSelfAttention, RotaryEmbedding, apply_rotary_pos_emb


BRANCH = "codex/copy-corrected-v01-l8-log5"
OLD_TEST_SHA256 = "50de40e9b6f7c53af8a912cf0967ae1129e84028bcc7f90c14a94620d0760fac"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def deployed_commit_exists() -> bool:
    for name in [".deployed_git_commit_v08", ".deployed_git_commit_copy_corrected_v01_l8_log5"]:
        marker = Path(name)
        if marker.exists() and marker.read_text(encoding="utf-8").strip():
            return True
    return False


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def record_from_manifest(path: Path) -> dict:
    manifest = read_json(path)
    return manifest["tasks"][0]


def test_static_contract(config: dict, record: dict) -> None:
    try:
        branch = git_value("branch", "--show-current")
    except subprocess.CalledProcessError:
        branch = ""
    require(branch == BRANCH or deployed_commit_exists(), "must run on corrected Copy branch or deployed marker")
    cwd = str(Path.cwd())
    require(
        "expander-copy-corrected-v01-l8-log5" in cwd
        or "zigzag_attention_copy_corrected_v01_l8_log5" in cwd
        or deployed_commit_exists(),
        "must run in corrected Copy l8/log5 worktree or deployed directory",
    )
    require(record.get("copy_corrected_variant") == "copy_corrected_v01_l8_log5", "variant must be l8/log5")
    require(record["copy_corrected_v01"] is True, "task record must be marked corrected")
    require(record["resolved_padded_sequence_length"] == 2048, "T must be 2048")
    require(record["resolved_runtime_input_length"] == 2048, "runtime input length must be 2048")
    require(record["resolved_runtime_target_length"] == 1024, "target length must be 1024")
    require(record["resolved_readout_start"] == 1024, "readout start must be 1024")
    require(record["resolved_vocab_or_value_space_size"] == 64, "vocab size must be 64")
    require(record["resolved_token_output_size"] == 64, "token output size must be 64")
    require(record["position_encoding"] == "rope", "position encoding must be RoPE")
    require(record["rope_learnable"] is False, "RoPE must be nonlearnable")
    require(record["absolute_position_embedding"] == "none", "absolute pos embedding must be absent")
    require(record["resolved_graph_block_size"] == 64, "graph B must be 64")
    require(record["resolved_graph_num_blocks_or_nodes"] == 32, "graph q must be 32")
    require(record["resolved_layers"] == 8, "model layers must be 8")
    require(config["train"]["log_every"] == 5, "train log_every must be 5")
    require(record["discarded_old_test_sha256"] == OLD_TEST_SHA256, "discarded old test hash must be recorded")
    require(OLD_TEST_SHA256 not in json.dumps(config), "old OOD test hash must not appear in config")
    data_dir = Path(record["version_path"])
    require((data_dir / "train.jsonl").exists(), "train.jsonl missing")
    require((data_dir / "test.jsonl").exists(), "test.jsonl missing")
    require(not (data_dir / "validation.jsonl").exists(), "validation.jsonl must not exist")


def test_dataset(record: dict) -> None:
    data_dir = Path(record["version_path"])
    train_ids: set[str] = set()
    train_targets: set[tuple[int, ...]] = set()
    for split, expected_rows in [("train", 10_000), ("test", 1_000)]:
        ids: set[str] = set()
        targets: set[tuple[int, ...]] = set()
        rows = 0
        for row in iter_jsonl(data_dir / f"{split}.jsonl"):
            rows += 1
            ids.add(str(row["id"]))
            target_tuple = tuple(int(v) for v in row["target"])
            targets.add(target_tuple)
            require(len(row["input"]) == 2048, f"{split} input length")
            require(len(row["target"]) == 1024, f"{split} target length")
            require(row["input"][:1024] == row["target"], f"{split} source==target")
            require(row["input"][1024:] == [63] * 1024, f"{split} marker suffix")
            require(min(row["target"]) >= 1 and max(row["target"]) <= 62, f"{split} target range")
            require(min(row["input"]) >= 1 and max(row["input"]) <= 63, f"{split} input range")
        require(rows == expected_rows, f"{split} row count")
        require(len(ids) == expected_rows, f"{split} ids unique")
        require(len(targets) == expected_rows, f"{split} targets unique")
        if split == "train":
            train_ids = ids
            train_targets = targets
        else:
            require(not (train_ids & ids), "train/test ids overlap")
            require(not (train_targets & targets), "train/test targets overlap")


def test_batch_and_loss(record: dict) -> None:
    store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    rows = [store.row(0), store.row(1)]
    batch = make_probe_batch(rows, record, encoder, torch.device("cpu"))
    require(tuple(batch.tokens.shape) == (2, 2048), "tokens shape")
    require(tuple(batch.targets.shape) == (2, 1024), "targets shape")
    require(tuple(batch.target_positions.shape) == (2, 1024), "target positions shape")
    require(bool(batch.target_mask.all()), "target mask all true")
    require(bool(batch.pad_mask.all()), "valid token mask all true")
    require(torch.equal(batch.target_positions.cpu(), torch.arange(1024, 2048).repeat(2, 1)), "target positions")
    require(torch.equal(batch.tokens[:, :1024].cpu(), batch.targets.cpu()), "targets equal source")
    require(torch.equal(batch.tokens[:, 1024:].cpu(), torch.full((2, 1024), 63)), "marker suffix")
    model = ProbeTransformer(
        vocab_size=64,
        token_output_size=64,
        class_count=2,
        seq_len=2048,
        d_model=32,
        layers=1,
        heads=4,
        ffn_dim=64,
        dropout=0.0,
        attention_backend="dense_mask",
        block_size=64,
        position_encoding="rope",
        use_class_head=False,
    )
    mask = torch.ones((2048, 2048), dtype=torch.bool)
    local_valid = torch.ones((2048, 64), dtype=torch.bool)
    artifacts = SimpleNamespace(
        mask=mask,
        local_valid=local_valid,
        neighbors=None,
        valid_neighbors=None,
        block_pair_index=None,
        local_log_m=None,
        neighbor_log_m=None,
        route_transport_src=None,
        route_transport_dst=None,
        route_transport_scale=None,
        route_transport_mode=None,
    )
    loss, _metrics, _per_sample = forward_loss_and_metrics(model, artifacts, batch, record)
    require(torch.isfinite(loss), "loss finite")
    with torch.no_grad():
        token_logits, _ = model(batch.tokens, batch.pad_mask, mask, local_valid)
        targets = batch.targets
        marker_loss_1 = torch.nn.functional.cross_entropy(token_logits[:, 1024:2048, :].reshape(-1, 64), targets.reshape(-1), reduction="sum")
        modified_source = token_logits.clone()
        modified_source[:, :1024, :] += torch.randn_like(modified_source[:, :1024, :]) * 100.0
        marker_loss_2 = torch.nn.functional.cross_entropy(modified_source[:, 1024:2048, :].reshape(-1, 64), targets.reshape(-1), reduction="sum")
        modified_marker = token_logits.clone()
        modified_marker[:, 1024:2048, :] += torch.randn_like(modified_marker[:, 1024:2048, :]) * 100.0
        marker_loss_3 = torch.nn.functional.cross_entropy(modified_marker[:, 1024:2048, :].reshape(-1, 64), targets.reshape(-1), reduction="sum")
    require(torch.allclose(marker_loss_1, marker_loss_2), "source logits must not directly affect loss")
    require(not torch.allclose(marker_loss_1, marker_loss_3), "marker logits must affect loss")


def test_rope(record: dict) -> None:
    model = ProbeTransformer(
        vocab_size=64,
        token_output_size=64,
        class_count=2,
        seq_len=2048,
        d_model=int(record["resolved_d_model"]),
        layers=int(record["resolved_layers"]),
        heads=int(record["resolved_heads"]),
        ffn_dim=int(record["resolved_ffn_dim"]),
        dropout=0.0,
        attention_backend="dense_mask",
        block_size=64,
        position_encoding="rope",
        use_class_head=False,
    )
    names = [name for name, _ in model.named_parameters()]
    require(not any("pos" in name.lower() for name in names), "learned position parameters must be absent")
    require(not any("class_head" in name for name in names), "unused class head must be absent")
    inv_freqs = [buf for name, buf in model.named_buffers() if name.endswith("inv_freq")]
    require(inv_freqs, "RoPE inv_freq buffers must exist")
    require(all(not buf.requires_grad for buf in inv_freqs), "RoPE buffers must not require grad")
    try:
        ProbeTransformer(64, 64, 2, 2048, 30, 1, 2, 64, 0.0, "dense_mask", 64, position_encoding="rope", use_class_head=False)
    except ValueError:
        pass
    else:
        raise AssertionError("odd head_dim should fail")
    q_base = torch.randn(1, 2, 1, 16).expand(1, 2, 8, 16).clone()
    k_base = torch.randn(1, 2, 1, 16).expand(1, 2, 8, 16).clone()
    rope = RotaryEmbedding(16)
    cos, sin = rope(8, q_base.device, q_base.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q_base, k_base, cos, sin)
    require(not torch.allclose(q_base, q_rot), "Q should rotate")
    require(not torch.allclose(k_base, k_rot), "K should rotate")
    dot_0_3 = (q_rot[:, :, 0, :] * k_rot[:, :, 3, :]).sum(-1)
    dot_2_5 = (q_rot[:, :, 2, :] * k_rot[:, :, 5, :]).sum(-1)
    require(torch.allclose(dot_0_3, dot_2_5, atol=1e-5), "RoPE dot product should depend on relative offset")
    require(not any(key.endswith("pos.weight") for key in model.state_dict()), "state dict must not contain learned pos table")


def test_reachability(record: dict) -> None:
    reach = read_json(Path(record["reachability_path"]))
    require(reach["dense"]["target_in_1hop_rate"] == 1.0, "dense 1-hop reachability")
    require(reach["zigzag_certified"]["target_in_Lhop_rate"] == 1.0, "zigzag L-hop reachability")
    require(reach["random_regular"]["target_in_Lhop_rate"] == 1.0, "random L-hop reachability")
    require(reach["local"]["unreachable_rate"] == 1.0, "local negative control unreachable")
    require(reach["random_regular"]["random_k_alignment_error_max"] == 0, "random K alignment max")


def test_sampler() -> None:
    p0 = deterministic_permutation(10_000, 0, 0)
    p1 = deterministic_permutation(10_000, 0, 0)
    p2 = deterministic_permutation(10_000, 1, 0)
    require(p0 == p1, "same data seed permutation")
    require(p0 != p2, "different data seed permutation")
    cov = epoch_coverage(p0, 10_000)
    require(cov == {"draw_count": 10000, "unique_count": 10000, "never_seen": 0, "max_repeat_count": 1}, f"coverage {cov}")


def test_metrics() -> None:
    rows = [
        {"examples": 1, "tokens": 2, "loss_sum": 2.0, "copy_token_accuracy": 0.5, "copy_sequence_accuracy": 0.0},
        {"examples": 1, "tokens": 4, "loss_sum": 4.0, "copy_token_accuracy": 1.0, "copy_sequence_accuracy": 1.0},
    ]
    agg = aggregate_metric_rows(rows, "copy_token_accuracy")
    require(abs(agg["loss"] - 1.0) < 1e-12, "loss sum/token aggregation")
    require(abs(agg["primary_metric_value"] - (5.0 / 6.0)) < 1e-12, "token accuracy token-weighted")
    require(abs(agg["secondary_metrics"]["copy_sequence_accuracy"] - 0.5) < 1e-12, "sequence accuracy example-weighted")


def test_backend_consistency() -> None:
    torch.manual_seed(0)
    seq_len = 128
    block_size = 16
    q = torch.randn(2, 2, seq_len, 8)
    k = torch.randn(2, 2, seq_len, 8)
    v = torch.randn(2, 2, seq_len, 8)
    args = SimpleNamespace(block_size=block_size, degree=4, causal=False, graph_config=None, seed=0, multiplicity_mode="boolean")
    split_artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "split")
    dense_ref = dense_attention(q, k, v, split_artifacts.mask)
    neighbor_artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "neighbor")
    neighbor_out = neighbor_attention_from_table(q, k, v, neighbor_artifacts.neighbors, neighbor_artifacts.valid_neighbors)
    split_out = local_cross_attention(q, k, v, block_size, split_artifacts.local_valid, split_artifacts.neighbors, split_artifacts.valid_neighbors)
    blockpair_artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "blockpair")
    blockpair_out = local_blockpair_attention(q, k, v, block_size, blockpair_artifacts.local_valid, blockpair_artifacts.neighbors, blockpair_artifacts.valid_neighbors, blockpair_artifacts.block_pair_index)
    require(torch.allclose(dense_ref, neighbor_out, atol=1e-5, rtol=1e-5), "neighbor backend mismatch")
    require(torch.allclose(dense_ref, split_out, atol=1e-5, rtol=1e-5), "split backend mismatch")
    require(torch.allclose(dense_ref, blockpair_out, atol=1e-5, rtol=1e-5), "blockpair backend mismatch")
    row, col = torch.where(split_artifacts.mask)
    require(bool(split_artifacts.mask[row[0], col[0]]), "mask direction query row/key col")


def test_value_rope_backend_consistency() -> None:
    torch.manual_seed(0)
    seq_len = 128
    block_size = 16
    d_model = 32
    heads = 4
    x = torch.randn(2, seq_len, d_model)
    args = SimpleNamespace(block_size=block_size, degree=4, causal=False, graph_config=None, seed=0, multiplicity_mode="boolean")
    artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "split")
    dense_attn = MaskedSelfAttention(
        d_model,
        heads,
        0.0,
        "dense_mask",
        block_size,
        position_encoding="rope",
        value_position_encoding="rope_relative",
    )
    split_attn = MaskedSelfAttention(
        d_model,
        heads,
        0.0,
        "split",
        block_size,
        position_encoding="rope",
        value_position_encoding="rope_relative",
    )
    split_attn.load_state_dict(dense_attn.state_dict())
    dense_out = dense_attn(
        x,
        artifacts.mask,
        artifacts.local_valid,
        None,
        None,
        None,
        None,
        None,
    )
    split_out = split_attn(
        x,
        artifacts.mask,
        artifacts.local_valid,
        artifacts.neighbors,
        artifacts.valid_neighbors,
        None,
        None,
        None,
    )
    require(torch.allclose(dense_out, split_out, atol=1e-5, rtol=1e-5), "value-RoPE split backend mismatch")
    try:
        MaskedSelfAttention(
            d_model,
            heads,
            0.0,
            "dense_mask",
            block_size,
            position_encoding="learned_absolute",
            value_position_encoding="rope_relative",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("value-RoPE must require RoPE position encoding")


def test_headwise_log_bias_backend_consistency() -> None:
    torch.manual_seed(1)
    seq_len = 128
    block_size = 16
    heads = 4
    head_dim = 8
    q = torch.randn(2, heads, seq_len, head_dim)
    k = torch.randn(2, heads, seq_len, head_dim)
    v = torch.randn(2, heads, seq_len, head_dim)
    args = SimpleNamespace(block_size=block_size, degree=4, causal=False, graph_config=None, seed=0, multiplicity_mode="boolean")
    artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "split")
    query = torch.arange(seq_len)
    key = torch.arange(seq_len)
    relative_table = torch.randn(heads, 2 * seq_len - 1) * 0.01
    full_bias = relative_table[:, key[None, :] - query[:, None] + seq_len - 1]
    offsets = torch.arange(block_size)
    block_starts = (query // block_size) * block_size
    local_positions = block_starts[:, None] + offsets[None, :]
    local_bias = relative_table[:, local_positions - query[:, None] + seq_len - 1]
    cross_bias = relative_table[:, artifacts.neighbors - query[:, None] + seq_len - 1]
    dense_out = dense_attention(q, k, v, artifacts.mask, full_bias)
    split_out = local_cross_attention(
        q,
        k,
        v,
        block_size,
        artifacts.local_valid,
        artifacts.neighbors,
        artifacts.valid_neighbors,
        local_log_m=local_bias,
        cross_log_m=cross_bias,
    )
    require(torch.allclose(dense_out, split_out, atol=1e-5, rtol=1e-5), "headwise log bias split mismatch")


def test_attention_logit_scale_backend_consistency() -> None:
    torch.manual_seed(2)
    seq_len = 128
    block_size = 16
    d_model = 32
    heads = 4
    x = torch.randn(2, seq_len, d_model)
    args = SimpleNamespace(block_size=block_size, degree=4, causal=False, graph_config=None, seed=0, multiplicity_mode="boolean")
    artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "split")
    dense_attn = MaskedSelfAttention(
        d_model,
        heads,
        0.0,
        "dense_mask",
        block_size,
        position_encoding="rope",
        attention_logit_scale_multiplier=2.0,
    )
    split_attn = MaskedSelfAttention(
        d_model,
        heads,
        0.0,
        "split",
        block_size,
        position_encoding="rope",
        attention_logit_scale_multiplier=2.0,
    )
    split_attn.load_state_dict(dense_attn.state_dict())
    dense_out = dense_attn(x, artifacts.mask, artifacts.local_valid, None, None, None, None, None)
    split_out = split_attn(
        x,
        artifacts.mask,
        artifacts.local_valid,
        artifacts.neighbors,
        artifacts.valid_neighbors,
        None,
        None,
        None,
    )
    require(torch.allclose(dense_out, split_out, atol=1e-5, rtol=1e-5), "scaled-logit split backend mismatch")
    default_attn = MaskedSelfAttention(d_model, heads, 0.0, "dense_mask", block_size, position_encoding="rope")
    scaled_one_attn = MaskedSelfAttention(
        d_model,
        heads,
        0.0,
        "dense_mask",
        block_size,
        position_encoding="rope",
        attention_logit_scale_multiplier=1.0,
    )
    scaled_one_attn.load_state_dict(default_attn.state_dict())
    default_out = default_attn(x, artifacts.mask, artifacts.local_valid, None, None, None, None, None)
    scaled_one_out = scaled_one_attn(x, artifacts.mask, artifacts.local_valid, None, None, None, None, None)
    require(torch.allclose(default_out, scaled_one_out, atol=1e-6, rtol=1e-6), "scale=1 should match default")


def test_random_multihop_copy_route_structure() -> None:
    seq_len = 2048
    source_len = 1024
    layers = 8
    stride = 128
    density = 0.1
    route_multiplicity = 65536
    args = SimpleNamespace(
        block_size=64,
        degree=32,
        causal=False,
        graph_config=None,
        seed=0,
        random_layer_index=0,
        random_multihop_copy_route=True,
        random_route_layers=layers,
        random_route_source_length=source_len,
        random_route_target_start=source_len,
        random_route_stride=stride,
        random_route_multiplicity=route_multiplicity,
        random_use_log_m=True,
        multiplicity_mode="unique_log_m",
    )
    args.random_aligned_rows = build_random_remote_rows_for_multihop_copy_route(seq_len, args, density)
    artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "split")
    expected_pairs = int(round(density * seq_len * seq_len))
    require(int(artifacts.mask.sum().item()) == expected_pairs, "multihop route must preserve actual density")
    require(artifacts.metrics["random_multihop_copy_route"] is True, "route metric missing")
    require(artifacts.metrics["random_route_edge_missing"] == 0, "route edges missing")
    require(artifacts.metrics["random_route_edge_count"] == seq_len - stride, "route edge count")
    require(artifacts.metrics["random_direct_copy_edge_count"] == 0, "direct copy shortcuts must be absent")
    for offset in range(source_len):
        target = source_len + offset
        source = offset
        require(not bool(artifacts.mask[target, source].item()), "target must not have one-hop direct source edge")
        pos = target
        for _hop in range(layers):
            next_pos = pos - stride
            require(bool(artifacts.mask[pos, next_pos].item()), "missing route edge")
            pos = next_pos
        require(pos == source, "route must land on matching source after L hops")

    route_row = source_len
    route_dst = route_row - stride
    slots = torch.nonzero(artifacts.neighbors[route_row] == route_dst, as_tuple=False).flatten()
    require(int(slots.numel()) == 1, "route edge should appear once in boolean neighbors")
    slot = int(slots[0].item())
    require(bool(artifacts.valid_neighbors[route_row, slot].item()), "route neighbor slot should be valid")
    expected_log_m = torch.tensor(float(route_multiplicity)).log()
    observed_log_m = artifacts.neighbor_log_m[route_row, slot]
    require(torch.allclose(observed_log_m, expected_log_m), "route multiplicity log-bias missing")


def test_pure_random_no_local_actual_density_structure() -> None:
    seq_len = 2048
    block_size = 64
    density = 0.1
    args = SimpleNamespace(
        block_size=block_size,
        degree=32,
        causal=False,
        graph_config=None,
        seed=0,
        random_layer_index=0,
        random_include_local_edges=False,
        multiplicity_mode="unique_log_m",
    )
    args.random_aligned_rows = build_pure_random_rows_for_actual_mask_density(
        seq_len,
        args,
        density,
        exclude_block_local=True,
    )
    artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "split")
    expected_pairs = int(round(density * seq_len * seq_len))
    require(int(artifacts.mask.sum().item()) == expected_pairs, "pure random must preserve actual density")
    require(artifacts.metrics["local_attention_pair_count"] == 0, "pure random must have no block-local pairs")
    require(artifacts.metrics["remote_attention_pair_count"] == expected_pairs, "all pure random pairs must be remote")
    row_k = artifacts.mask.sum(dim=1)
    require(int(row_k.min().item()) == expected_pairs // seq_len, "pure random min row k")
    require(int(row_k.max().item()) == expected_pairs // seq_len + 1, "pure random max row k")
    local_widths = artifacts.local_valid.sum(dim=1)
    require(int(local_widths.max().item()) == 0, "local_valid must be empty for pure random no-local")


def test_random_multihop_copy_route_layerwise_staged_structure() -> None:
    seq_len = 2048
    source_len = 1024
    layers = 8
    stride = 128
    block_size = 64
    density = 0.1
    route_multiplicity = 65536
    layer_artifacts = []
    for layer_index in range(layers):
        args = SimpleNamespace(
            block_size=block_size,
            degree=32,
            causal=False,
            graph_config=None,
            seed=0,
            random_layer_index=layer_index,
            random_multihop_copy_route=True,
            random_route_layerwise_staged=True,
            random_route_layers=layers,
            random_route_source_length=source_len,
            random_route_target_start=source_len,
            random_route_stride=stride,
            random_route_multiplicity=route_multiplicity,
            random_use_log_m=True,
            random_route_transport=True,
            random_route_transport_scale=1.0,
            random_route_transport_mode="memory_replace",
            multiplicity_mode="unique_log_m",
        )
        args.random_aligned_rows = build_random_remote_rows_for_multihop_copy_route(seq_len, args, density)
        artifacts = make_attention_artifacts("random_regular", seq_len, args, torch.device("cpu"), "split")
        layer_artifacts.append(artifacts)
        expected_pairs = int(round(density * seq_len * seq_len))
        require(int(artifacts.mask.sum().item()) == expected_pairs, "staged route must preserve actual density")
        require(artifacts.metrics["random_multihop_copy_route"] is True, "staged route metric missing")
        require(artifacts.metrics["random_route_edge_missing"] == 0, "staged route edges missing")
        require(artifacts.metrics["random_route_edge_count"] == source_len, "staged route edge count")
        require(artifacts.metrics["random_route_transport"] is True, "staged transport metric missing")
        require(artifacts.route_transport_src is not None, "staged transport src missing")
        require(artifacts.route_transport_dst is not None, "staged transport dst missing")
        require(artifacts.route_transport_mode == "memory_replace", "staged transport mode")
        require(artifacts.metrics["random_route_transport_mode"] == "memory_replace", "staged transport mode metric")
        require(int(artifacts.route_transport_src.numel()) == source_len, "staged transport src count")
        require(int(artifacts.route_transport_dst.numel()) == source_len, "staged transport dst count")
        require(artifacts.metrics["random_direct_copy_edge_count"] == 0, "staged direct copy shortcuts must be absent")
        for offset in range(source_len):
            target = source_len + offset
            require(not bool(artifacts.mask[target, offset].item()), "staged target must not have one-hop direct source edge")

    def stage_position(stage: int, offset: int) -> int:
        if stage <= 0:
            return offset
        if stage >= layers:
            return source_len + offset
        return source_len + ((offset + stage * stride) % source_len)

    for offset in range(source_len):
        pos = source_len + offset
        for layer_index in reversed(range(layers)):
            expected_dst = stage_position(layer_index, offset)
            require(bool(layer_artifacts[layer_index].mask[pos, expected_dst].item()), "missing staged layer route edge")
            pos = expected_dst
        require(pos == offset, "staged route must land on matching source after L hops")


def test_route_transport_memory_replace_forward() -> None:
    model = ProbeTransformer(
        vocab_size=8,
        token_output_size=8,
        class_count=2,
        seq_len=4,
        d_model=8,
        layers=2,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
        attention_backend="dense_mask",
        block_size=4,
        position_encoding="rope",
        use_class_head=False,
    )
    model.eval()
    with torch.no_grad():
        for block in model.blocks:
            block.attn.qkv.weight.zero_()
            block.attn.qkv.bias.zero_()
            block.attn.out.weight.zero_()
            block.attn.out.bias.zero_()
            for module in block.ffn:
                if hasattr(module, "weight"):
                    module.weight.zero_()
                if hasattr(module, "bias"):
                    module.bias.zero_()
        model.norm.weight.fill_(1.0)
        model.norm.bias.zero_()
        model.token.weight.zero_()
        for token_id in range(8):
            model.token.weight[token_id, token_id] = 1.0
        model.token_head.weight.zero_()
        model.token_head.bias.zero_()
        for token_id in range(8):
            model.token_head.weight[token_id, token_id] = 1.0

    tokens = torch.tensor([[3, 1, 6, 7]])
    pad_mask = torch.ones((1, 4), dtype=torch.bool)
    mask = torch.ones((4, 4), dtype=torch.bool)
    local_valid = torch.ones((4, 4), dtype=torch.bool)
    route_src = [torch.tensor([2]), torch.tensor([3])]
    route_dst = [torch.tensor([0]), torch.tensor([2])]
    logits, _ = model(
        tokens,
        pad_mask,
        mask,
        local_valid,
        route_transport_src=route_src,
        route_transport_dst=route_dst,
        route_transport_scale=[1.0, 1.0],
        route_transport_mode=["memory_replace", "memory_replace"],
    )
    pred = int(logits[0, 3].argmax().item())
    require(pred == 3, "memory_replace transport should carry token 3 over two route hops")


def test_learned_edge_memory_transport_forward() -> None:
    model = ProbeTransformer(
        vocab_size=8,
        token_output_size=8,
        class_count=2,
        seq_len=4,
        d_model=8,
        layers=2,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
        attention_backend="split",
        block_size=2,
        position_encoding="rope",
        use_class_head=False,
        edge_bias_shapes=[(4, 2, 1), (4, 2, 1)],
        learned_edge_memory_transport_mode="replace",
        learned_edge_memory_transport_scale=1.0,
        learned_edge_memory_transport_temperature=1.0,
        learned_edge_bias_init=-20.0,
    )
    model.eval()
    with torch.no_grad():
        for block in model.blocks:
            block.attn.qkv.weight.zero_()
            block.attn.qkv.bias.zero_()
            block.attn.out.weight.zero_()
            block.attn.out.bias.zero_()
            for module in block.ffn:
                if hasattr(module, "weight"):
                    module.weight.zero_()
                if hasattr(module, "bias"):
                    module.bias.zero_()
        model.norm.weight.fill_(1.0)
        model.norm.bias.zero_()
        model.token.weight.zero_()
        for token_id in range(8):
            model.token.weight[token_id, token_id] = 1.0
        model.token_head.weight.zero_()
        model.token_head.bias.zero_()
        for token_id in range(8):
            model.token_head.weight[token_id, token_id] = 1.0
        for local_bias, neighbor_bias in zip(model.learned_local_edge_log_bias, model.learned_neighbor_edge_log_bias):
            local_bias.fill_(-20.0)
            neighbor_bias.fill_(-20.0)
        model.learned_neighbor_edge_log_bias[0][2, 0] = 20.0
        model.learned_neighbor_edge_log_bias[1][3, 0] = 20.0

    tokens = torch.tensor([[3, 1, 6, 7]])
    pad_mask = torch.ones((1, 4), dtype=torch.bool)
    mask = torch.ones((4, 4), dtype=torch.bool)
    local_valid = [
        torch.ones((4, 2), dtype=torch.bool),
        torch.ones((4, 2), dtype=torch.bool),
    ]
    neighbors = [
        torch.tensor([[0], [0], [0], [2]], dtype=torch.long),
        torch.tensor([[0], [0], [0], [2]], dtype=torch.long),
    ]
    valid_neighbors = [
        torch.ones((4, 1), dtype=torch.bool),
        torch.ones((4, 1), dtype=torch.bool),
    ]
    logits, _ = model(
        tokens,
        pad_mask,
        mask,
        local_valid,
        neighbors=neighbors,
        valid_neighbors=valid_neighbors,
    )
    pred = int(logits[0, 3].argmax().item())
    require(pred == 3, "learned edge memory transport should carry token 3 over learned random edges")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/copy_corrected_v01.json"))
    args = parser.parse_args()
    config = read_json(args.config)
    manifest_path = Path(config["task_parameter_manifest"])
    record = record_from_manifest(manifest_path)
    tests = [
        ("static", lambda: test_static_contract(config, record)),
        ("dataset", lambda: test_dataset(record)),
        ("batch_loss", lambda: test_batch_and_loss(record)),
        ("rope", lambda: test_rope(record)),
        ("reachability", lambda: test_reachability(record)),
        ("sampler", test_sampler),
        ("metrics", test_metrics),
        ("backend", test_backend_consistency),
        ("value_rope_backend", test_value_rope_backend_consistency),
        ("headwise_log_bias_backend", test_headwise_log_bias_backend_consistency),
        ("attention_logit_scale_backend", test_attention_logit_scale_backend_consistency),
        ("random_multihop_route", test_random_multihop_copy_route_structure),
        ("pure_random_no_local_actual_density", test_pure_random_no_local_actual_density_structure),
        ("random_multihop_route_staged", test_random_multihop_copy_route_layerwise_staged_structure),
        ("route_transport_memory_replace_forward", test_route_transport_memory_replace_forward),
        ("learned_edge_memory_transport_forward", test_learned_edge_memory_transport_forward),
    ]
    passed = []
    for name, fn in tests:
        fn()
        passed.append(name)
        print(json.dumps({"test": name, "status": "passed"}), flush=True)
    print(json.dumps({"status": "passed", "tests": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
