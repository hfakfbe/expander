from __future__ import annotations


def certificate_summary(graph_records: list[dict]) -> dict:
    return {
        "graph_artifact_count": len([item for item in graph_records if item.get("layer_index") != "manifest"]),
        "all_artifacts_have_sha256": all(bool(item.get("sha256")) for item in graph_records),
    }

