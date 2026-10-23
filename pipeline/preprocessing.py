"""Check the stored degree and cardinality labels before a run, and conversion receipts."""

from collections import Counter
import json
from pathlib import Path

from .artifacts import atomic_json_write, canonical_json_hash, sha256_file

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _portable_path(path):
    """Return ``path`` relative to the repository root, or its file name if outside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def write_conversion_receipt(output_file, labels_file, *, source_file, source_hash,
                             source_kind, converter_file, label_mode, vertex_dims,
                             edge_dims, graphs, isolate_policy):
    """Store a conversion recipe after successful output, preserving source identity."""
    atomic_json_write(str(output_file) + ".preprocessing.json", {
        "schema_version": 1,
        "source": {"path": _portable_path(source_file), "sha256": source_hash,
                   "kind": source_kind},
        "converter_sha256": sha256_file(converter_file),
        "hypergraph_sha256": sha256_file(output_file),
        "labels_sha256": sha256_file(labels_file) if labels_file is not None else None,
        "label_mode": label_mode, "vertex_composite_dims": vertex_dims,
        "edge_composite_dims": edge_dims, "graphs": graphs,
        "isolate_policy": isolate_policy,
        "graph_order": "preserved from source",
    })


def verify_preprocessing(hypergraphs, labels, config, hypergraph_file, labels_file):
    """Recompute every vertex degree and hyperedge cardinality label from incidence.

    This checks the converted inputs, not the raw source files, and does not
    modify them. The split manifests separately freeze graph and label order.
    """
    mode = config["_label_mode"]
    if len(hypergraphs) != len(labels) or not hypergraphs:
        raise ValueError("Nonempty hypergraphs and labels must have equal lengths")
    if any(int(label) != label or not 0 <= label < config["num_classes"] for label in labels):
        raise ValueError("Labels must be zero-based integers within num_classes")
    isolated = 0
    for graph_id, graph in enumerate(hypergraphs):
        degree = {vertex: 0 for vertex in graph.vertex_label}
        for edge in graph.hyper_edges:
            for vertex in edge.vertices:
                degree[vertex] += 1
        isolated += sum(value == 0 for value in degree.values())
        for vertex, value in degree.items():
            if graph.vertex_label[vertex] != value:
                raise ValueError(
                    f"Graph {graph_id}, vertex {vertex}: label is not the vertex degree; "
                    "regenerate the data and split manifests"
                )
        for edge in graph.hyper_edges:
            if edge.label != len(edge.vertices):
                raise ValueError(
                    f"Graph {graph_id}, edge {edge.id}: label is not the hyperedge cardinality"
                )
    conversion = None
    receipt_path = Path(str(hypergraph_file) + ".preprocessing.json")
    if receipt_path.is_file():
        with receipt_path.open(encoding="utf-8") as handle:
            conversion = json.load(handle)
        expected = {
            "schema_version": 1, "hypergraph_sha256": sha256_file(hypergraph_file),
            "label_mode": mode, "vertex_composite_dims": 1,
            "edge_composite_dims": 1, "graphs": len(hypergraphs),
        }
        if any(conversion.get(key) != value for key, value in expected.items()):
            raise ValueError("Preprocessing receipt does not match the converted input")
        if conversion.get("labels_sha256") not in (None, sha256_file(labels_file)):
            raise ValueError("Preprocessing receipt label order/content has changed")
    return {
        "schema_version": 1,
        "verification": "recomputed_from_converted_incidence",
        "raw_source_verified": False,
        "conversion_receipt": conversion,
        "dataset": config["_dataset"], "label_mode": mode,
        "vertex_composite_dims": 1, "edge_composite_dims": 1,
        "hypergraph_sha256": sha256_file(hypergraph_file),
        "labels_sha256": sha256_file(labels_file),
        "ordered_labels_sha256": canonical_json_hash(list(map(int, labels))),
        "graphs": len(hypergraphs),
        "class_counts": dict(Counter(map(int, labels))),
        "isolated_vertices_present": isolated,
        "isolate_policy": "preserve vertices present in converted input",
    }
