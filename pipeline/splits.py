"""Validation for literal DHG-Bench split manifests."""

from dataclasses import dataclass
import json
import numpy as np
from .artifacts import canonical_json_hash, sha256_file


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    repeat: int
    model_seed: int
    split_seed: int
    manifest_hash: str


def load_and_validate_split_manifest(path, *, dataset, label_mode, hypergraph_file,
                                     labels_file, num_graphs, expected_repeat, expected_domain=None):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {"schema_version": 1, "protocol": "dhgbench_graph_classification",
                "dataset": dataset, "label_mode": label_mode, "repeat": expected_repeat,
                "model_seed": expected_repeat, "split_seed": 42 + expected_repeat}
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Invalid split manifest {path}: {key} must be {expected!r}")
    if expected_domain is not None and manifest.get("domain") != expected_domain:
        raise ValueError(f"Invalid split manifest {path}: domain must be {expected_domain!r}")
    if manifest.get("ratios") != {"train": 0.8, "validation": 0.1}:
        raise ValueError("Split manifest ratios must be DHG-Bench 80/10/10")
    generator = manifest.get("generator", {})
    if generator.get("torch") != "2.2.2" or generator.get("device") != "cpu":
        raise ValueError("Split manifest must be generated with torch==2.2.2 on CPU")
    source = manifest.get("source", {})
    if source.get("hypergraph_sha256") != sha256_file(hypergraph_file):
        raise ValueError("Split manifest hypergraph source hash does not match current input")
    if source.get("labels_sha256") != sha256_file(labels_file):
        raise ValueError("Split manifest labels source hash does not match current input")
    indices = manifest.get("indices", {})
    names = ("train", "validation", "test")
    try:
        arrays = {name: np.asarray(indices[name], dtype=np.int64) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Split manifest indices must be integer arrays") from exc
    # Reject JSON floats/bools explicitly rather than allowing numpy coercion.
    if any(any(type(value) is not int for value in indices[name]) for name in names):
        raise ValueError("Split manifest indices must contain JSON integers")
    joined = np.concatenate([arrays[name] for name in names])
    if len(joined) != num_graphs or np.any(joined < 0) or np.any(joined >= num_graphs) or len(np.unique(joined)) != num_graphs:
        raise ValueError("Split manifest indices must be disjoint, in bounds, and cover all graphs")
    train_n, val_n = int(.8 * num_graphs), int(.1 * num_graphs)
    expected_sizes = {"train": train_n, "validation": val_n, "test": num_graphs - train_n - val_n}
    if {name: len(arrays[name]) for name in names} != expected_sizes or manifest.get("sizes", {}) != {"total": num_graphs, **expected_sizes}:
        raise ValueError("Split manifest does not use DHG-Bench floor sizes")
    canonical = {name: indices[name] for name in names}
    if manifest.get("indices_sha256") != canonical_json_hash(canonical):
        raise ValueError("Split manifest index hash is invalid")
    return SplitIndices(arrays["train"], arrays["validation"], arrays["test"], expected_repeat,
                        expected_repeat, 42 + expected_repeat, canonical_json_hash(manifest))
