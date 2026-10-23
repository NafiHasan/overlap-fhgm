"""Logarithmic binning of the degree and cardinality labels before mining.

A vertex label is its degree and a hyperedge label is its cardinality.
``label_binning="log2"`` replaces both by logarithmic bins (1, 2, 3-4, 5-8,
...), so hypergraphs that differ only in exact large degrees share labels and
more patterns reach the support threshold.

Binning is applied in memory after ``verify_preprocessing`` has checked the
stored exact labels; the input files are unchanged. It enters the mining cache
key through :func:`effective_label_mode`, so binned and exact vocabularies
never share a cache entry. The default ``"none"`` returns the input unchanged.
"""

import math

from hypergraph.h_graph import HyperGraph

LABEL_BINNINGS = ("none", "log2")


def log2_bin(value):
    """0 -> 0, 1 -> 1, 2 -> 2, 3-4 -> 3, 5-8 -> 4, ..."""
    if type(value) is not int or value < 0:
        raise ValueError("Degrees and cardinalities must be non-negative integers")
    return 0 if value == 0 else math.ceil(math.log2(value)) + 1


def resolve_label_binning(config):
    binning = config.get("label_binning") or "none"
    if binning not in LABEL_BINNINGS:
        raise ValueError(f"label_binning must be one of {LABEL_BINNINGS}")
    return binning


def effective_label_mode(label_mode, config):
    """Label mode as seen by the miner; binned modes get their own cache keys."""
    binning = resolve_label_binning(config)
    return label_mode if binning == "none" else f"{label_mode}+{binning}"


def bin_hypergraph_labels(hypergraphs, config):
    """Return relabelled copies (same vertex IDs and edge order), or the input."""
    binning = resolve_label_binning(config)
    if binning == "none":
        return hypergraphs
    binned = []
    for graph in hypergraphs:
        copy = HyperGraph()
        for vertex, label in graph.vertex_label.items():
            copy.add_vertex(vertex, log2_bin(label))
        for edge in graph.hyper_edges:
            if edge.label != len(edge.vertices):
                raise ValueError("Edge label is not the edge cardinality; cannot bin")
            copy.add_hyper_edge(list(edge.vertices), log2_bin(edge.label))
        binned.append(copy)
    return binned
