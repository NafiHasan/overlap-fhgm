"""
Readers for hypergraph files, binary coverage files and class labels.
"""

import logging
import numpy as np
from hypergraph import HyperGraph

logger = logging.getLogger(__name__)


def load_hypergraphs_from_file(filepath):
    """
    Load hypergraphs from a text file.

    File format:
        t # <id>           — start of a new hypergraph
        v <v_id> <v_label> — vertex definition
        e <v1> <v2> ... <edge_label> — hyperedge definition

    Args:
        filepath (str): Path to the hypergraph data file.

    Returns:
        list[HyperGraph]: List of parsed hypergraphs.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    hypergraphs, graph_ids = [], set()
    hypergraph = None
    for number, line in enumerate(lines, 1):
        tokens = line.split()
        if not tokens:
            continue
        try:
            if tokens[0] == "t" and len(tokens) == 3 and tokens[1] == "#":
                graph_id = int(tokens[2])
                if graph_id in graph_ids:
                    raise ValueError("duplicate graph ID")
                graph_ids.add(graph_id)
                if hypergraph is not None:
                    hypergraphs.append(hypergraph)
                hypergraph = HyperGraph()
            elif tokens[0] == "v" and len(tokens) == 3 and hypergraph is not None:
                vertex, label = map(int, tokens[1:])
                if vertex in hypergraph.vertex_label:
                    raise ValueError("duplicate vertex ID")
                hypergraph.add_vertex(vertex, label)
            elif tokens[0] == "e" and len(tokens) >= 3 and hypergraph is not None:
                vertices = list(map(int, tokens[1:-1]))
                if len(set(vertices)) != len(vertices):
                    raise ValueError("duplicate vertex in hyperedge")
                if any(vertex not in hypergraph.vertex_label for vertex in vertices):
                    raise ValueError("hyperedge references an undefined vertex")
                hypergraph.add_hyper_edge(vertices, int(tokens[-1]))
            else:
                raise ValueError("invalid graph record")
        except ValueError as exc:
            raise ValueError(f"{filepath}:{number}: {exc}") from exc

    if hypergraph is not None:
        hypergraphs.append(hypergraph)

    return hypergraphs


def load_coverage_file(filepath):
    """
    Load a binary coverage file (``cover_freq.txt``).

    Returns:
        dict: Mapping from pattern ID to ``{hypergraph_id: 1}`` for the
        hypergraphs that contain the pattern.
    """
    with open(filepath, "r") as f:
        freq = f.readlines()

    codes_count = 0
    codes = dict()

    for i in range(len(freq)):
        if freq[i][0] == "#" or freq[i][0] == '{':
            # Skip the subhypergraph descriptions
            continue

        if freq[i] == "\n":
            codes_count += 1  # New subhypergraph after line break
        else:
            parts = freq[i].replace("\n", "").split(" ")
            hypergraph_id, iso_count = int(parts[0]), int(parts[1])
            if codes_count not in codes:
                codes[codes_count] = dict()
            codes[codes_count][hypergraph_id] = iso_count

    return codes


def load_labels(filepath):
    """
    Load hypergraph class labels from file.

    Args:
        filepath (str): Path to the labels file.

    Returns:
        numpy.ndarray: Array of integer labels.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        labels.append(int(line.split(" ")[0]))
    return np.array(labels)
