"""Hypergraph Weisfeiler-Lehman kernels of Feng et al. (TPAMI 2024), as a baseline.

Port of ``HypergraphSubtreeKernel`` and ``HypergraphHyedgeKernel`` from the
authors' code (github.com/iMoonLab/HIC, commit 179e420,
``models/hypergraph_wl_*.py``) with its defaults: four iterations and cosine
normalization. Each iteration first recolours every hyperedge from its own
colour and the sorted colours of its vertices, then every vertex from its own
colour and the sorted colours of its hyperedges.

- ``subtree`` features count vertex colours of iterations 0 to ``n_iter``.
- ``hyperedge`` features count, per hyperedge, the sorted colours of its
  vertices; as in the authors' code, the initial vertex colours are counted
  twice and the colours of the last iteration are not counted.

Colours are learned from the training graphs only. A held-out colour unseen
in training gets a temporary id that never enters the vocabulary, so it adds
nothing to the cross-kernel. ``held_out_norm`` decides whether it counts in
the held-out graph's own norm:

- ``"seen"`` (default): no, as in the authors' ``transform``, which replaces
  unseen colours by -1 and skips them;
- ``"all"``: yes, so the held-out row equals the exact cosine-normalized WL
  kernel (the rows of a kernel computed on training and held-out graphs
  together).

The module uses only numpy and scipy, so the same file can be checked against
the authors' code in their pinned environment.
"""

from collections import Counter

import numpy as np
from scipy import sparse

KERNELS = ("subtree", "hyperedge")
HELD_OUT_NORMS = ("seen", "all")
N_ITER = 4  # authors' default


class WLGraph:
    """Hypergraph on vertices 0..num_v-1 with initial vertex and edge labels."""

    def __init__(self, num_v, edges, v_lbl, e_lbl):
        if len(v_lbl) != num_v or len(e_lbl) != len(edges):
            raise ValueError("One label per vertex and per edge is required")
        self.num_v = num_v
        self.edges = [tuple(edge) for edge in edges]
        self.v_lbl = list(v_lbl)
        self.e_lbl = list(e_lbl)
        self.v_edges = [[] for _ in range(num_v)]
        for index, edge in enumerate(self.edges):
            for vertex in edge:
                self.v_edges[vertex].append(index)


def feng_input(graph):
    """The authors' input: duplicate hyperedges merged (as ``dhg.Hypergraph``
    does), vertex label = degree after merging, edge label = cardinality."""
    vertices = sorted(graph.vertex_label)
    index = {vertex: i for i, vertex in enumerate(vertices)}
    edges, seen = [], set()
    for edge in graph.hyper_edges:
        key = tuple(sorted(index[vertex] for vertex in edge.vertices))
        if key not in seen:
            seen.add(key)
            edges.append(key)
    degree = [0] * len(vertices)
    for edge in edges:
        for vertex in edge:
            degree[vertex] += 1
    return WLGraph(len(vertices), edges, degree, [len(edge) for edge in edges])


def stored_label_input(graph):
    """The graph exactly as the pipeline sees it: every hyperedge kept, stored
    vertex and edge labels (log2-binned when the caller has binned them)."""
    vertices = sorted(graph.vertex_label)
    index = {vertex: i for i, vertex in enumerate(vertices)}
    edges = [tuple(sorted(index[vertex] for vertex in edge.vertices))
             for edge in graph.hyper_edges]
    return WLGraph(len(vertices), edges,
                   [graph.vertex_label[vertex] for vertex in vertices],
                   [edge.label for edge in graph.hyper_edges])


class _Palette:
    """Colour ids; frozen palettes give unseen keys negative temporary ids."""

    def __init__(self):
        self.ids = {}
        self.frozen = False
        self._new = {}

    def __call__(self, key):
        if key in self.ids:
            return self.ids[key]
        if not self.frozen:
            self.ids[key] = len(self.ids)
            return self.ids[key]
        return self._new.setdefault(key, -1 - len(self._new))

    def held_out(self):
        self.frozen = True
        self._new = {}


class HypergraphWLKernel:
    def __init__(self, kind="subtree", n_iter=N_ITER, held_out_norm="seen"):
        if kind not in KERNELS:
            raise ValueError(f"kind must be one of {KERNELS}")
        if held_out_norm not in HELD_OUT_NORMS:
            raise ValueError(f"held_out_norm must be one of {HELD_OUT_NORMS}")
        self.kind = kind
        self.n_iter = n_iter
        self.held_out_norm = held_out_norm

    def _counts(self, graphs):
        v_pal, e_pal, f_pal = self._palettes
        counts = []
        for graph in graphs:
            count = Counter()
            v = [v_pal(("v", label)) for label in graph.v_lbl]
            e = [e_pal(("e", label)) for label in graph.e_lbl]

            def count_vertices():
                count.update(f_pal(color) for color in v)

            def count_edges():
                count.update(f_pal(tuple(sorted(v[i] for i in edge)))
                             for edge in graph.edges)

            if self.kind == "subtree":
                count_vertices()
            else:
                count_edges()
            for _ in range(self.n_iter):
                e = [e_pal((e[j], tuple(sorted(v[i] for i in edge))))
                     for j, edge in enumerate(graph.edges)]
                if self.kind == "hyperedge":
                    count_edges()
                v = [v_pal((v[i], tuple(sorted(e[j] for j in graph.v_edges[i]))))
                     for i in range(graph.num_v)]
                if self.kind == "subtree":
                    count_vertices()
            counts.append(count)
        return counts

    def _matrix(self, counts):
        """Vocabulary columns, plus each row's squared norm (see ``held_out_norm``)."""
        rows, cols, data = [], [], []
        norms = np.zeros(len(counts))
        for row, count in enumerate(counts):
            for color, value in count.items():
                if color >= 0 or self.held_out_norm == "all":
                    norms[row] += value * value
                if color >= 0:
                    rows.append(row)
                    cols.append(color)
                    data.append(value)
        width = len(self._palettes[2].ids)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(counts), width),
                                   dtype=np.float64)
        return matrix, norms

    @staticmethod
    def _normalize(gram, row_norms, col_norms):
        with np.errstate(divide="ignore", invalid="ignore"):
            gram = gram / np.sqrt(np.outer(row_norms, col_norms))
        gram[~np.isfinite(gram)] = 0.0
        return gram

    def fit_transform(self, graphs):
        """Learn the colours of the training graphs; return their kernel."""
        self._palettes = (_Palette(), _Palette(), _Palette())
        self.train_matrix, self.train_norms = self._matrix(self._counts(graphs))
        for palette in self._palettes:
            palette.held_out()
        gram = (self.train_matrix @ self.train_matrix.T).toarray()
        return self._normalize(gram, self.train_norms, self.train_norms)

    def transform(self, graphs):
        """Kernel of held-out graphs against the training graphs."""
        for palette in self._palettes:
            palette.held_out()
        matrix, norms = self._matrix(self._counts(graphs))
        gram = (matrix @ self.train_matrix.T).toarray()
        return self._normalize(gram, norms, self.train_norms)

    @property
    def vocabulary_size(self):
        return len(self._palettes[2].ids)
