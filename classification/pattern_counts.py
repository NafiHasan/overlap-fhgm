"""Occurrence counts of mined patterns.

A pattern's count in a graph is its number of distinct images: distinct sets
of (vertex, hyperedge) incidences used by exact matches under the miner's
semantics (labels equal, a pattern edge contained in its target edge, and,
for overlap-labelled vocabularies, equal overlap labels on pattern edge pairs
that share a vertex). Each image is counted once, whatever the pattern's
symmetry.

Every connected pattern skeleton with at most three incidences is tallied
per graph in one pass:

* ``i``    -- one vertex in one edge;
* ``vv``   -- two vertices in one edge;
* ``ee``   -- two edges at a common vertex;
* ``vvv``  -- three vertices in one edge;
* ``path`` -- edges e1 = {x, y} and e2 = {x};
* ``eee``  -- three edges at a common vertex.

``graph_tally(..., max_depth=4)`` also tallies the seven four-incidence
skeletons:

* ``vvvv``  -- four vertices in one edge;
* ``eeee``  -- four edges at a common vertex;
* ``vpath`` -- e1 = {x, y} and e2 = {x, z};
* ``epath`` -- e0 = {x, y}, e1 = {x} and e2 = {y};
* ``vfork`` -- e1 = {x, y}, e2 = {x} and e3 = {x};
* ``efork`` -- e0 = {x, y, z} and e1 = {x};
* ``ee2``   -- e1 = {u, v} and e2 = {u, v}.

Without overlap labels the four-incidence counts come from closed-form label
counts; no image is listed. With overlap labels (``overlap_binning``) they carry the binned overlap
labels of the edge pairs that share a pattern vertex: ``eeee`` and ``vfork``
are tallied by enumerating the edge sets at each vertex, the other shapes by
grouped label counts; ``vvvv`` has one edge and no overlap label.

Tallies of at most three incidences use exact overlap sizes and are converted
to a vocabulary's overlap binning on demand; four-incidence overlap keys are
binned when tallied and name their binning. They need no class labels and no
fitting, so they are computed once for all graphs. A pattern's relative
frequency divides its count by the graph's total count of its skeleton (all
labels removed). This module is outside ``mining/`` and ``hypergraph/`` so a
change here does not invalidate mining caches; the pipeline checks that a
count is positive exactly when frozen matching reports the pattern present.
"""

from collections import Counter
from itertools import combinations_with_replacement, permutations
from math import comb

import numpy as np

from hypergraph.h_graph import overlap_bin

SKELETON_DEPTH = {"i": 1, "vv": 2, "ee": 2, "vvv": 3, "path": 3, "eee": 3,
                  "vvvv": 4, "eeee": 4, "vpath": 4, "epath": 4, "vfork": 4,
                  "efork": 4, "ee2": 4}
DEPTH4_SKELETONS = frozenset(name for name, depth in SKELETON_DEPTH.items() if depth == 4)
# The base tally covers three incidences; four-incidence skeletons are added
# for deeper vocabularies, with or without overlap labels.
MAX_COUNT_DEPTH = 3
DEEP_COUNT_DEPTH = 4
COUNT_ENCODINGS = ("relative", "log", "presence")
# Radix for packing six small labels into one int64 (512**6 = 2**54).
_RADIX = 512
_PERMUTATIONS = tuple(permutations(range(3)))
# Four-incidence overlap keys: radix for packing up to eleven labels into one
# int64 (32**11 = 2**55), edge pairs of four edges, and rows per block.
_RADIX4 = 32
_PERMUTATIONS4 = tuple(permutations(range(4)))
_PAIRS4 = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_BLOCK_ROWS = 1 << 21


def resolve_count_options(config):
    """Count-encoding options: ``(encoding, min_depth, min_train_graphs)``."""
    encoding = config.get("mined_count_encoding") or "relative"
    min_depth = config.get("mined_min_depth") or 1
    min_train_graphs = config.get("mined_min_train_graphs") or 2
    limit = DEEP_COUNT_DEPTH
    if encoding not in COUNT_ENCODINGS:
        raise ValueError(f"mined_count_encoding must be one of {COUNT_ENCODINGS}")
    if type(min_depth) is not int or not 1 <= min_depth <= limit:
        raise ValueError(f"mined_min_depth must be an integer in 1..{limit}")
    if type(min_train_graphs) is not int or min_train_graphs < 1:
        raise ValueError("mined_min_train_graphs must be a positive integer")
    max_depth = config.get("max_depth")
    if max_depth is None or max_depth > limit:
        raise ValueError(f"Pattern counts need max_depth <= {limit}")
    return encoding, min_depth, min_train_graphs


def tally_depth(config):
    """Skeleton depth to tally for a counts run: 4 only for deeper vocabularies."""
    return DEEP_COUNT_DEPTH if (config.get("max_depth") or 0) > MAX_COUNT_DEPTH \
        else MAX_COUNT_DEPTH


def tally_overlap_binning(config):
    """Overlap binning of the four-incidence tallies of a counts run, or None."""
    binning = config.get("overlap_labels") or "none"
    return None if tally_depth(config) == MAX_COUNT_DEPTH or binning == "none" else binning


def _pair(a, b):
    return (a, b) if a < b else (b, a)


def canonical_eee(vertex_label, edge_labels, overlaps):
    """Canonical key of three edges at a vertex.

    ``overlaps`` holds the labels of edge pairs (0, 1), (0, 2), (1, 2), or is
    None for a vocabulary without overlap labels.
    """
    if overlaps is None:
        return ("eee", vertex_label, *sorted(edge_labels), None, None, None)
    pair = {(0, 1): overlaps[0], (0, 2): overlaps[1], (1, 2): overlaps[2]}
    best = None
    for p, q, r in _PERMUTATIONS:
        key = (edge_labels[p], edge_labels[q], edge_labels[r],
               pair[_pair(p, q)], pair[_pair(p, r)], pair[_pair(q, r)])
        if best is None or key < best:
            best = key
    return ("eee", vertex_label, *best)


def _combinations3(d):
    """All index triples a < b < c below d, as an (n, 3) array."""
    parts = []
    for a in range(d - 2):
        rest = d - a - 1
        b, c = np.triu_indices(rest, k=1)
        parts.append(np.column_stack((np.full(len(b), a), b + a + 1, c + a + 1)))
    return np.concatenate(parts) if parts else np.zeros((0, 3), dtype=np.int64)


def graph_tally(graph, max_depth=MAX_COUNT_DEPTH, overlap_binning=None):
    """Exact-overlap tally and skeleton totals of one hypergraph.

    Returns ``(tally, totals)``: ``tally`` maps a skeleton key with exact
    overlap sizes to its number of images, ``totals`` maps each skeleton to
    its total number of images in the graph. With ``max_depth`` 4 the
    four-incidence skeletons are added: without overlap labels when
    ``overlap_binning`` is None, else with overlap labels in that binning.
    """
    if max_depth not in (MAX_COUNT_DEPTH, DEEP_COUNT_DEPTH):
        raise ValueError(f"max_depth must be {MAX_COUNT_DEPTH} or {DEEP_COUNT_DEPTH}")
    tally, totals = _tally_depth3(graph)
    if max_depth == DEEP_COUNT_DEPTH and overlap_binning is None:
        _add_depth4(graph, tally, totals)
    elif max_depth == DEEP_COUNT_DEPTH:
        _add_depth4_overlap(graph, tally, totals, overlap_binning)
    return tally, totals


def _tally_depth3(graph):
    """Tallies and totals of the skeletons of at most three incidences."""
    vertices = list(graph.vertex_label)
    position = {vertex: index for index, vertex in enumerate(vertices)}
    vertex_labels = np.array([graph.vertex_label[v] for v in vertices], dtype=np.int64)
    members = [np.array([position[v] for v in edge.vertices], dtype=np.int64)
               for edge in graph.hyper_edges]
    edge_labels = np.array([edge.label for edge in graph.hyper_edges], dtype=np.int64)
    for value in (*vertex_labels.tolist(), *edge_labels.tolist()):
        if type(value) is not int or not 0 <= value < _RADIX:
            raise ValueError(f"Labels must be integers in [0, {_RADIX})")
    n_edges = len(members)
    tally = Counter()
    totals = Counter({name: 0 for name, depth in SKELETON_DEPTH.items() if depth <= 3})

    # One edge: i, vv, vvv from the label counts of its vertices.
    for edge_index, vertex_ids in enumerate(members):
        size = len(vertex_ids)
        if len(set(vertex_ids.tolist())) != size:
            raise ValueError("A hyperedge lists a vertex twice")
        label = int(edge_labels[edge_index])
        counts = Counter(vertex_labels[vertex_ids].tolist())
        totals["i"] += size
        totals["vv"] += comb(size, 2)
        totals["vvv"] += comb(size, 3)
        for vertex_label, n in counts.items():
            tally[("i", vertex_label, label)] += n
        keys = sorted(counts)
        for k in (2, 3):
            for combo in combinations_with_replacement(keys, k):
                n = 1
                for vertex_label, multiplicity in Counter(combo).items():
                    n *= comb(counts[vertex_label], multiplicity)
                if n:
                    tally[("vv" if k == 2 else "vvv", label, *combo)] += n

    if n_edges < 2:
        return tally, dict(totals)

    incidence = np.zeros((len(vertices), n_edges), dtype=np.int64)
    for edge_index, vertex_ids in enumerate(members):
        incidence[vertex_ids, edge_index] = 1
    overlap = incidence.T @ incidence                       # |e_a ∩ e_b|
    label_values = np.unique(vertex_labels)
    label_column = {int(value): column for column, value in enumerate(label_values)}
    # Vertex-label counts per edge, for the y vertex of a path.
    label_counts = np.zeros((n_edges, len(label_values)), dtype=np.int64)
    for edge_index, vertex_ids in enumerate(members):
        np.add.at(label_counts[edge_index],
                  np.searchsorted(label_values, vertex_labels[vertex_ids]), 1)
    sizes = np.array([len(m) for m in members], dtype=np.int64)

    pair_x, pair_a, pair_b = [], [], []
    for vertex in range(len(vertices)):
        incident = np.flatnonzero(incidence[vertex])
        d = len(incident)
        lx = int(vertex_labels[vertex])
        totals["ee"] += comb(d, 2)
        totals["eee"] += comb(d, 3)
        totals["path"] += int((sizes[incident] - 1).sum()) * (d - 1)
        if d < 2:
            continue
        a, b = np.meshgrid(incident, incident, indexing="ij")
        keep = a != b
        pair_x.append(np.full(int(keep.sum()), vertex))
        pair_a.append(a[keep])
        pair_b.append(b[keep])
        if d < 3:
            continue
        triples = incident[_combinations3(d)]
        l1, l2, l3 = (edge_labels[triples[:, k]] for k in range(3))
        o12 = overlap[triples[:, 0], triples[:, 1]]
        o13 = overlap[triples[:, 0], triples[:, 2]]
        o23 = overlap[triples[:, 1], triples[:, 2]]
        labels = (l1, l2, l3)
        pairs = {(0, 1): o12, (0, 2): o13, (1, 2): o23}
        best = None
        for p, q, r in _PERMUTATIONS:
            code = labels[p]
            for field in (labels[q], labels[r], pairs[_pair(p, q)],
                          pairs[_pair(p, r)], pairs[_pair(q, r)]):
                code = code * _RADIX + field
            best = code if best is None else np.minimum(best, code)
        codes, counts = np.unique(best, return_counts=True)
        for code, n in zip(codes.tolist(), counts.tolist()):
            fields = []
            for _ in range(6):
                code, field = divmod(code, _RADIX)
                fields.append(field)
            tally[("eee", lx, *reversed(fields))] += n

    if not pair_x:
        return tally, dict(totals)
    x = np.concatenate(pair_x)
    a = np.concatenate(pair_a)
    b = np.concatenate(pair_b)
    lx, la, lb, o = vertex_labels[x], edge_labels[a], edge_labels[b], overlap[a, b]

    # ee: unordered edge pairs at a vertex.
    first = a < b
    low = np.minimum(la[first], lb[first])
    high = np.maximum(la[first], lb[first])
    code = ((lx[first] * _RADIX + low) * _RADIX + high) * _RADIX + o[first]
    codes, counts = np.unique(code, return_counts=True)
    for code, n in zip(codes.tolist(), counts.tolist()):
        code, ov = divmod(code, _RADIX)
        code, high = divmod(code, _RADIX)
        vertex_label, low = divmod(code, _RADIX)
        tally[("ee", vertex_label, low, high, ov)] += n

    # path: ordered pairs (e1 = a with a second vertex y, e2 = b at x).
    code = ((lx * _RADIX + la) * _RADIX + lb) * _RADIX + o
    groups, inverse, group_sizes = np.unique(code, return_inverse=True, return_counts=True)
    per_edge = np.bincount(inverse * n_edges + a, minlength=len(groups) * n_edges)
    y_counts = per_edge.reshape(len(groups), n_edges) @ label_counts   # groups x labels
    for group, (code, n_pairs) in enumerate(zip(groups.tolist(), group_sizes.tolist())):
        code, ov = divmod(code, _RADIX)
        code, l2 = divmod(code, _RADIX)
        vertex_label, l1 = divmod(code, _RADIX)
        row = y_counts[group].copy()
        row[label_column[vertex_label]] -= n_pairs           # y differs from x
        for column in np.flatnonzero(row).tolist():
            tally[("path", vertex_label, int(label_values[column]), l1, l2, ov)] += int(row[column])
    return tally, dict(totals)


def _binomial(n, k):
    """Elementwise C(n, k) of an integer array (k <= 4)."""
    result = np.ones_like(n)
    for i in range(k):
        result = result * (n - i)
    return result // {0: 1, 1: 1, 2: 2, 3: 6, 4: 24}[k]


def _one_hot(labels):
    """Sorted label values, each item's column, and the items x values one-hot matrix."""
    values, column = np.unique(np.array(labels, dtype=np.int64), return_inverse=True)
    hot = np.zeros((len(column), len(values)), dtype=np.int64)
    hot[np.arange(len(column)), column] = 1
    return [int(v) for v in values], column, hot


def _add_depth4(graph, tally, totals):
    """Add the four-incidence skeletons (plain keys) to ``tally`` and ``totals``.

    The shapes come in dual pairs, the vertex and edge roles swapped: a
    centre item with its arms (``links`` is centre x arm) and the arms'
    other centres as leaves. Vertex centres give eeee, vpath, vfork; edge
    centres give vvvv, epath, efork; ee2 is its own dual.
    """
    for name in DEPTH4_SKELETONS:
        totals[name] = 0
    vertices = list(graph.vertex_label)
    if not vertices or not graph.hyper_edges:
        return
    position = {vertex: index for index, vertex in enumerate(vertices)}
    incidence = np.zeros((len(vertices), len(graph.hyper_edges)), dtype=np.int64)
    for edge_index, edge in enumerate(graph.hyper_edges):
        incidence[[position[v] for v in edge.vertices], edge_index] = 1
    vertex_side = _one_hot([graph.vertex_label[v] for v in vertices])
    edge_side = _one_hot([edge.label for edge in graph.hyper_edges])
    for names, links, centre, arm in ((("eeee", "vpath", "vfork"), incidence, vertex_side, edge_side),
                                      (("vvvv", "epath", "efork"), incidence.T, edge_side, vertex_side)):
        star, path, fork = names
        shared = _shared_by_label(links, centre[2])
        _star4(star, links, centre, arm, tally, totals)
        _path4(path, links, centre, arm, shared, tally, totals)
        _fork4(fork, links, centre, arm, tally, totals)
        if star == "eeee":
            _ee2(incidence, vertex_side, edge_side, shared, tally, totals)


def _shared_by_label(links, centre_hot):
    """centre labels x arm x arm: centres of each label on both arms (diagonal 0)."""
    shared = np.stack([(links * centre_hot[:, [a]]).T @ links
                       for a in range(centre_hot.shape[1])])
    index = np.arange(links.shape[1])
    shared[:, index, index] = 0
    return shared


def _star4(name, links, centre, arm, tally, totals):
    """One centre with four arms: vvvv (edge centre) or eeee (vertex centre)."""
    centre_values, _column, centre_hot = centre
    arm_values, _arm_column, arm_hot = arm
    arm_count = links @ arm_hot                                 # centre x arm label
    totals[name] += int(_binomial(arm_count.sum(axis=1), 4).sum())
    present = np.flatnonzero(arm_count.sum(axis=0)).tolist()
    for combo in combinations_with_replacement(present, 4):
        product = np.ones(len(arm_count), dtype=np.int64)
        for label, multiplicity in Counter(combo).items():
            product = product * _binomial(arm_count[:, label], multiplicity)
        by_label = centre_hot.T @ product
        for label in np.flatnonzero(by_label).tolist():
            key = (name, centre_values[label], *(arm_values[a] for a in combo))
            tally[key] += int(by_label[label])


def _path4(name, links, centre, arm, shared, tally, totals):
    """A centre and two arms, each with one leaf: vpath or epath.

    Key: (centre label, (arm label, leaf label) of each arm, sorted). Over
    ordered arm pairs, each arm's leaves exclude the centre; leaf pairs that
    coincide (a common leaf of both arms) are subtracted. Equal roles halve.
    """
    centre_values, centre_column, centre_hot = centre
    arm_values, arm_column, arm_hot = arm
    n_centres, n_arms = links.shape
    n_leaf, n_arm_labels = len(centre_values), len(arm_values)
    sizes = links.sum(axis=0)
    overlap = links.T @ links
    upper = np.triu_indices(n_arms, k=1)
    first, second = links @ (sizes - 1), links @ ((sizes - 1) ** 2)
    totals[name] += int(((first ** 2 - second) // 2).sum() - (overlap[upper] ** 2).sum()
                        + _binomial(links.sum(axis=1), 2).sum())

    leaf_count = links.T @ centre_hot                           # arm x leaf label
    slots = np.zeros((n_arms, n_arm_labels, n_leaf), dtype=np.int64)
    slots[np.arange(n_arms), arm_column] = leaf_count
    slots = slots.reshape(n_arms, -1)                           # arm x (arm label, leaf label)
    # Per incidence (centre x, arm f): f's slot vector less x itself as a leaf.
    rows, cols = np.nonzero(links)
    per_incidence = slots[cols].reshape(-1, n_arm_labels, n_leaf)
    per_incidence[np.arange(len(rows)), arm_column[cols], centre_column[rows]] -= 1
    per_incidence = per_incidence.reshape(len(rows), -1)
    summed = np.zeros((n_centres, slots.shape[1]), dtype=np.int64)
    np.add.at(summed, rows, per_incidence)
    for a in range(n_leaf):
        mine = centre_column == a
        if not mine.any():
            continue
        ordered = summed[mine].T @ summed[mine]
        diagonal = per_incidence[mine[rows]]
        ordered -= diagonal.T @ diagonal
        ordered = ordered.reshape(n_arm_labels, n_leaf, n_arm_labels, n_leaf)
        for b in range(n_leaf):
            common = shared[a] * (shared[b] - (a == b))
            ordered[:, b, :, b] -= arm_hot.T @ common @ arm_hot
        ordered = ordered.reshape(n_arm_labels * n_leaf, -1)
        for s, t in zip(*np.nonzero(np.triu(ordered))):
            n = int(ordered[s, t])
            if s == t:
                if n % 2:
                    raise RuntimeError(f"{name}: odd ordered count for a symmetric key")
                n //= 2
            (l1, b1), (l2, b2) = divmod(int(s), n_leaf), divmod(int(t), n_leaf)
            key = (name, centre_values[a], arm_values[l1], centre_values[b1],
                   arm_values[l2], centre_values[b2])
            tally[key] += n


def _fork4(name, links, centre, arm, tally, totals):
    """A centre with three arms, one with a leaf: vfork or efork.

    Key: (centre label, special arm label, leaf label, the other two arm
    labels sorted).
    """
    centre_values, _column, centre_hot = centre
    arm_values, _arm_column, arm_hot = arm
    sizes = links.sum(axis=0)
    arm_count = links @ arm_hot                                 # centre x arm label
    totals[name] += int(((links @ (sizes - 1)) * _binomial(arm_count.sum(axis=1) - 1, 2)).sum())
    leaf_count = links.T @ centre_hot                           # arm x leaf label
    present = np.flatnonzero(arm_count.sum(axis=0)).tolist()
    for special in present:
        # Leaves on the special arms at each centre, the centre itself excluded.
        leaves = links @ (arm_hot[:, [special]] * leaf_count) - centre_hot * arm_count[:, [special]]
        rest = arm_count.copy()
        rest[:, special] -= 1
        for i, p in enumerate(present):
            for q in present[i:]:
                pairs = rest[:, p] * rest[:, q] if p != q else _binomial(rest[:, p], 2)
                block = centre_hot.T @ (leaves * pairs[:, None])   # centre label x leaf label
                for a, b in zip(*np.nonzero(block)):
                    key = (name, centre_values[a], arm_values[special], centre_values[b],
                           arm_values[p], arm_values[q])
                    tally[key] += int(block[a, b])


def _ee2(incidence, vertex_side, edge_side, shared, tally, totals):
    """Two edges sharing two vertices. Key: (edge labels sorted, vertex labels sorted)."""
    vertex_values = vertex_side[0]
    edge_values, _column, edge_hot = edge_side
    overlap = incidence.T @ incidence
    upper = np.triu_indices(overlap.shape[0], k=1)
    totals["ee2"] += int(_binomial(overlap[upper], 2).sum())
    for a in range(len(vertex_values)):
        for b in range(a, len(vertex_values)):
            pairs = shared[a] * shared[b] if a != b else _binomial(shared[a], 2)
            ordered = edge_hot.T @ pairs @ edge_hot               # ordered edge pairs by label
            for l1, l2 in zip(*np.nonzero(np.triu(ordered))):
                n = int(ordered[l1, l2])
                if l1 == l2:
                    if n % 2:
                        raise RuntimeError("ee2: odd ordered count for a symmetric key")
                    n //= 2
                key = ("ee2", edge_values[l1], edge_values[l2], vertex_values[a], vertex_values[b])
                tally[key] += n


def canonical_eeee(vertex_label, edge_labels, overlaps, binning):
    """Canonical key of four edges at a vertex with overlap labels.

    ``overlaps`` maps each index pair of ``_PAIRS4`` to its overlap label.
    The key is the smallest field tuple over the 24 orders of the edges.
    """
    best = min(tuple(edge_labels[p] for p in order)
               + tuple(overlaps[_pair(order[i], order[j])] for i, j in _PAIRS4)
               for order in _PERMUTATIONS4)
    return ("eeee", binning, vertex_label, *best)


def _unpack(code, n_fields):
    """Fields of a radix-``_RADIX4`` code, most significant first."""
    fields = []
    for _ in range(n_fields):
        code, field = divmod(code, _RADIX4)
        fields.append(field)
    return fields[::-1]


_COLEX = {}


def _colex(n, k):
    """The k-subsets of range(n) in colexicographic order, as a (C(n, k), k) array.

    The subsets of range(m) are the first C(m, k) rows, so one array serves
    every m <= n; the largest one built is kept per k.
    """
    cached = _COLEX.get(k)
    if cached is None or cached[0] < n:
        if k == 1:
            rows = np.arange(n, dtype=np.int64)[:, None]
        else:
            lower = _colex(n - 1, k - 1) if n > 1 else np.zeros((0, k - 1), dtype=np.int64)
            parts = [np.column_stack((lower[:comb(m, k - 1)], np.full(comb(m, k - 1), m)))
                     for m in range(k - 1, n)]
            rows = np.concatenate(parts) if parts else np.zeros((0, k), dtype=np.int64)
        cached = _COLEX[k] = (n, rows)
    return cached[1][:comb(n, k)]


def _subsets_at_centres(neighbours, k):
    """Every k-subset of each centre's neighbours, in blocks of rows.

    Yields ``(centres, subsets)``: the centre of each row and a rows x k
    array of neighbour indices. Centres of equal degree are taken together;
    a high-degree centre is split by the largest member of its subsets.
    """
    by_degree = {}
    for centre, items in enumerate(neighbours):
        if len(items) >= k:
            by_degree.setdefault(len(items), []).append(centre)
    for degree, centres in sorted(by_degree.items()):
        matrix = np.array([neighbours[c] for c in centres], dtype=np.int64)
        centres = np.array(centres, dtype=np.int64)
        per_centre = comb(degree, k)
        if per_centre <= _BLOCK_ROWS:
            local = _colex(degree, k)
            step = max(1, _BLOCK_ROWS // per_centre)
            for start in range(0, len(centres), step):
                block = matrix[start:start + step][:, local].reshape(-1, k)
                yield np.repeat(centres[start:start + step], per_centre), block
            continue
        lower = _colex(degree - 1, k - 1)
        for row, centre in zip(matrix, centres.tolist()):
            largest = k - 1
            while largest < degree:
                parts, size = [], 0
                while largest < degree and (not parts or size + comb(largest, k - 1) <= _BLOCK_ROWS):
                    n = comb(largest, k - 1)
                    parts.append(np.column_stack((lower[:n], np.full(n, largest))))
                    size += n
                    largest += 1
                local = np.concatenate(parts)
                yield np.full(len(local), centre, dtype=np.int64), row[local]


def _add_depth4_overlap(graph, tally, totals, binning):
    """Add the four-incidence skeletons with overlap labels in ``binning``.

    Overlap labels are those of the edge pairs that share a pattern vertex. Totals are label-free, so
    they are the plain shapes' totals; ``vvvv`` has no overlap label and
    keeps its plain key.
    """
    plain = Counter()
    _add_depth4(graph, plain, totals)
    for key, n in plain.items():
        if key[0] == "vvvv":
            tally[key] += n
    vertices = list(graph.vertex_label)
    if not vertices or len(graph.hyper_edges) < 2:
        return
    position = {vertex: index for index, vertex in enumerate(vertices)}
    incidence = np.zeros((len(vertices), len(graph.hyper_edges)), dtype=np.int64)
    for edge_index, edge in enumerate(graph.hyper_edges):
        incidence[[position[v] for v in edge.vertices], edge_index] = 1
    vertex_side = _one_hot([graph.vertex_label[v] for v in vertices])
    edge_side = _one_hot([edge.label for edge in graph.hyper_edges])
    overlap = incidence.T @ incidence
    table = np.array([overlap_bin(size, binning) for size in range(int(overlap.max()) + 1)],
                     dtype=np.int64)
    bins = table[overlap]
    sharing = overlap > 0
    np.fill_diagonal(sharing, False)
    for name, values in (("vertex labels", vertex_side[0]), ("edge labels", edge_side[0]),
                         ("overlap labels", table.tolist())):
        if min(values) < 0 or max(values) >= _RADIX4:
            raise ValueError(f"Four-incidence overlap counts need {name} in [0, {_RADIX4})")
    shared = _shared_by_label(incidence, vertex_side[2])
    leaf_count = incidence.T @ vertex_side[2]                   # edge x vertex label
    neighbours = [np.flatnonzero(row) for row in incidence]
    _eeee_overlap(vertex_side, edge_side, bins, neighbours, binning, tally)
    _vfork_overlap(vertex_side, edge_side, bins, leaf_count, neighbours, binning, tally)
    _vpath_ee2_overlap(vertex_side, edge_side, bins, sharing, overlap, shared, leaf_count,
                       binning, tally)
    _epath_efork_overlap(incidence, vertex_side, edge_side, bins, sharing, overlap, shared,
                         leaf_count, binning, tally)


def _eeee_overlap(vertex_side, edge_side, bins, neighbours, binning, tally):
    """Four edges at a vertex: labels and all six overlap labels."""
    vertex_values, vertex_column, _hot = vertex_side
    edge_values, edge_column, _edge_hot = edge_side
    vertex_labels = np.array(vertex_values, dtype=np.int64)[vertex_column]
    edge_labels = np.array(edge_values, dtype=np.int64)[edge_column]
    codes, counts = [], []
    for centres, quads in _subsets_at_centres(neighbours, 4):
        code = vertex_labels[centres]
        for k in range(4):
            code = code * _RADIX4 + edge_labels[quads[:, k]]
        for i, j in _PAIRS4:
            code = code * _RADIX4 + bins[quads[:, i], quads[:, j]]
        block_codes, block_counts = np.unique(code, return_counts=True)
        codes.append(block_codes)
        counts.append(block_counts)
    if not codes:
        return
    codes, inverse = np.unique(np.concatenate(codes), return_inverse=True)
    counts = np.bincount(inverse, weights=np.concatenate(counts)).astype(np.int64)
    # Canonical form of each distinct code: the smallest over the edge orders.
    fields = np.stack([(codes // _RADIX4 ** (10 - f)) % _RADIX4 for f in range(11)], axis=1)
    pair_field = {pair: 5 + index for index, pair in enumerate(_PAIRS4)}
    best = None
    for order in _PERMUTATIONS4:
        code = fields[:, 0]
        for p in order:
            code = code * _RADIX4 + fields[:, 1 + p]
        for i, j in _PAIRS4:
            code = code * _RADIX4 + fields[:, pair_field[_pair(order[i], order[j])]]
        best = code if best is None else np.minimum(best, code)
    canonical, inverse = np.unique(best, return_inverse=True)
    sums = np.bincount(inverse, weights=counts)
    for code, n in zip(canonical.tolist(), sums.tolist()):
        tally[("eeee", binning, *_unpack(code, 11))] += int(round(n))


def _vfork_overlap(vertex_side, edge_side, bins, leaf_count, neighbours, binning, tally):
    """x in e1, e2, e3 and y in e1: (x, e1, y) with the other arms' labels and overlaps.

    Key: (x label, e1 label, y label, (label, overlap with e1) of e2 and e3
    sorted, overlap of e2 and e3).
    """
    vertex_values, vertex_column, _hot = vertex_side
    edge_values, edge_column, _edge_hot = edge_side
    vertex_labels = np.array(vertex_values, dtype=np.int64)[vertex_column]
    edge_labels = np.array(edge_values, dtype=np.int64)[edge_column]
    groups, leaves, sizes = [], [], []
    for centres, triples in _subsets_at_centres(neighbours, 3):
        x_label = vertex_labels[centres]
        for special, (p, q) in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
            e1, e2, e3 = triples[:, special], triples[:, p], triples[:, q]
            arm2 = edge_labels[e2] * _RADIX4 + bins[e1, e2]
            arm3 = edge_labels[e3] * _RADIX4 + bins[e1, e3]
            code = (x_label * _RADIX4 + edge_labels[e1]) * _RADIX4 ** 2 + np.minimum(arm2, arm3)
            code = (code * _RADIX4 ** 2 + np.maximum(arm2, arm3)) * _RADIX4 + bins[e2, e3]
            block, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
            groups.append(block)
            sizes.append(counts)
            leaves.append(np.stack([np.bincount(inverse, weights=leaf_count[e1, b], minlength=len(block))
                                    for b in range(len(vertex_values))], axis=1))
    if not groups:
        return
    codes, inverse = np.unique(np.concatenate(groups), return_inverse=True)
    sizes = np.bincount(inverse, weights=np.concatenate(sizes))
    leaves = np.concatenate(leaves)
    leaves = np.stack([np.bincount(inverse, weights=leaves[:, b], minlength=len(codes))
                       for b in range(leaves.shape[1])], axis=1)
    column = {value: index for index, value in enumerate(vertex_values)}
    for code, size, row in zip(codes.tolist(), sizes.tolist(), leaves.tolist()):
        x_label, l1, la, oa, lb, ob, o23 = _unpack(code, 7)
        row[column[x_label]] -= size                         # y differs from x
        for b, n in enumerate(row):
            n = int(round(n))
            if n < 0:
                raise RuntimeError("vfork: negative count")
            if n:
                tally[("vfork", binning, x_label, l1, vertex_values[b], la, oa, lb, ob, o23)] += n


def _emit_pairs(ordered, name, decode, tally):
    """Images of a shape with two interchangeable arms, from ordered arm-pair counts."""
    ordered = np.rint(ordered).astype(np.int64)
    if (ordered < 0).any():
        raise RuntimeError(f"{name}: negative count")
    for s, t in zip(*np.nonzero(np.triu(ordered))):
        n = int(ordered[s, t])
        if s == t:
            if n % 2:
                raise RuntimeError(f"{name}: odd ordered count for a symmetric key")
            n //= 2
        tally[decode(int(s), int(t))] += n


def _vpath_ee2_overlap(vertex_side, edge_side, bins, sharing, overlap, shared, leaf_count,
                       binning, tally):
    """Edge pairs sharing a vertex, grouped by their overlap label: vpath and ee2.

    vpath: x in e1 and e2, y in e1, z in e2, as in the plain counter with
    the edge pairs restricted to one overlap label. ee2: e1 and e2 share u, v.
    """
    vertex_values, _column, _hot = vertex_side
    edge_values, edge_column, edge_hot = edge_side
    n_edges, n_leaf, n_arm = len(edge_column), len(vertex_values), len(edge_values)
    edge_hot = edge_hot.astype(np.float64)
    slots = np.zeros((n_edges, n_arm, n_leaf))
    slots[np.arange(n_edges), edge_column] = leaf_count
    for o in np.unique(bins[sharing]).tolist():
        mask = sharing & (bins == o)
        for a in range(n_leaf):
            weight = shared[a] * mask                      # choices of x per edge pair
            if not weight.any():
                continue
            arms = slots.copy()
            arms[np.arange(n_edges), edge_column, a] -= 1     # y, z differ from x
            arms = arms.reshape(n_edges, -1)
            ordered = (arms.T @ weight @ arms).reshape(n_arm, n_leaf, n_arm, n_leaf)
            for b in range(n_leaf):
                common = weight * (shared[b] - (a == b))   # y = z
                ordered[:, b, :, b] -= edge_hot.T @ common @ edge_hot
            _emit_pairs(
                ordered.reshape(n_arm * n_leaf, -1), "vpath",
                lambda s, t, a=a, o=o: ("vpath", binning, vertex_values[a],
                                        edge_values[s // n_leaf], vertex_values[s % n_leaf],
                                        edge_values[t // n_leaf], vertex_values[t % n_leaf], o),
                tally)
        two = mask & (overlap >= 2)
        if not two.any():
            continue
        for a in range(n_leaf):
            for b in range(a, n_leaf):
                pairs = shared[a] * shared[b] if a != b else _binomial(shared[a], 2)
                _emit_pairs(edge_hot.T @ (pairs * two) @ edge_hot, "ee2",
                            lambda s, t, a=a, b=b, o=o: ("ee2", binning, edge_values[s],
                                                         edge_values[t], vertex_values[a],
                                                         vertex_values[b], o),
                            tally)


def _epath_efork_overlap(incidence, vertex_side, edge_side, bins, sharing, overlap, shared,
                         leaf_count, binning, tally):
    """A centre edge e0 and arms at its vertices: epath and efork.

    For each incidence (x, e0), ``arms`` counts the edges e1 != e0 at x by
    (label, overlap label with e0). epath: e0 holds x and y, e1 is at x and
    e2 at y, e1 != e2. efork: e0 holds x, y and z, e1 is at x.
    """
    vertex_values, vertex_column, _hot = vertex_side
    edge_values, edge_column, edge_hot = edge_side
    n_leaf, n_arm = len(vertex_values), len(edge_values)
    overlap_values = np.unique(bins[sharing]).tolist()
    if not overlap_values:
        return
    n_slot = n_arm * len(overlap_values)
    rows_x, rows_e = np.nonzero(incidence)
    incidence_f = incidence.astype(np.float64)
    arms = np.zeros((len(rows_x), n_arm, len(overlap_values)))
    for k, o in enumerate(overlap_values):
        mask = (sharing & (bins == o)).astype(np.float64)
        for label in range(n_arm):
            per_pair = incidence_f @ (mask * edge_hot[:, label][None, :]).T   # x x e0
            arms[:, label, k] = per_pair[rows_x, rows_e]
    arms = arms.reshape(len(rows_x), n_slot)
    x_column = vertex_column[rows_x]
    e0_column = edge_column[rows_e]

    def arm_fields(slot):
        label, k = divmod(slot, len(overlap_values))
        return edge_values[label], overlap_values[k]

    # epath: ordered pairs (x, y) of e0, slots (vertex label, arm label, overlap).
    per_edge = np.zeros((incidence.shape[1], n_leaf, n_slot))
    np.add.at(per_edge, (rows_e, x_column), arms)
    per_edge = per_edge.reshape(incidence.shape[1], -1)
    # e1 = e2: an edge h != e0 holding both x and y, counted for each ordered (x, y).
    first, second = np.nonzero((overlap >= 2) & sharing)
    common = shared[:, first, second].T                   # pairs x vertex label
    pair_slot = edge_column[second] * len(overlap_values) + np.searchsorted(
        overlap_values, bins[first, second])
    for centre_label in range(n_arm):
        mine = edge_column == centre_label
        rows = e0_column == centre_label
        if not rows.any():
            continue
        ordered = per_edge[mine].T @ per_edge[mine]
        ordered = ordered.reshape(n_leaf, n_slot, n_leaf, n_slot)
        for a in range(n_leaf):
            own = arms[rows & (x_column == a)]
            ordered[a, :, a, :] -= own.T @ own                # x = y
        coincide = edge_column[first] == centre_label
        for slot in np.unique(pair_slot[coincide]).tolist():
            s = common[coincide & (pair_slot == slot)]
            ordered[:, slot, :, slot] -= s.T @ s - np.diag(s.sum(axis=0))
        _emit_pairs(
            ordered.reshape(n_leaf * n_slot, -1), "epath",
            lambda s, t, c=centre_label: ("epath", binning, edge_values[c],
                                          vertex_values[s // n_slot], *arm_fields(s % n_slot),
                                          vertex_values[t // n_slot], *arm_fields(t % n_slot)),
            tally)

    # efork: unordered pairs {y, z} of e0 other than x, by label.
    rest = leaf_count[rows_e].copy()
    rest[np.arange(len(rows_e)), x_column] -= 1
    label_pairs = [(b, c) for b in range(n_leaf) for c in range(b, n_leaf)]
    pairs = np.stack([rest[:, b] * rest[:, c] if b != c else _binomial(rest[:, b], 2)
                      for b, c in label_pairs], axis=1).astype(np.float64)
    for centre_label in range(n_arm):
        for a in range(n_leaf):
            rows = (e0_column == centre_label) & (x_column == a)
            if not rows.any():
                continue
            block = np.rint(arms[rows].T @ pairs[rows]).astype(np.int64)   # slot x label pair
            for slot, p in zip(*np.nonzero(block)):
                b, c = label_pairs[p]
                tally[("efork", binning, edge_values[centre_label], vertex_values[a],
                       *arm_fields(int(slot)), vertex_values[b], vertex_values[c])] += int(block[slot, p])


def binned_tally(tally, binning):
    """Convert an exact-overlap tally to ``binning`` ("log2", "exact" or None)."""
    def convert(value):
        return None if binning is None else overlap_bin(value, binning)

    result = Counter()
    for key, n in tally.items():
        kind = key[0]
        if kind == "ee":
            key = (*key[:4], convert(key[4]))
        elif kind == "path":
            key = (*key[:5], convert(key[5]))
        elif kind == "eee":
            overlaps = None if binning is None else tuple(convert(v) for v in key[5:])
            key = canonical_eee(key[1], key[2:5], overlaps)
        elif kind in DEPTH4_SKELETONS and kind != "vvvv":
            tagged = isinstance(key[1], str)
            if binning is None and tagged:
                raise ValueError("Four-incidence tallies with overlap labels need an overlap-labelled vocabulary")
            if binning is not None and key[1] != binning:
                raise ValueError(f"Four-incidence tallies were not computed with overlap binning {binning!r}")
        result[key] += n
    return result


def pattern_key(code):
    """Tally key of a mined pattern (a ``CandidateCode``).

    Patterns of at most four incidences.
    """
    graph = code.get_hypergraph()
    edges = graph.hyper_edges
    labels = graph.vertex_label
    depth = sum(len(edge.vertices) for edge in edges)
    if depth != len(code.get_extensions()):
        raise ValueError("Pattern depth differs from its incidence count")
    overlaps = graph.edge_overlaps if code.overlap_binning else None

    def overlap(i, j):
        return None if overlaps is None else overlaps[_pair(i, j)]

    if depth == 1:
        return ("i", labels[edges[0].vertices[0]], edges[0].label)
    if depth == 2 and len(edges) == 1:
        return ("vv", edges[0].label, *sorted(labels[v] for v in edges[0].vertices))
    if depth == 2:
        x = edges[0].vertices[0]
        low, high = sorted((edges[0].label, edges[1].label))
        return ("ee", labels[x], low, high, overlap(0, 1))
    if depth == 3 and len(edges) == 1:
        return ("vvv", edges[0].label, *sorted(labels[v] for v in edges[0].vertices))
    if depth == 3 and len(edges) == 3:
        x = edges[0].vertices[0]
        return canonical_eee(labels[x], [edge.label for edge in edges],
                             None if overlaps is None else
                             (overlap(0, 1), overlap(0, 2), overlap(1, 2)))
    if depth == 3 and len(edges) == 2:
        first, second = (0, 1) if len(edges[0].vertices) == 2 else (1, 0)
        x = edges[second].vertices[0]
        y = next(v for v in edges[first].vertices if v != x)
        return ("path", labels[x], labels[y], edges[first].label,
                edges[second].label, overlap(0, 1))
    if depth == DEEP_COUNT_DEPTH and overlaps is None:
        return _depth4_key(edges, labels)
    if depth == DEEP_COUNT_DEPTH:
        return _depth4_overlap_key(edges, labels, overlap, code.overlap_binning)
    raise ValueError(f"Counts support patterns of at most {DEEP_COUNT_DEPTH} incidences")


def _depth4_key(edges, labels):
    """Tally key of a four-incidence pattern without overlap labels."""
    sizes = sorted(len(edge.vertices) for edge in edges)
    if sizes == [4]:
        edge = edges[0]
        return ("vvvv", edge.label, *sorted(labels[v] for v in edge.vertices))
    if sizes == [1, 1, 1, 1]:
        return ("eeee", labels[edges[0].vertices[0]], *sorted(edge.label for edge in edges))
    if sizes == [1, 3]:
        big, small = sorted(edges, key=lambda edge: -len(edge.vertices))
        x = small.vertices[0]
        others = sorted(labels[v] for v in big.vertices if v != x)
        return ("efork", big.label, labels[x], small.label, *others)
    if sizes == [2, 2]:
        first, second = edges
        common = set(first.vertices) & set(second.vertices)
        if len(common) == 2:
            return ("ee2", *sorted((first.label, second.label)),
                    *sorted(labels[v] for v in common))
        (x,) = common
        arms = sorted((edge.label, labels[next(v for v in edge.vertices if v != x)])
                      for edge in edges)
        return ("vpath", labels[x], *arms[0], *arms[1])
    if sizes == [1, 1, 2]:
        pair = next(edge for edge in edges if len(edge.vertices) == 2)
        singles = [edge for edge in edges if len(edge.vertices) == 1]
        if singles[0].vertices[0] == singles[1].vertices[0]:
            x = singles[0].vertices[0]
            y = next(v for v in pair.vertices if v != x)
            return ("vfork", labels[x], pair.label, labels[y],
                    *sorted(edge.label for edge in singles))
        arms = sorted((labels[edge.vertices[0]], edge.label) for edge in singles)
        return ("epath", pair.label, *arms[0], *arms[1])
    raise ValueError(f"Unexpected four-incidence pattern with edge sizes {sizes}")


def _depth4_overlap_key(edges, labels, overlap, binning):
    """Tally key of a four-incidence pattern with overlap labels.

    ``overlap(i, j)`` is the label of edges i and j, defined exactly for the
    edge pairs that share a pattern vertex.
    """
    sizes = sorted(len(edge.vertices) for edge in edges)
    if sizes == [4]:
        return _depth4_key(edges, labels)
    if sizes == [1, 1, 1, 1]:
        return canonical_eeee(labels[edges[0].vertices[0]], [edge.label for edge in edges],
                              {pair: overlap(*pair) for pair in _PAIRS4}, binning)
    if sizes == [1, 3]:
        big, small = (0, 1) if len(edges[0].vertices) == 3 else (1, 0)
        x = edges[small].vertices[0]
        others = sorted(labels[v] for v in edges[big].vertices if v != x)
        return ("efork", binning, edges[big].label, labels[x], edges[small].label,
                overlap(big, small), *others)
    if sizes == [2, 2]:
        first, second = edges
        common = set(first.vertices) & set(second.vertices)
        if len(common) == 2:
            return ("ee2", binning, *sorted((first.label, second.label)),
                    *sorted(labels[v] for v in common), overlap(0, 1))
        (x,) = common
        arms = sorted((edge.label, labels[next(v for v in edge.vertices if v != x)])
                      for edge in edges)
        return ("vpath", binning, labels[x], *arms[0], *arms[1], overlap(0, 1))
    if sizes == [1, 1, 2]:
        pair = next(i for i, edge in enumerate(edges) if len(edge.vertices) == 2)
        s, t = (i for i in range(3) if i != pair)
        if edges[s].vertices[0] == edges[t].vertices[0]:
            x = edges[s].vertices[0]
            y = next(v for v in edges[pair].vertices if v != x)
            arms = sorted((edges[i].label, overlap(pair, i)) for i in (s, t))
            return ("vfork", binning, labels[x], edges[pair].label, labels[y],
                    *arms[0], *arms[1], overlap(s, t))
        arms = sorted((labels[edges[i].vertices[0]], edges[i].label, overlap(pair, i))
                      for i in (s, t))
        return ("epath", binning, edges[pair].label, *arms[0], *arms[1])
    raise ValueError(f"Unexpected four-incidence pattern with edge sizes {sizes}")


def compute_pattern_tallies(hypergraphs, max_depth=MAX_COUNT_DEPTH, overlap_binning=None):
    """Exact-overlap tallies and skeleton totals for every graph."""
    return [graph_tally(graph, max_depth, overlap_binning) for graph in hypergraphs]


def pattern_count_matrix(patterns, tallies=None, *, hypergraphs=None):
    """Counts, skeleton totals and depths for a frozen vocabulary.

    Pass precomputed ``tallies`` (binned labels: small), or ``hypergraphs`` to
    tally one graph at a time and keep only the vocabulary's keys (exact
    labels: a Twitter graph has about 10**5 distinct keys). Returns
    ``(counts, totals, depths)``: graphs x patterns integer arrays and the
    per-pattern incidence count.
    """
    if (tallies is None) == (hypergraphs is None):
        raise ValueError("Pass exactly one of tallies and hypergraphs")
    modes = {code.overlap_binning for code in patterns}
    if len(modes) > 1:
        raise ValueError("Patterns mix overlap modes")
    binning = modes.pop() if modes else None
    keys = [pattern_key(code) for code in patterns]
    if len(set(keys)) != len(keys):
        raise ValueError("Two vocabulary patterns have the same count key")
    skeletons = [key[0] for key in keys]
    depth = max((SKELETON_DEPTH[name] for name in skeletons), default=1)
    depth = DEEP_COUNT_DEPTH if depth > MAX_COUNT_DEPTH else MAX_COUNT_DEPTH
    rows = tallies if tallies is not None else hypergraphs
    counts = np.zeros((len(rows), len(keys)), dtype=np.int64)
    totals = np.zeros((len(rows), len(keys)), dtype=np.int64)
    for row, item in enumerate(rows):
        tally, skeleton_totals = (item if tallies is not None else
                                  graph_tally(item, depth, binning if depth > MAX_COUNT_DEPTH else None))
        if depth > MAX_COUNT_DEPTH and not DEPTH4_SKELETONS <= skeleton_totals.keys():
            raise ValueError("Four-incidence patterns need tallies with max_depth=4")
        converted = binned_tally(tally, binning)
        counts[row] = [converted.get(key, 0) for key in keys]
        totals[row] = [skeleton_totals[name] for name in skeletons]
    depths = np.array([SKELETON_DEPTH[name] for name in skeletons], dtype=np.int64)
    return counts, totals, depths


def encode_pattern_counts(counts, totals, depths, train_index, *, encoding,
                          min_train_graphs=2, min_depth=1):
    """Training-fitted column choice and count encoding.

    Columns are the patterns present in at least ``min_train_graphs`` training
    graphs with at least ``min_depth`` incidences; no label is used. Returns
    the encoded matrix and the chosen column indices.
    """
    if encoding not in COUNT_ENCODINGS:
        raise ValueError(f"mined_count_encoding must be one of {COUNT_ENCODINGS}")
    support = (counts[train_index] > 0).sum(axis=0)
    columns = np.flatnonzero((support >= min_train_graphs) & (depths >= min_depth))
    if len(columns) == 0:
        raise ValueError("No mined pattern occurs in enough training graphs")
    selected = counts[:, columns].astype(np.float64)
    if encoding == "presence":
        values = (selected > 0).astype(np.float64)
    elif encoding == "log":
        values = np.log1p(selected)
    else:
        denominators = totals[:, columns].astype(np.float64)
        values = np.divide(selected, denominators, out=np.zeros_like(selected),
                           where=denominators > 0)
    return values, columns
