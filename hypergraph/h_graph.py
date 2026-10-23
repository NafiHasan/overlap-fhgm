"""HyperGraph data structure with exact sub-hypergraph matching."""

from collections import Counter, defaultdict

import numpy as np

from .h_edge import HyperEdge

OVERLAP_BINNINGS = ("log2", "exact")


def overlap_bin(size, binning):
    """Overlap label of two hyperedges that share ``size`` vertices.

    ``log2`` uses the bins of the degree and cardinality labels
    (1, 2, 3-4, 5-8, ...); ``exact`` keeps the size.
    """
    if type(size) is not int or size < 0:
        raise ValueError("Overlap sizes must be non-negative integers")
    if binning == "exact":
        return size
    if binning == "log2":
        return 0 if size == 0 else (size - 1).bit_length() + 1
    raise ValueError(f"overlap binning must be one of {OVERLAP_BINNINGS}")


class HyperGraph:
    """
    Hypergraph data structure.

    Attributes:
        vertex_label (dict): Mapping from vertex ID to vertex label.
        vertex_label_bucket (dict): Mapping from vertex label to count of vertices with that label.
        edge_label_bucket (dict): Mapping from edge label to count of edges with that label.
        hyper_edges (list): List of HyperEdge objects in the hypergraph.
        vertex_count (int): Total number of vertices in the hypergraph.
        edge_count (int): Total number of edges in the hypergraph.
    """

    def __init__(self):
        self.vertex_label = dict()
        self.vertex_label_bucket = dict()
        self.edge_label_bucket = dict()
        self.hyper_edges = []
        self.vertex_count = 0
        self.edge_count = 0
        # Lookup indices: label -> list of IDs with that label
        self._label_to_vertices = defaultdict(list)
        self._label_to_edges = defaultdict(list)
        # Overlap-labelled patterns: {(i, j): label} for edge pairs i < j that
        # share a pattern vertex, and the binning applied to target overlaps.
        # Data graphs keep None and compute pair overlaps on demand.
        self.edge_overlaps = None
        self.overlap_binning = None
        self._overlap_sizes = {}
        self._overlap_labels = {}
        self._pair_index = {}
        self._constrained = None
        self._edge_vertex_labels = {}
        self._edge_candidate_cache = {}
        self._label_matrices = {}
        self._incident_edge_sets = None
        self._candidate_sets = {}

    # -----------------------------------------------------------------
    # Edge-pair overlaps
    # -----------------------------------------------------------------

    def set_edge_overlaps(self, overlaps, binning):
        """Store pattern overlap labels; ``overlaps`` maps (i, j), i < j, to a label."""
        if binning not in OVERLAP_BINNINGS:
            raise ValueError(f"overlap binning must be one of {OVERLAP_BINNINGS}")
        self.edge_overlaps = dict(overlaps)
        self.overlap_binning = binning
        self._pair_index = {}
        self._constrained = None

    def edge_pair_label(self, a, b, binning):
        """Overlap label of edges ``a`` and ``b``.

        A pattern returns its stored label (None for a pair that shares no
        pattern vertex); a data graph bins the intersection size.
        """
        key = (a, b) if a < b else (b, a)
        if self.edge_overlaps is not None:
            return self.edge_overlaps.get(key)
        labels = self._overlap_labels.get(binning)
        if labels is None:
            labels = self._overlap_labels[binning] = {}
        label = labels.get(key)
        if label is None:
            size = self._overlap_sizes.get(key)
            if size is None:
                size = len(set(self.hyper_edges[a].vertices).intersection(
                    self.hyper_edges[b].vertices))
                self._overlap_sizes[key] = size
            label = labels[key] = overlap_bin(size, binning)
        return label

    def overlap_pair_index(self, binning):
        """Overlap labels seen at each (vertex label, edge label, edge label).

        Maps the labels of a common vertex and of two distinct edges at it
        (edge labels sorted) to the set of their overlap labels. A pattern
        pair can match only if its overlap label is in the set.
        """
        index = self._pair_index.get(binning)
        if index is None:
            incident = defaultdict(list)
            for edge in self.hyper_edges:
                for vertex_id in edge.vertices:
                    incident[vertex_id].append(edge)
            index = defaultdict(set)
            for vertex_id, edges in incident.items():
                label = self.vertex_label[vertex_id]
                for position, first in enumerate(edges):
                    for second in edges[position + 1:]:
                        low, high = sorted((first.label, second.label))
                        index[(label, low, high)].add(
                            self.edge_pair_label(first.id, second.id, binning))
            index = dict(index)
            self._pair_index[binning] = index
        return index

    def overlap_label_matrix(self, binning):
        """Edges x edges array of ``edge_pair_label`` values (-2 for None)."""
        matrix = self._label_matrices.get(binning)
        if matrix is None:
            n = self.edge_count
            matrix = np.full((n, n), -2, dtype=np.int64)
            if self.edge_overlaps is not None:
                for (i, j), label in self.edge_overlaps.items():
                    matrix[i, j] = matrix[j, i] = label
            elif n:
                position = {vertex: index for index, vertex in enumerate(self.vertex_label)}
                incidence = np.zeros((len(position), n), dtype=np.int64)
                for edge in self.hyper_edges:
                    incidence[[position[v] for v in edge.vertices], edge.id] = 1
                sizes = incidence.T @ incidence
                table = np.array([overlap_bin(size, binning)
                                  for size in range(int(sizes.max()) + 1)], dtype=np.int64)
                matrix = table[sizes]
            self._label_matrices[binning] = matrix
        return matrix

    def constrained_pairs(self):
        """Pattern edge pairs with an overlap label: [(i, j, shared vertex, label)]."""
        if not self.edge_overlaps:
            return []
        if self._constrained is not None:
            return self._constrained
        pairs = []
        for (i, j), label in sorted(self.edge_overlaps.items()):
            shared = set(self.hyper_edges[i].vertices).intersection(
                self.hyper_edges[j].vertices)
            if not shared:
                raise ValueError("Overlap label on a pattern edge pair without a shared vertex")
            pairs.append((i, j, min(shared), label))
        self._constrained = pairs
        return pairs

    # -----------------------------------------------------------------
    # Sub-hypergraph isomorphism
    # -----------------------------------------------------------------

    def is_subhypergraph_of(self, H):
        """Return exactly whether ``self`` occurs in ``H``.

        Pattern vertices and hyperedges map injectively to target vertices and
        hyperedges with equal labels. A mapped pattern hyperedge may be a subset
        of its target hyperedge, matching the containment semantics used by the
        miner while candidates are grown one vertex at a time.

        The search stops at the first match and has no search cap. It prunes
        partial vertex assignments whenever the pattern hyperedges can no longer
        be assigned injectively to compatible target hyperedges.
        """
        if not self._presence_precheck(H):
            return False

        edge_candidates = self._presence_edge_candidates(H)
        if edge_candidates is None:
            return False
        relevant_target_edges = {
            target_edge_id
            for candidates in edge_candidates.values()
            for target_edge_id in candidates
        }
        target_edge_vertices = {
            edge.id: frozenset(edge.vertices)
            for edge in H.hyper_edges
            if edge.id in relevant_target_edges
        }
        if not self._has_injective_edge_assignment(
                edge_candidates, {}, target_edge_vertices):
            return False

        incident_edges = {vertex_id: [] for vertex_id in self.vertex_label}
        for edge in self.hyper_edges:
            for vertex_id in edge.vertices:
                incident_edges[vertex_id].append(edge.id)

        target_domains = self._presence_vertex_domains(
            H,
            incident_edges,
            edge_candidates,
            target_edge_vertices,
        )
        if target_domains is None:
            return False

        insertion_order = {
            vertex_id: index
            for index, vertex_id in enumerate(self.vertex_label)
        }
        vertices = sorted(
            self.vertex_label,
            key=lambda vertex_id: (
                len(target_domains[vertex_id]),
                -len(incident_edges[vertex_id]),
                insertion_order[vertex_id],
            ),
        )
        return self._has_exact_isomorphism_v(
            0,
            vertices,
            {},
            set(),
            target_domains,
            edge_candidates,
            target_edge_vertices,
            target=H,
            index=H._matching_index(edge_candidates),
        )

    def realized_pair_labels(self, H, new_pairs, wanted):
        """Overlap labels that unconstrained pattern edge pairs take in exact matches.

        ``self`` is an overlap-labelled pattern whose pairs ``new_pairs``
        ([(i, j)]) carry no label yet. Returns the subset of ``wanted`` (tuples
        of labels, one per pair in order) realised by some exact match in
        ``H``. One search replaces a presence check per labelled variant and
        stops as soon as every wanted tuple is found.
        """
        found = set()
        if not wanted or not self._presence_precheck(H):
            return found
        edge_candidates = self._presence_edge_candidates(H)
        if edge_candidates is None:
            return found
        relevant = {edge for candidates in edge_candidates.values() for edge in candidates}
        target_edge_vertices = {edge.id: frozenset(edge.vertices)
                                for edge in H.hyper_edges if edge.id in relevant}
        if not self._has_injective_edge_assignment(edge_candidates, {}, target_edge_vertices):
            return found
        incident_edges = {vertex_id: [] for vertex_id in self.vertex_label}
        for edge in self.hyper_edges:
            for vertex_id in edge.vertices:
                incident_edges[vertex_id].append(edge.id)
        domains = self._presence_vertex_domains(
            H, incident_edges, edge_candidates, target_edge_vertices)
        if domains is None:
            return found
        order = {vertex_id: index for index, vertex_id in enumerate(self.vertex_label)}
        vertices = sorted(self.vertex_label, key=lambda v: (
            len(domains[v]), -len(incident_edges[v]), order[v]))
        binning = self.overlap_binning
        wanted = set(wanted)

        labels = H.overlap_label_matrix(binning)
        matching_index = H._matching_index(edge_candidates)
        incident, candidate_sets = matching_index
        pair_constraints = [(i, j, label) for i, j, _vertex_id, label in self.constrained_pairs()]
        edge_count = len(self.hyper_edges)

        def leaf(v_iso):
            # Every exact edge assignment at once: injective, contained, and
            # with the labelled pairs equal; collect the new pairs' labels.
            arrays = []
            for edge in self.hyper_edges:
                shared = candidate_sets[edge.id]
                for vertex_id in edge.vertices:
                    shared = shared & incident[v_iso[vertex_id]]
                candidates = sorted(shared)
                if not candidates:
                    return False
                arrays.append(np.array(candidates, dtype=np.int64))
            grids = np.ix_(*arrays)
            shape = tuple(len(array) for array in arrays)
            valid = np.ones(shape, dtype=bool)
            for a in range(edge_count):
                for b in range(a + 1, edge_count):
                    valid &= grids[a] != grids[b]
            for i, j, label in pair_constraints:
                valid &= labels[grids[i], grids[j]] == label
            if not valid.any():
                return False
            columns = [np.broadcast_to(labels[grids[i], grids[j]], shape)[valid]
                       for i, j in new_pairs]
            found.update(map(tuple, np.unique(np.stack(columns, axis=1), axis=0).tolist()))
            return found >= wanted

        def assign(index, v_iso, used_targets):
            if index == len(vertices):
                return leaf(v_iso)
            vertex_id = vertices[index]
            for target in domains[vertex_id]:
                if target in used_targets:
                    continue
                v_iso[vertex_id] = target
                used_targets.add(target)
                done = (self._has_injective_edge_assignment(
                            edge_candidates, v_iso, target_edge_vertices,
                            index=matching_index)
                        and assign(index + 1, v_iso, used_targets))
                used_targets.remove(target)
                del v_iso[vertex_id]
                if done:
                    return True
            return False

        assign(0, {}, set())
        return found & wanted

    def _presence_precheck(self, H):
        """Reject count and label combinations that cannot possibly match."""
        if self.vertex_count > H.vertex_count or self.edge_count > H.edge_count:
            return False
        for label, count in self.vertex_label_bucket.items():
            if H.vertex_label_bucket.get(label, 0) < count:
                return False
        for label, count in self.edge_label_bucket.items():
            if H.edge_label_bucket.get(label, 0) < count:
                return False
        if self.edge_overlaps:
            index = H.overlap_pair_index(self.overlap_binning)
            for i, j, vertex_id, label in self.constrained_pairs():
                low, high = sorted((self.hyper_edges[i].label, self.hyper_edges[j].label))
                if label not in index.get((self.vertex_label[vertex_id], low, high), ()):
                    return False
        return True

    def _presence_edge_candidates(self, H):
        """Build label/cardinality-compatible target domains for pattern edges."""
        candidates = {}
        for edge in self.hyper_edges:
            compatible = H._compatible_edges(
                edge.label, len(edge.vertices),
                Counter(self.vertex_label[vertex_id] for vertex_id in edge.vertices))
            if not compatible:
                return None
            candidates[edge.id] = compatible
        return candidates

    def _matching_index(self, edge_candidates):
        """Per-vertex incident edge sets and candidate sets for faster pruning.

        Candidate lists are in ascending edge ID, so sorted intersections
        reproduce the filtered lists exactly.
        """
        if self._incident_edge_sets is None:
            incident = defaultdict(set)
            for edge in self.hyper_edges:
                for vertex_id in edge.vertices:
                    incident[vertex_id].add(edge.id)
            self._incident_edge_sets = {v: frozenset(e) for v, e in incident.items()}
        sets = {}
        for edge_id, candidates in edge_candidates.items():
            cached = self._candidate_sets.get(id(candidates))
            if cached is None or cached[0] is not candidates:
                cached = (candidates, frozenset(candidates))
                self._candidate_sets[id(candidates)] = cached
            sets[edge_id] = cached[1]
        return self._incident_edge_sets, sets

    def _compatible_edges(self, edge_label, size, required_labels):
        """Edges of this graph that can hold a pattern edge with these labels.

        Cached per requirement on the graph; this changes cost, not results.
        """
        key = (edge_label, size, frozenset(required_labels.items()))
        compatible = self._edge_candidate_cache.get(key)
        if compatible is None:
            compatible = []
            for target_edge in self._label_to_edges.get(edge_label, []):
                if len(target_edge.vertices) < size:
                    continue
                available = self._edge_vertex_labels.get(target_edge.id)
                if available is None:
                    available = Counter(self.vertex_label[vertex_id]
                                        for vertex_id in target_edge.vertices)
                    self._edge_vertex_labels[target_edge.id] = available
                if any(available.get(label, 0) < count
                       for label, count in required_labels.items()):
                    continue
                compatible.append(target_edge.id)
            self._edge_candidate_cache[key] = compatible
        return compatible

    def _presence_vertex_domains(self, H, incident_edges, edge_candidates,
                                 target_edge_vertices):
        """Restrict each pattern vertex to label- and edge-compatible targets."""
        domains = {}
        for vertex_id, label in self.vertex_label.items():
            allowed = set(H._label_to_vertices.get(label, []))
            for edge_id in incident_edges[vertex_id]:
                vertices_in_candidate_edges = set()
                for target_edge_id in edge_candidates[edge_id]:
                    vertices_in_candidate_edges.update(
                        target_edge_vertices[target_edge_id]
                    )
                allowed.intersection_update(vertices_in_candidate_edges)
            if not allowed:
                return None
            # Preserve target insertion order so exact matching is deterministic.
            domains[vertex_id] = [
                candidate
                for candidate in H._label_to_vertices.get(label, [])
                if candidate in allowed
            ]
        return domains

    def _has_exact_isomorphism_v(self, i, vertices, v_iso, used_targets,
                                 target_domains, edge_candidates,
                                 target_edge_vertices, target=None, index=None):
        """Find the first exact presence mapping with structure-aware pruning.

        ``target`` is needed only to check overlap labels at the leaves.
        """
        if i == len(vertices):
            return self._has_injective_edge_assignment(
                edge_candidates, v_iso, target_edge_vertices, target=target,
                index=index,
            )

        vertex_id = vertices[i]
        for target_vertex_id in target_domains[vertex_id]:
            if target_vertex_id in used_targets:
                continue
            v_iso[vertex_id] = target_vertex_id
            used_targets.add(target_vertex_id)
            feasible = self._has_injective_edge_assignment(
                edge_candidates, v_iso, target_edge_vertices, index=index
            )
            found = feasible and self._has_exact_isomorphism_v(
                i + 1,
                vertices,
                v_iso,
                used_targets,
                target_domains,
                edge_candidates,
                target_edge_vertices,
                target,
                index,
            )
            used_targets.remove(target_vertex_id)
            del v_iso[vertex_id]
            if found:
                return True
        return False

    def _has_injective_edge_assignment(self, edge_candidates, v_iso,
                                        target_edge_vertices, target=None,
                                        index=None):
        """Check whether partial vertex mappings permit distinct target edges.

        With ``target`` (only once every pattern vertex is mapped) the
        assignment must also give every overlap-labelled pattern edge pair a
        target pair with the same overlap label; that is a small exact search
        instead of bipartite matching. Without it the check ignores overlap
        labels, which keeps it a valid relaxation for pruning.
        """
        available = {}
        for edge in self.hyper_edges:
            mapped_vertices = [
                v_iso[vertex_id]
                for vertex_id in edge.vertices
                if vertex_id in v_iso
            ]
            if index is not None and mapped_vertices:
                # ``index`` = (target vertex -> incident edge IDs, pattern
                # edge -> candidate set); same list as the filter below.
                incident, candidate_sets = index
                shared = incident[mapped_vertices[0]]
                for vertex_id in mapped_vertices[1:]:
                    shared = shared & incident[vertex_id]
                compatible = sorted(shared & candidate_sets[edge.id])
            else:
                compatible = [
                    target_edge_id
                    for target_edge_id in edge_candidates[edge.id]
                    if all(vertex_id in target_edge_vertices[target_edge_id]
                           for vertex_id in mapped_vertices)
                ]
            if not compatible:
                return False
            available[edge.id] = compatible

        if target is not None and self.edge_overlaps:
            return self._has_overlap_edge_assignment(available, target)

        target_owner = {}

        def assign(pattern_edge_id, visited_targets):
            for target_edge_id in available[pattern_edge_id]:
                if target_edge_id in visited_targets:
                    continue
                visited_targets.add(target_edge_id)
                previous_owner = target_owner.get(target_edge_id)
                if previous_owner is None or assign(
                        previous_owner, visited_targets):
                    target_owner[target_edge_id] = pattern_edge_id
                    return True
            return False

        for pattern_edge_id in sorted(
                available, key=lambda item: len(available[item])):
            if not assign(pattern_edge_id, set()):
                return False
        return True

    def _has_overlap_edge_assignment(self, available, target):
        """Exact injective edge assignment that respects pair overlap labels."""
        constraints = {edge_id: [] for edge_id in available}
        for i, j, _vertex_id, label in self.constrained_pairs():
            constraints[i].append((j, label))
            constraints[j].append((i, label))
        order = sorted(available, key=lambda edge_id: len(available[edge_id]))
        assigned, used = {}, set()
        binning = self.overlap_binning

        def search(position):
            if position == len(order):
                return True
            edge_id = order[position]
            for target_edge_id in available[edge_id]:
                if target_edge_id in used:
                    continue
                if any(other in assigned and
                       target.edge_pair_label(target_edge_id, assigned[other], binning) != label
                       for other, label in constraints[edge_id]):
                    continue
                assigned[edge_id] = target_edge_id
                used.add(target_edge_id)
                if search(position + 1):
                    return True
                del assigned[edge_id]
                used.remove(target_edge_id)
            return False

        return search(0)

    # -----------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------

    def add_vertex(self, v_id, v_label):
        if v_id in self.vertex_label and self.vertex_label[v_id] == v_label:
            return
        if v_id in self.vertex_label:
            raise ValueError("Vertex ID exists")
        self.vertex_label[v_id] = v_label
        if v_label in self.vertex_label_bucket:
            self.vertex_label_bucket[v_label] = self.vertex_label_bucket[v_label] + 1
        else:
            self.vertex_label_bucket[v_label] = 1
        self._label_to_vertices[v_label].append(v_id)
        self.vertex_count += 1

    def add_hyper_edge(self, vertices, edge_label):
        edge = HyperEdge(self.edge_count, vertices, edge_label)
        self.hyper_edges.append(edge)
        if edge_label in self.edge_label_bucket:
            self.edge_label_bucket[edge_label] += 1
        else:
            self.edge_label_bucket[edge_label] = 1
        self._label_to_edges[edge_label].append(edge)
        self.edge_count += 1
        self._invalidate_pair_caches()

    def extend_hyper_edge(self, edge_id, vertex_id):
        self.hyper_edges[edge_id].vertices.append(vertex_id)
        self._invalidate_pair_caches()

    def _invalidate_pair_caches(self):
        self._overlap_sizes = {}
        self._overlap_labels = {}
        self._pair_index = {}
        self._constrained = None
        self._edge_vertex_labels = {}
        self._edge_candidate_cache = {}
        self._label_matrices = {}
        self._incident_edge_sets = None
        self._candidate_sets = {}

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    def get_hyperedge(self, id):
        return self.hyper_edges[id]

    # -----------------------------------------------------------------
    # String representation
    # -----------------------------------------------------------------

    def __str__(self):
        result = ""
        for v in self.vertex_label:
            result += "v " + str(v) + " " + str(self.vertex_label[v]) + "\n"
        for e in self.hyper_edges:
            result += "e " + str(e.vertices) + " " + str(e.label) + "\n"
        for (i, j), label in sorted((self.edge_overlaps or {}).items()):
            result += "o " + str(i) + " " + str(j) + " " + str(label) + "\n"
        return result
