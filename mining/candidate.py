"""
Candidate code representation and extension utilities for FHGM.

Contains the canonical code representation for sub-hypergraphs and
all functions related to extending, comparing, and validating candidates.
"""

import copy
from itertools import product

from hypergraph import HyperGraph
from hypergraph.h_graph import OVERLAP_BINNINGS


# =============================================================================
# Extension Tuple
# =============================================================================

class ExtensionTuple:
    """
    Represents a single hyperedge in the canonical code, storing
    the vertices and edge label for that tuple.
    """

    def __init__(self, vertices, edge_label):
        self.vertices = vertices
        self.edge_label = edge_label

    def add_vertex(self, id, label):
        for v in self.vertices:
            if v >= id:
                return False
        self.vertices[id] = label


# =============================================================================
# Candidate Code
# =============================================================================

class CandidateCode:
    """
    Canonical code representation of a sub-hypergraph candidate.

    The candidate is built incrementally via extensions. Each extension
    either adds a new hyperedge ("a") or extends the last hyperedge ("e").

    With ``overlap_binning`` ("log2" or "exact") extensions are 5-tuples whose
    last entry lists the overlap labels of the edge pairs the extension makes
    share a vertex; otherwise they are 4-tuples.
    """

    def __init__(self, overlap_binning=None):
        if overlap_binning is not None and overlap_binning not in OVERLAP_BINNINGS:
            raise ValueError(f"overlap_binning must be None or one of {OVERLAP_BINNINGS}")
        self.tuples = []
        self.extensions = []
        self.last_edge = -1
        self.last_vertex = -1
        self.overlap_binning = overlap_binning
        # {(earlier edge, later edge): overlap label}
        self.overlaps = {}

    def extend(self, ext):
        """
        Apply an extension to this candidate.

        Args:
            ext: Tuple of (type, time/vertex_id, vertex_label, edge_label),
                 plus (earlier edge, overlap label) pairs in overlap mode.
                 type "a": add a new hyperedge.
                 type "e": extend the current last hyperedge.

        Returns:
            None on success, False if extension is invalid.
        """
        if len(ext) != (5 if self.overlap_binning else 4):
            raise ValueError("Extension length does not match the overlap mode")
        self.extensions.append(ext)
        if ext[0] == "a":
            t = ExtensionTuple({ext[1]: ext[2]}, ext[3])
            for v in t.vertices:
                if v > self.last_vertex:
                    self.last_vertex = v
            self.last_edge += 1
            self.tuples.append(t)
            self._add_overlaps(ext)
        else:
            vertex_id, vertex_label = ext[1], ext[2]
            if vertex_id is None:
                self.last_vertex += 1
                self.tuples[self.last_edge].add_vertex(self.last_vertex, vertex_label)
                self._add_overlaps(ext)
            else:
                self._add_overlaps(ext)
                return self.tuples[self.last_edge].add_vertex(vertex_id, vertex_label)

    def _add_overlaps(self, ext):
        if not self.overlap_binning:
            return
        for earlier, label in ext[4]:
            if not 0 <= earlier < self.last_edge or (earlier, self.last_edge) in self.overlaps:
                raise ValueError("Invalid overlap label in extension")
            self.overlaps[(earlier, self.last_edge)] = label

    def get_hypergraph(self):
        """Convert this candidate code to a HyperGraph instance."""
        h = HyperGraph()
        for t in self.tuples:
            vertices = []
            for v in t.vertices:
                vertices.append(v)
                h.add_vertex(v, t.vertices[v])
            h.add_hyper_edge(vertices, t.edge_label)
        if self.overlap_binning:
            h.set_edge_overlaps(self.overlaps, self.overlap_binning)
        return h

    def get_extensions(self):
        return self.extensions

    def get_last_edge(self):
        return self.last_edge

    def _signature(self):
        """
        Canonical, hashable signature of this candidate.

        Every code stored during mining has passed ``is_minimum``, so two
        structurally identical sub-hypergraphs have identical extension
        sequences. The extension tuples contain only immutable values
        (str/int/None), so the sequence is directly hashable and serves as
        a sound structural identity for deduplication.
        """
        return tuple(self.extensions)

    def __eq__(self, other):
        if not isinstance(other, CandidateCode):
            return NotImplemented
        return self._signature() == other._signature()

    def __hash__(self):
        return hash(self._signature())

    def __str__(self):
        result = ""
        for t in self.tuples:
            result += str(t.vertices) + " " + str(t.edge_label) + "\n"
        return result


# =============================================================================
# Extension comparison and canonical ordering
# =============================================================================

def _min_by_labels(ext1, ext2, v_label1, v_label2):
    """Vertex label first, then the overlap labels of 5-tuple extensions."""
    if v_label1 != v_label2:
        return ext1 if v_label1 < v_label2 else ext2
    if len(ext1) > 4 and ext1[4] < ext2[4]:
        return ext1
    return ext2


def min_ext(ext1, ext2):
    """Given two extensions, return the minimum one according to canonical ordering.

    Overlap labels (5-tuples) break ties only between extensions that are
    equal in all other fields, so 4-tuple ordering is unchanged.
    """
    ext_type1, time1, v_label1, e_label1 = ext1[:4]
    ext_type2, time2, v_label2, e_label2 = ext2[:4]
    if ext_type1 == "a" and ext_type2 == "e":
        return ext2
    if ext_type1 == "e" and ext_type2 == "a":
        return ext1
    if ext_type1 == "a" and ext_type2 == "a":
        if time1 < time2:
            return ext1
        elif time1 > time2:
            return ext2
        elif e_label1 < e_label2:
            return ext1
        elif e_label1 > e_label2:
            return ext2
        elif e_label1 == e_label2:
            return _min_by_labels(ext1, ext2, v_label1, v_label2)
    if ext_type1 == "e" and ext_type2 == "e":
        if time1 is None and time2 is None:
            return _min_by_labels(ext1, ext2, v_label1, v_label2)
        elif time1 is None:
            return ext2
        elif time2 is None:
            return ext1
        elif time1 < time2:
            return ext1
        elif time1 > time2:
            return ext2
        elif time1 == time2:
            return _min_by_labels(ext1, ext2, v_label1, v_label2)


def get_minimum_ext(extensions):
    """Return the minimum extension from a dictionary of extensions."""
    result = None
    for ext in extensions:
        if result is None:
            result = ext
        else:
            result = min_ext(result, ext)
    return result


def is_minimum(code):
    """
    Determine if a candidate code is in canonical (minimum) form.

    This prevents duplicate sub-hypergraph generation during DFS mining.
    """
    h = code.get_hypergraph()
    c = CandidateCode(code.overlap_binning)
    for i in range(len(code.get_extensions())):
        ext = get_minimum_ext(get_extensions(c, [h]))
        if ext != code.get_extensions()[i]:
            return False
        c.extend(ext)
    return True


# =============================================================================
# Extension discovery and application
# =============================================================================

def _potential_extensions(code, code_graph, target):
    """Return a deterministic, safely filtered extension search space.

    These are necessary structural possibilities, not confirmed extensions.
    Exact presence of the extended pattern is checked separately.
    """
    potential = {}
    last_edge = code_graph.get_hyperedge(code.get_last_edge())

    # A new vertex can extend the last pattern edge only if its label occurs in
    # a target edge with the same edge label.
    for target_edge in target._label_to_edges.get(last_edge.label, []):
        for target_vertex_id in target_edge.vertices:
            extension = (
                "e", None, target.vertex_label[target_vertex_id], None
            )
            potential.setdefault(extension, None)

    # Any already-mapped pattern vertex outside the last edge may structurally
    # extend it. Keep even canonically invalid IDs here because is_minimum()
    # relies on seeing all structural alternatives; extend_code() still rejects
    # extensions that violate the code ordering.
    last_vertices = set(last_edge.vertices)
    for pattern_vertex_id, vertex_label in code_graph.vertex_label.items():
        if pattern_vertex_id not in last_vertices:
            extension = ("e", pattern_vertex_id, vertex_label, None)
            potential.setdefault(extension, None)

    # A new edge can be anchored at a mapped pattern vertex only when the
    # target contains that vertex-label/edge-label incidence somewhere.
    incident_edge_labels = {}
    for target_edge in target.hyper_edges:
        for target_vertex_id in target_edge.vertices:
            vertex_label = target.vertex_label[target_vertex_id]
            labels = incident_edge_labels.setdefault(vertex_label, {})
            labels.setdefault(target_edge.label, None)
    for pattern_vertex_id, vertex_label in code_graph.vertex_label.items():
        for edge_label in incident_edge_labels.get(vertex_label, {}):
            extension = ("a", pattern_vertex_id, vertex_label, edge_label)
            potential.setdefault(extension, None)

    return potential


def _build_extended_hypergraph(code, code_graph, extension):
    """Construct the structural pattern represented by one raw extension.

    A 4-tuple applied to an overlap-mode code keeps the code's overlap labels
    and leaves the new edge pairs unconstrained, so one search finds its 5-tuples.
    """
    extended = copy.deepcopy(code_graph)
    extension_type, vertex_id, vertex_label, edge_label = extension[:4]
    if extension_type == "a":
        extended.add_hyper_edge([vertex_id], edge_label)
        changed_edge = extended.edge_count - 1
    elif extension_type == "e":
        changed_edge = code.get_last_edge()
        if vertex_id is None:
            new_vertex_id = code.last_vertex + 1
            extended.add_vertex(new_vertex_id, vertex_label)
            extended.extend_hyper_edge(changed_edge, new_vertex_id)
        else:
            extended.extend_hyper_edge(changed_edge, vertex_id)
    else:
        raise ValueError(f"Unknown extension type: {extension_type!r}")
    if code.overlap_binning:
        overlaps = dict(code.overlaps)
        for earlier, label in (extension[4] if len(extension) > 4 else ()):
            overlaps[(earlier, changed_edge)] = label
        extended.set_edge_overlaps(overlaps, code.overlap_binning)
    return extended


def _new_overlap_pairs(code, code_graph, extension):
    """Edges that first share a vertex with the new or last edge, with that vertex."""
    extension_type, vertex_id = extension[0], extension[1]
    if vertex_id is None:
        return []
    if extension_type == "a":
        return [(edge.id, vertex_id) for edge in code_graph.hyper_edges
                if vertex_id in edge.vertices]
    last = code_graph.get_hyperedge(code.get_last_edge())
    last_vertices = set(last.vertices)
    return [(edge.id, vertex_id) for edge in code_graph.hyper_edges
            if edge.id != last.id and vertex_id in edge.vertices
            and not last_vertices.intersection(edge.vertices)]


def _overlap_extensions(code, code_graph, extension, target):
    """All 5-tuple versions of a 4-tuple extension whose labels occur in ``target``.

    Candidate labels for each new pair come from the target's overlap index at
    the shared vertex label; exact presence is checked by the caller.
    """
    pairs = _new_overlap_pairs(code, code_graph, extension)
    if not pairs:
        return [extension + ((),)]
    if extension[0] == "a":
        new_label = extension[3]
    else:
        new_label = code_graph.get_hyperedge(code.get_last_edge()).label
    index = target.overlap_pair_index(code.overlap_binning)
    choices = []
    for edge_id, vertex_id in pairs:
        low, high = sorted((code_graph.get_hyperedge(edge_id).label, new_label))
        labels = index.get((code_graph.vertex_label[vertex_id], low, high))
        if not labels:
            return []
        choices.append(sorted(labels))
    edge_ids = [edge_id for edge_id, _ in pairs]
    return [extension + (tuple(zip(edge_ids, labels)),) for labels in product(*choices)]


def get_extensions(code, hypergraphs, track_graph_ids=False):
    """
    Find all possible extensions for a candidate code against a set of hypergraphs.

    Args:
        code (CandidateCode): Current sub-hypergraph candidate.
        hypergraphs (list): Database of hypergraphs to search.
        track_graph_ids (bool): If True, also track which graph indices
            contain each extension (for coverage propagation).

    Returns:
        dict: Mapping from extension tuple to support count
              (number of hypergraphs containing that extension).
              If track_graph_ids is True, values are (count, [graph_indices]).
    """
    extensions = dict()
    code_graph = code.get_hypergraph() if code.tuples else None
    extended_graphs = {}
    for h_idx, h in enumerate(hypergraphs):
        temp_extensions = dict()
        if len(code.tuples) == 0:
            suffix = ((),) if code.overlap_binning else ()
            for he in h.hyper_edges:
                for v in he.vertices:
                    temp_extensions[("a", 0, h.vertex_label[v], he.label) + suffix] = 0
        else:
            # Every extension contains the base pattern. One exact gate avoids
            # repeating doomed extended-pattern searches on non-supporting
            # graphs, which remain in the phase pool at deeper DFS levels.
            if not code_graph.is_subhypergraph_of(h):
                continue
            # An occurrence emits an extension exactly when the pattern with
            # that one structural addition occurs. Query that projection
            # directly instead of enumerating capped, symmetric occurrences.
            for extension in _potential_extensions(code, code_graph, h):
                if extension not in extended_graphs:
                    extended_graphs[extension] = _build_extended_hypergraph(
                        code, code_graph, extension
                    )
                probe = extended_graphs[extension]
                if not code.overlap_binning:
                    if probe.is_subhypergraph_of(h):
                        temp_extensions[extension] = 0
                    continue
                labelled = _overlap_extensions(code, code_graph, extension, h)
                if labelled and not labelled[0][4]:
                    if probe.is_subhypergraph_of(h):
                        temp_extensions[labelled[0]] = 0
                    continue
                # The 4-tuple leaves the new pairs unconstrained; one search
                # finds which of the candidate label tuples its matches take.
                changed = (probe.edge_count - 1 if extension[0] == "a"
                           else code.get_last_edge())
                pairs = [(edge_id, changed) for edge_id, _ in labelled[0][4]] if labelled else []
                realized = probe.realized_pair_labels(
                    h, pairs, {tuple(label for _, label in ext[4]) for ext in labelled})
                for ext in labelled:
                    if tuple(label for _, label in ext[4]) in realized:
                        temp_extensions[ext] = 0

        if track_graph_ids:
            for key in temp_extensions:
                if key in extensions:
                    count, graph_ids = extensions[key]
                    extensions[key] = (count + 1, graph_ids + [h_idx])
                else:
                    extensions[key] = (1, [h_idx])
        else:
            for key in temp_extensions:
                if key in extensions:
                    extensions[key] += 1
                else:
                    extensions[key] = 1

    return extensions


def extend_code(code, extensions, tracked=False):
    """
    Generate new candidate codes by applying each extension to the current code.

    Args:
        code (CandidateCode): Current candidate.
        extensions (dict): Extensions from get_extensions().
        tracked (bool): If True, extensions values are (count, [graph_ids]).

    Returns:
        list: List of (new_code, support_count) tuples.
              If tracked, list of (new_code, support_count, [graph_ids]) tuples.
    """
    new_codes = []
    for ext in extensions:
        new_code = copy.deepcopy(code)
        if new_code.extend(ext) is None:
            if tracked:
                count, graph_ids = extensions[ext]
                new_codes.append((new_code, count, graph_ids))
            else:
                new_codes.append((new_code, extensions[ext]))
    return new_codes
