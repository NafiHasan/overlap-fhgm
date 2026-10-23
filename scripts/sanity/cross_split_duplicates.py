"""Isomorphic duplicates across partitions.

Counts, for every repeat, the validation and test hypergraphs that are
isomorphic to at least one training hypergraph, ignoring all labels. The
method's features are invariant under isomorphism, so such a hypergraph gets
the same representation as its training copies.

Two hypergraphs are isomorphic when a bijection between their vertices and a
bijection between their hyperedges preserve incidence. Repeated hyperedges are
kept, as the pipeline reads them. Candidates are grouped by colour refinement
on the vertex-hyperedge incidence graph and each candidate pair is then
checked exactly by individualization and refinement.

Also reported:

- label agreement: the share of duplicated held-out hypergraphs whose class is
  the most frequent class among their training copies;
- a lookup reference: predict that class for duplicated hypergraphs and the
  training majority class otherwise, scored like the method;
- conflicting classes: hypergraphs whose isomorphism class, over the whole
  dataset, holds more than one class label.

Writes ``results/sanity/cross_split_duplicates.{json,md}``.

Usage:
    uv run python scripts/sanity/cross_split_duplicates.py
"""

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from dataset_stats import DATASETS, LABEL_MODE, REPEATS, SPLIT_PROTOCOL, load_dataset  # noqa: E402
from config import DATASETS as DATASET_INFO  # noqa: E402
from pipeline.artifacts import atomic_json_write, atomic_text_write  # noqa: E402
from pipeline.provenance import capture_provenance  # noqa: E402
from pipeline.splits import load_and_validate_split_manifest  # noqa: E402

OUTPUT = REPO / "results" / "sanity"


class Incidence:
    """Vertex-hyperedge incidence graph of one hypergraph, with refined colours."""

    def __init__(self, graph):
        vertices = sorted(graph.vertex_label)
        index = {vertex: i for i, vertex in enumerate(vertices)}
        self.n_vertices = len(vertices)
        self.edges = [sorted(index[v] for v in edge.vertices) for edge in graph.hyper_edges]
        self.n = self.n_vertices + len(self.edges)
        self.adjacency = [set() for _ in range(self.n)]
        for j, edge in enumerate(self.edges):
            node = self.n_vertices + j
            for v in edge:
                self.adjacency[v].add(node)
                self.adjacency[node].add(v)
        self.colours = self._refine()
        self.key = hashlib.sha256(repr((
            self.n_vertices, len(self.edges), sorted(Counter(self.colours).items()),
        )).encode()).hexdigest()

    def _refine(self):
        # Hashed colours have a fixed length and are equal for isomorphic inputs.
        colours = ["v" if i < self.n_vertices else "e" for i in range(self.n)]
        for _ in range(self.n):
            new = [hashlib.sha256(repr((colours[i], sorted(colours[j] for j in self.adjacency[i])))
                                  .encode()).hexdigest()[:24] for i in range(self.n)]
            if len(set(new)) == len(set(colours)):
                return colours
            colours = new
        return colours


class SearchBudgetExceeded(Exception):
    pass


SEARCH_BUDGET = 20_000  # colour refinements per pair


def _refine_union(adjacency, colours):
    """Stable colour refinement of a node set; hashed colours stay comparable."""
    while True:
        new = [hashlib.sha256(repr((colours[i], sorted(colours[j] for j in adjacency[i])))
                              .encode()).hexdigest()[:24] for i in range(len(colours))]
        if len(set(new)) == len(set(colours)):
            return colours
        colours = new


def isomorphic(a, b):
    """Exact test by individualization and refinement on the disjoint union.

    Nodes of ``b`` are shifted by ``a.n``. A branch individualizes one node of
    ``a`` and one of ``b`` with the same colour, refines both together and is
    pruned when the two sides' colour counts differ; once every colour holds
    one node per side, the induced bijection is checked edge by edge.
    """
    if a.key != b.key:
        return False
    n = a.n
    adjacency = [set(x) for x in a.adjacency] + [{j + n for j in x} for x in b.adjacency]
    calls = [0]

    def balanced(colours):
        return Counter(colours[:n]) == Counter(colours[n:])

    def search(colours):
        calls[0] += 1
        if calls[0] > SEARCH_BUDGET:
            raise SearchBudgetExceeded
        colours = _refine_union(adjacency, colours)
        if not balanced(colours):
            return False
        classes = defaultdict(lambda: ([], []))
        for i, c in enumerate(colours):
            classes[c][i >= n].append(i)
        open_classes = [members for members in classes.values() if len(members[0]) > 1]
        if not open_classes:
            mapping = {members[0][0]: members[1][0] - n for members in classes.values()}
            return all({mapping[j] for j in a.adjacency[i]} == b.adjacency[mapping[i]]
                       for i in range(n))
        left, right = min(open_classes, key=lambda members: len(members[0]))
        x = left[0]
        for y in right:
            marked = list(colours)
            marked[x] = marked[y] = "*" + colours[x]
            if search(marked):
                return True
        return False

    return search(list(a.colours) + list(b.colours))


def isomorphism_classes(graphs):
    """Class index of every hypergraph (equal index = isomorphic)."""
    structures = [Incidence(graph) for graph in graphs]
    groups = defaultdict(list)
    for i, structure in enumerate(structures):
        groups[structure.key].append(i)
    classes = [None] * len(graphs)
    next_class = undecided = 0
    for members in groups.values():
        representatives = []
        for i in members:
            for rep, cls in representatives:
                try:
                    match = isomorphic(structures[rep], structures[i])
                except SearchBudgetExceeded:
                    # Undecided pairs count as distinct, so duplicates are a lower bound.
                    undecided += 1
                    match = False
                if match:
                    classes[i] = cls
                    break
            else:
                representatives.append((i, next_class))
                classes[i] = next_class
                next_class += 1
    return classes, undecided


def scores(truth, predicted, classes):
    return (float(np.mean(predicted == truth) * 100),
            float(f1_score(truth, predicted, average="macro", labels=classes,
                           zero_division=0) * 100))


def dataset_record(dataset):
    domain, hypergraph_file, labels_file, graphs, labels = load_dataset(dataset)
    split_dir = REPO / "data" / dataset / "splits" / SPLIT_PROTOCOL / LABEL_MODE
    splits = [load_and_validate_split_manifest(
        split_dir / f"repeat_{repeat}.json", dataset=dataset, label_mode=LABEL_MODE,
        hypergraph_file=hypergraph_file, labels_file=labels_file, num_graphs=len(graphs),
        expected_repeat=repeat, expected_domain=domain) for repeat in REPEATS]
    iso, undecided = isomorphism_classes(graphs)
    iso = np.array(iso)
    class_labels = defaultdict(set)
    for cls, label in zip(iso, labels):
        class_labels[cls].add(int(label))
    conflicting = sum(len(class_labels[cls]) > 1 for cls in iso)
    classes = list(range(DATASET_INFO[dataset]["num_classes"]))

    rows = []
    for split in splits:
        train_labels = defaultdict(Counter)
        for i in split.train:
            train_labels[iso[i]][int(labels[i])] += 1
        overall = Counter(labels[split.train].tolist())
        majority = min(overall, key=lambda c: (-overall[c], c))
        row = {"repeat": split.repeat}
        for name, indices in (("val", split.validation), ("test", split.test)):
            duplicated = [i for i in indices if iso[i] in train_labels]
            agree = 0
            predicted = []
            for i in indices:
                counts = train_labels.get(iso[i])
                if counts:
                    guess = min(counts, key=lambda c: (-counts[c], c))
                    agree += i in duplicated and guess == labels[i]
                else:
                    guess = majority
                predicted.append(guess)
            accuracy, macro_f1 = scores(labels[indices], np.array(predicted), classes)
            row.update({
                f"{name}_graphs": len(indices),
                f"{name}_duplicated": len(duplicated),
                f"{name}_duplicated_share": len(duplicated) / len(indices) * 100,
                f"{name}_label_agreement": agree / len(duplicated) * 100 if duplicated else None,
                f"{name}_lookup_accuracy": accuracy,
                f"{name}_lookup_macro_f1": macro_f1,
            })
        rows.append(row)
    aggregate = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None and key != "repeat"]
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
    return {
        "graphs": len(graphs),
        "isomorphism_classes": int(len(set(iso))),
        "graphs_in_conflicting_classes": int(conflicting),
        "undecided_pairs": undecided,
        "repeats": rows,
        "aggregate": aggregate,
        "split_manifest_hashes": {str(s.repeat): s.manifest_hash for s in splits},
    }


def markdown(records):
    lines = [
        "# Isomorphic duplicates across partitions",
        "",
        "Generated by `scripts/sanity/cross_split_duplicates.py`; definitions in its docstring.",
        "Means ± population s.d. over repeats 0-4; labels are ignored when testing isomorphism.",
        "",
        "| Dataset | Graphs | Isomorphism classes | Graphs in classes with >1 label "
        "| Test duplicated (%) | Test label agreement (%) | Lookup test acc. | Lookup test Macro-F1 "
        "| Val duplicated (%) | Undecided pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, name, _ in DATASETS:
        r = records[dataset]
        a = r["aggregate"]

        def fmt(key):
            return (f"{a[key + '_mean']:.2f} ± {a[key + '_std']:.2f}"
                    if key + "_mean" in a else "—")
        lines.append(
            f"| {name} | {r['graphs']:,} | {r['isomorphism_classes']:,} "
            f"| {r['graphs_in_conflicting_classes']:,} | {fmt('test_duplicated_share')} "
            f"| {fmt('test_label_agreement')} | {fmt('test_lookup_accuracy')} "
            f"| {fmt('test_lookup_macro_f1')} | {fmt('val_duplicated_share')} "
            f"| {r['undecided_pairs']} |")
    return "\n".join(lines) + "\n"


def main():
    provenance = capture_provenance()
    records = {}
    for dataset, _, _ in DATASETS:
        records[dataset] = dataset_record(dataset)
        print(dataset, "done", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json_write(OUTPUT / "cross_split_duplicates.json",
                      {"schema_version": 1, "label_mode": LABEL_MODE,
                       "split_protocol": SPLIT_PROTOCOL, "provenance": provenance,
                       "datasets": records})
    atomic_text_write(OUTPUT / "cross_split_duplicates.md", markdown(records))
    print(markdown(records))


if __name__ == "__main__":
    main()
