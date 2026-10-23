"""Dataset statistics and the majority-class reference.

Reads the converted inputs (``data/<dataset>/composite/``), the class labels
and the DHG-Bench split manifests, and writes
``results/dataset_stats/dataset_stats.{json,md}``. Nothing is fitted except
the majority class, which is taken from each repeat's training labels.

Statistics, per dataset:

- average numbers of vertices and hyperedges per hypergraph; repeated
  hyperedges count separately (as the pipeline sees them), and a second column
  merges them (as DHG-Bench and Feng et al. count them);
- the share of hypergraphs in the largest class;
- overlap: over all edge-pair incidences, i.e. every vertex together with an
  unordered pair of distinct hyperedges that contain it (counted once per
  shared vertex, as the size-2 skeleton of two hyperedges at one vertex), the
  share whose two hyperedges share at least two vertices; and the share of
  hypergraphs with at least one such pair;
- partition sizes of the five repeats.

Majority-class reference: every validation and test hypergraph is assigned
the most frequent training class of its repeat (ties go to the smallest class
index); accuracy and Macro-F1 over all predefined classes, with an undefined
class F1 counted as zero, as in the method's and the baselines' records.

Usage:
    uv run python scripts/analysis/dataset_stats.py
"""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from config import DATASETS as DATASET_INFO  # noqa: E402
from mining.io import load_hypergraphs_from_file, load_labels  # noqa: E402
from pipeline.artifacts import atomic_json_write, atomic_text_write  # noqa: E402
from pipeline.preprocessing import verify_preprocessing  # noqa: E402
from pipeline.provenance import capture_provenance  # noqa: E402
from pipeline.splits import load_and_validate_split_manifest  # noqa: E402

DATASETS = (("imdb_dir_form", "IMDB-Dir-Form", "Real"),
            ("imdb_dir_genre", "IMDB-Dir-Genre", "Real"),
            ("steam_player", "Steam-Player", "Real"),
            ("twitter_friend", "Twitter-Friend", "Real"),
            ("rhg_10", "RHG-10", "Synthetic"),
            ("rhg_3", "RHG-3", "Synthetic"))
LABEL_MODE = "composite"
SPLIT_PROTOCOL = "dhgbench_torch_2_2_2"
REPEATS = range(5)
OUTPUT = REPO / "results" / "dataset_stats"


def load_dataset(dataset):
    domain = DATASET_INFO[dataset]["domain"]
    hypergraph_file = str(REPO / "data" / dataset / LABEL_MODE / f"{domain}.txt")
    labels_file = str(REPO / "data" / dataset / "hg_label.txt")
    graphs, labels = load_hypergraphs_from_file(hypergraph_file), load_labels(labels_file)
    config = {"_dataset": dataset, "_label_mode": LABEL_MODE,
              "num_classes": DATASET_INFO[dataset]["num_classes"]}
    verify_preprocessing(graphs, labels, config, hypergraph_file, labels_file)
    return domain, hypergraph_file, labels_file, graphs, labels


def overlap_counts(graph):
    """(edge-pair incidences, those with overlap >= 2) of one hypergraph."""
    edges = [frozenset(edge.vertices) for edge in graph.hyper_edges]
    incident = {}
    for index, edge in enumerate(edges):
        for vertex in edge:
            incident.setdefault(vertex, []).append(index)
    total = large = 0
    for indices in incident.values():
        for a, b in combinations(indices, 2):
            total += 1
            large += len(edges[a] & edges[b]) >= 2
    return total, large


def graph_stats(graphs, labels):
    vertices = [len(graph.vertex_label) for graph in graphs]
    edges = [len(graph.hyper_edges) for graph in graphs]
    merged = [len({frozenset(edge.vertices) for edge in graph.hyper_edges}) for graph in graphs]
    pairs = [overlap_counts(graph) for graph in graphs]
    total = sum(count for count, _ in pairs)
    large = sum(count for _, count in pairs)
    return {
        "graphs": len(graphs),
        "avg_vertices": float(np.mean(vertices)),
        "avg_hyperedges": float(np.mean(edges)),
        "avg_hyperedges_merged": float(np.mean(merged)),
        "largest_class_share": max(Counter(labels.tolist()).values()) / len(labels) * 100,
        "edge_pair_incidences": total,
        "overlap_ge2_share": large / total * 100 if total else 0.0,
        "graphs_with_overlap_ge2_share": float(np.mean([count > 0 for _, count in pairs]) * 100),
    }


def majority_reference(labels, splits, classes):
    rows = []
    for split in splits:
        counts = Counter(labels[split.train].tolist())
        majority = min(counts, key=lambda c: (-counts[c], c))
        row = {"repeat": split.repeat, "majority_class": int(majority)}
        for name, indices in (("val", split.validation), ("test", split.test)):
            truth = labels[indices]
            predicted = np.full_like(truth, majority)
            row[f"{name}_accuracy"] = float(np.mean(predicted == truth) * 100)
            row[f"{name}_macro_f1"] = float(f1_score(
                truth, predicted, average="macro", labels=classes, zero_division=0) * 100)
        rows.append(row)
    aggregate = {}
    for key in ("val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"):
        values = [row[key] for row in rows]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
    return {"repeats": rows, "aggregate": aggregate}


def dataset_record(dataset):
    domain, hypergraph_file, labels_file, graphs, labels = load_dataset(dataset)
    split_dir = REPO / "data" / dataset / "splits" / SPLIT_PROTOCOL / LABEL_MODE
    splits = [load_and_validate_split_manifest(
        split_dir / f"repeat_{repeat}.json", dataset=dataset, label_mode=LABEL_MODE,
        hypergraph_file=hypergraph_file, labels_file=labels_file, num_graphs=len(graphs),
        expected_repeat=repeat, expected_domain=domain) for repeat in REPEATS]
    classes = list(range(DATASET_INFO[dataset]["num_classes"]))
    sizes = {(len(s.train), len(s.validation), len(s.test)) for s in splits}
    if len(sizes) != 1:
        raise ValueError(f"{dataset}: partition sizes differ between repeats")
    train, validation, test = sizes.pop()
    return {
        **graph_stats(graphs, labels),
        "classes": len(classes),
        "partition_sizes": {"train": train, "validation": validation, "test": test},
        "majority_class": majority_reference(labels, splits, classes),
        "split_manifest_hashes": {str(s.repeat): s.manifest_hash for s in splits},
    }


def markdown(records):
    lines = [
        "# Dataset statistics and majority-class reference",
        "",
        "Generated by `scripts/analysis/dataset_stats.py`; see its docstring for definitions.",
        "",
        "| Dataset | Type | Graphs | Classes | Train / val / test | Avg. |V| | Avg. |E| "
        "| Avg. |E| merged | Largest class (%) | Overlap >= 2 (%) | Graphs with overlap >= 2 (%) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, name, kind) in DATASETS:
        r = records[dataset]
        p = r["partition_sizes"]
        lines.append(
            f"| {name} | {kind} | {r['graphs']:,} | {r['classes']} "
            f"| {p['train']:,} / {p['validation']} / {p['test']} "
            f"| {r['avg_vertices']:.2f} | {r['avg_hyperedges']:.2f} | {r['avg_hyperedges_merged']:.2f} "
            f"| {r['largest_class_share']:.1f} | {r['overlap_ge2_share']:.1f} "
            f"| {r['graphs_with_overlap_ge2_share']:.1f} |")
    lines += [
        "",
        "## Majority-class reference (training majority, mean ± population s.d. over repeats 0-4)",
        "",
        "| Dataset | Test accuracy | Test Macro-F1 | Val accuracy | Val Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for (dataset, name, _) in DATASETS:
        a = records[dataset]["majority_class"]["aggregate"]
        lines.append(
            f"| {name} | {a['test_accuracy_mean']:.2f} ± {a['test_accuracy_std']:.2f} "
            f"| {a['test_macro_f1_mean']:.2f} ± {a['test_macro_f1_std']:.2f} "
            f"| {a['val_accuracy_mean']:.2f} ± {a['val_accuracy_std']:.2f} "
            f"| {a['val_macro_f1_mean']:.2f} ± {a['val_macro_f1_std']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    provenance = capture_provenance()
    records = {dataset: dataset_record(dataset) for dataset, _, _ in DATASETS}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json_write(OUTPUT / "dataset_stats.json",
                      {"schema_version": 1, "label_mode": LABEL_MODE,
                       "split_protocol": SPLIT_PROTOCOL, "provenance": provenance,
                       "datasets": records})
    atomic_text_write(OUTPUT / "dataset_stats.md", markdown(records))
    print(markdown(records))


if __name__ == "__main__":
    main()
