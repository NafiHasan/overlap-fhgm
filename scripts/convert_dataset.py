"""
Convert DHG-Bench hypergraph-classification files to the pipeline's text format.

Each vertex is labeled with its degree and each hyperedge with its cardinality.
Hypergraph order, vertex order and class labels are kept; vertex IDs become
1-based, and isolated vertices are kept with label 0.

DHG-Bench format:
    Line 1: total_hypergraphs
    Each hypergraph block:
        n_vertices n_edges class_label
        v_label_0 v_label_1 ... v_label_{n-1}
        edge_0_v1 edge_0_v2 ...
        edge_1_v1 edge_1_v2 ...
        ...

Pipeline format (per hypergraph):
    t # <global_id>
    v <v_id> <v_label>
    e <v1> <v2> ... <edge_label>

Outputs go to ``data/<dataset>/composite/<domain>.txt`` and
``data/<dataset>/hg_label.txt``, with a ``.preprocessing.json`` receipt.
Existing outputs whose receipt matches are skipped. After a conversion, the
hash-bound split manifests must be regenerated
(``tools/dhgbench_split_generator/``).

Usage:
    python scripts/convert_dataset.py --dataset rhg_3
    python scripts/convert_dataset.py --all-datasets --raw-root dhg_data/hgcls_data
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABEL_MODE
from pipeline.artifacts import sha256_file
from pipeline.preprocessing import write_conversion_receipt

# Dataset name -> (subdirectory, raw file name, domain name) under the raw root.
_DATASET_FILES = {
    'imdb_dir_form':  ('IMDB', 'IMDB_dir_form',  'IMDB_dir_form'),
    'imdb_dir_genre': ('IMDB', 'IMDB_dir_genre', 'IMDB_dir_genre'),
    'steam_player':   ('STEAM', 'stream_player',  'steam_player'),
    'twitter_friend': ('TWITTER', 'twitter_friend', 'twitter_friend'),
    'rhg_10':         ('RHG', 'RHG_10', 'rhg_10'),
    'rhg_3':          ('RHG', 'RHG_3', 'rhg_3'),
}


def parse_dhgbench_file(filepath):
    """
    Parse a DHG-Bench hypergraph-classification file.

    Returns:
        list of dicts, each with keys:
            - 'n_vertices': int
            - 'n_edges': int
            - 'label': int (class label)
            - 'vertex_labels': list[int]
            - 'edges': list[list[int]]  (each edge is a list of vertex indices)
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines()]

    if not lines:
        raise ValueError("Empty DHG-Bench file")
    total_hg = int(lines[0])
    if total_hg <= 0:
        raise ValueError("Graph count must be positive")
    hypergraphs = []
    idx = 1

    for graph_id in range(total_hg):
        if idx + 1 >= len(lines):
            raise ValueError(f"Truncated DHG-Bench input at graph {graph_id}")
        header = lines[idx].split()
        if len(header) != 3:
            raise ValueError(f"Invalid header at graph {graph_id}")
        n_v, n_e, label = int(header[0]), int(header[1]), int(header[2])
        if n_v < 0 or n_e < 0 or label < 0 or idx + 2 + n_e > len(lines):
            raise ValueError(f"Invalid sizes/label or truncated edges at graph {graph_id}")

        vertex_labels = [int(x) for x in lines[idx + 1].split()]
        if len(vertex_labels) != n_v:
            raise ValueError(f"Vertex-label count mismatch at graph {graph_id}")

        edges = []
        for e in range(n_e):
            edge_vertices = [int(x) for x in lines[idx + 2 + e].split()]
            if (not edge_vertices or len(set(edge_vertices)) != len(edge_vertices) or
                    any(vertex < 0 or vertex >= n_v for vertex in edge_vertices)):
                raise ValueError(f"Invalid hyperedge {e} at graph {graph_id}")
            edges.append(edge_vertices)

        hypergraphs.append({
            'n_vertices': n_v,
            'n_edges': n_e,
            'label': label,
            'vertex_labels': vertex_labels,
            'edges': edges,
        })
        idx += 2 + n_e

    if any(line for line in lines[idx:]):
        raise ValueError("Unexpected trailing records after declared graph count")
    return hypergraphs


def compute_degree_labels(hypergraphs):
    """Replace vertex labels with vertex degrees (isolated vertices get 0)."""
    for hg in hypergraphs:
        degrees = [0] * hg['n_vertices']
        for edge in hg['edges']:
            for v in edge:
                degrees[v] += 1
        hg['vertex_labels'] = degrees
    return hypergraphs


def write_pipeline_format(hypergraphs, output_data_path, output_labels_path, *,
                          source_file=None):
    """
    Write hypergraphs in the pipeline's text format, with cardinality edge labels.

    All hypergraphs go into a single file and the class labels into a
    separate file. With ``source_file``, a conversion receipt is written next
    to the output.
    """
    source_hash = sha256_file(source_file) if source_file is not None else None
    os.makedirs(os.path.dirname(output_data_path) or '.', exist_ok=True)

    with open(output_data_path, 'w') as f:
        for i, hg in enumerate(hypergraphs):
            f.write(f"t # {i}\n")
            for v_id in range(hg['n_vertices']):
                f.write(f"v {v_id + 1} {hg['vertex_labels'][v_id]}\n")
            for edge in hg['edges']:
                edge_str = ' '.join(str(v + 1) for v in edge)
                f.write(f"e {edge_str} {len(edge)}\n")

    os.makedirs(os.path.dirname(output_labels_path) or '.', exist_ok=True)
    with open(output_labels_path, 'w') as f:
        for hg in hypergraphs:
            f.write(f"{hg['label']}\n")

    print(f"Wrote {len(hypergraphs)} hypergraphs to {output_data_path}")
    print(f"Wrote {len(hypergraphs)} labels to {output_labels_path}")
    if source_file is not None:
        write_conversion_receipt(
            output_data_path, output_labels_path, source_file=source_file,
            source_hash=source_hash, source_kind="dhgbench_raw", converter_file=__file__,
            label_mode=LABEL_MODE, vertex_dims=1, edge_dims=1,
            graphs=len(hypergraphs), isolate_policy="preserve",
        )


def output_paths(dataset):
    _, _, domain = _DATASET_FILES[dataset]
    return (os.path.join("data", dataset, LABEL_MODE, domain + ".txt"),
            os.path.join("data", dataset, "hg_label.txt"))


def matching_output(path, labels_path):
    """Return whether an existing converted file and its receipt match."""
    receipt_path = path + ".preprocessing.json"
    if (not os.path.isfile(path) or not os.path.isfile(labels_path)
            or not os.path.isfile(receipt_path)):
        return False
    try:
        with open(receipt_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    expected = {"schema_version": 1, "label_mode": LABEL_MODE, "vertex_composite_dims": 1,
                "edge_composite_dims": 1, "hypergraph_sha256": sha256_file(path)}
    if not all(receipt.get(key) == value for key, value in expected.items()):
        return False
    return receipt.get("labels_sha256") in (None, sha256_file(labels_path))


def convert(dataset, raw_root, *, overwrite=False):
    subdir, file_basename, _ = _DATASET_FILES[dataset]
    output_data, output_labels = output_paths(dataset)
    if matching_output(output_data, output_labels):
        print(f"SKIP: {dataset} is already converted ({output_data})")
        return "skipped"
    if os.path.exists(output_data) and not overwrite:
        raise FileExistsError(
            f"{output_data} exists but its receipt does not match; re-run with "
            "--overwrite, then regenerate the split manifests"
        )
    raw_input = os.path.join(raw_root, subdir, file_basename + ".txt")
    print(f"{dataset}: loading {raw_input}")
    hypergraphs = compute_degree_labels(parse_dhgbench_file(raw_input))
    write_pipeline_format(hypergraphs, output_data, output_labels, source_file=raw_input)
    return "converted"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert DHG-Bench datasets to the pipeline format")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--dataset', choices=list(_DATASET_FILES))
    selection.add_argument('--all-datasets', action='store_true')
    parser.add_argument('--raw-root', default='dhg_data/hgcls_data',
                        help='Directory holding the DHG-Bench IMDB, STEAM, TWITTER and RHG folders')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing output whose receipt does not match')
    args = parser.parse_args(argv)
    datasets = list(_DATASET_FILES) if args.all_datasets else [args.dataset]
    results = [convert(dataset, args.raw_root, overwrite=args.overwrite) for dataset in datasets]
    if "converted" in results:
        print("Converted inputs changed. Regenerate their hash-bound DHG-Bench "
              "split manifests before running the pipeline.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
