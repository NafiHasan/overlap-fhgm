"""Generate literal DHG-Bench graph-classification split manifests.

Run this only in this subproject's Python 3.11 / torch 2.2.2 environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

DATASET_ORDER = (
    "imdb_dir_form",
    "imdb_dir_genre",
    "steam_player",
    "twitter_friend",
    "rhg_10",
    "rhg_3",
)
SUPPORTED = set(DATASET_ORDER)
DOMAINS = {"imdb_dir_form": "IMDB_dir_form", "imdb_dir_genre": "IMDB_dir_genre",
           "steam_player": "steam_player", "twitter_friend": "twitter_friend",
           "rhg_10": "rhg_10", "rhg_3": "rhg_3"}
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_MODE = "composite"
DEFAULT_REPEATS = tuple(range(5))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def count_graphs(path):
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("t"))


def generate(repo_root, dataset, label_mode, repeats, output_dir=None, overwrite=False):
    if dataset not in SUPPORTED:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {sorted(SUPPORTED)}")
    if torch.__version__.split("+")[0] != "2.2.2":
        raise RuntimeError(f"Requires torch==2.2.2, found {torch.__version__}")
    root = Path(repo_root)
    # Dataset directory names are lowercase, while the two converted IMDB
    # graph files keep their uppercase ``IMDB_`` domain names.
    hypergraphs = root / "data" / dataset / label_mode / f"{DOMAINS[dataset]}.txt"
    labels = root / "data" / dataset / "hg_label.txt"
    total = count_graphs(hypergraphs)
    if total != len(labels.read_text(encoding="utf-8").splitlines()):
        raise ValueError("Hypergraph and label counts differ")
    train_size, val_size = int(.8 * total), int(.1 * total)
    output_root = Path(output_dir) if output_dir else root / "data" / dataset / "splits" / "dhgbench_torch_2_2_2" / label_mode
    destinations = []
    for repeat in repeats:
        train, validation, test = torch.utils.data.random_split(
            range(total), [train_size, val_size, total - train_size - val_size],
            generator=torch.Generator(device="cpu").manual_seed(42 + repeat))
        indices = {"train": list(train.indices), "validation": list(validation.indices), "test": list(test.indices)}
        manifest = {"schema_version": 1, "protocol": "dhgbench_graph_classification", "dataset": dataset,
                    "label_mode": label_mode, "domain": DOMAINS[dataset], "repeat": repeat, "model_seed": repeat,
                    "split_seed": 42 + repeat, "ratios": {"train": .8, "validation": .1},
                    "sizes": {"total": total, "train": train_size, "validation": val_size, "test": total-train_size-val_size},
                    "generator": {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", "torch": "2.2.2", "device": "cpu"},
                    "source": {"hypergraph_sha256": sha256(hypergraphs), "labels_sha256": sha256(labels)},
                    "indices": indices, "indices_sha256": canonical_hash(indices)}
        destination = output_root / f"repeat_{repeat}.json"
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        destinations.append(destination)

    return {
        "destinations": destinations,
        "output_root": output_root,
        "total_graphs": total,
        "train_size": train_size,
        "validation_size": val_size,
        "test_size": total - train_size - val_size,
    }


def format_completion_message(dataset, label_mode, repeats, summary):
    """Build the concise terminal confirmation printed only after all writes."""
    repeat_text = ", ".join(map(str, repeats))
    return (
        f"SPLIT GENERATION COMPLETE: {len(summary['destinations'])} manifest(s) "
        f"for {dataset}/{label_mode} (repeats: {repeat_text})\n"
        f"Split sizes: train={summary['train_size']}, "
        f"validation={summary['validation_size']}, test={summary['test_size']} "
        f"of {summary['total_graphs']} graphs\n"
        f"Output directory: {summary['output_root'].resolve()}"
    )


def build_parser():
    """Create the CLI parser with the project's standard split defaults."""
    parser = argparse.ArgumentParser(
        description="Generate exact DHG-Bench graph-classification split manifests"
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help=f"repository root (default: {DEFAULT_REPO_ROOT})",
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", choices=DATASET_ORDER)
    dataset_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="generate manifests for all six supported datasets",
    )
    parser.add_argument(
        "--label-mode",
        default=DEFAULT_LABEL_MODE,
        help=f"vertex label mode (default: {DEFAULT_LABEL_MODE})",
    )
    parser.add_argument(
        "--repeats",
        nargs="+",
        type=int,
        default=list(DEFAULT_REPEATS),
        help="repeat IDs (default: 0 1 2 3 4)",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if any(repeat not in range(5) for repeat in args.repeats):
        parser.error("--repeats must contain values in 0..4")
    if len(set(args.repeats)) != len(args.repeats):
        parser.error("--repeats must not contain duplicate values")
    if args.all_datasets and args.output_dir:
        parser.error("--output-dir cannot be combined with --all-datasets")
    datasets = DATASET_ORDER if args.all_datasets else (args.dataset,)
    manifest_count = 0
    for dataset in datasets:
        print(
            f"Generating DHG-Bench splits for {dataset}/{args.label_mode}: "
            f"repeats {', '.join(map(str, args.repeats))}...",
            flush=True,
        )
        summary = generate(
            repo_root=args.repo_root,
            dataset=dataset,
            label_mode=args.label_mode,
            repeats=args.repeats,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        manifest_count += len(summary["destinations"])
        print(format_completion_message(
            dataset, args.label_mode, args.repeats, summary
        ))
    if args.all_datasets:
        print(
            f"ALL SPLIT GENERATION COMPLETE: {len(datasets)} datasets, "
            f"{manifest_count} manifests"
        )
