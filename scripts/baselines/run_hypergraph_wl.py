"""Hypergraph WL kernel baseline (Feng et al., TPAMI 2024) on the DHG-Bench partitions.

For each repeat, the kernel's colours are learned from the training graphs
only, a kernel SVM is fit on the training labels, and the validation and test
graphs are scored against the training graphs. Settings are the authors'
published defaults and are not tuned: four WL iterations, cosine
normalization and ``SVC(kernel="precomputed")`` with sklearn's defaults
(C = 1). The validation split is reported but used for no choice.

Label variants:

- ``exact``: the authors' input (duplicate hyperedges merged, exact degree and
  cardinality labels);
- ``log2``: the input the method sees (every hyperedge kept, verified stored
  degree and cardinality labels, log2-binned).

Held-out norm (``hypergraph_wl.HypergraphWLKernel``): ``seen`` reproduces the
authors' code; ``all`` is the exact kernel, a sensitivity variant whose run
IDs end in ``_NORMALL``.

Records go to ``<out-root>/<dataset>/composite/baselines/HGWL_<KERNEL>_<LABELS>[_NORMALL]/``.
Runs require committed, clean source unless ``--allow-dirty`` is given (then
the record says so). Usage:

    .venv/bin/python scripts/baselines/run_hypergraph_wl.py all
    .venv/bin/python scripts/baselines/run_hypergraph_wl.py rhg_3 --kernel subtree \
        --labels exact --repeats 0 --out-root /tmp/hgwl --allow-dirty
"""
import argparse
import csv
import io
import logging
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from config import DATASETS as DATASET_INFO  # noqa: E402
from mining.io import load_hypergraphs_from_file, load_labels  # noqa: E402
from pipeline.artifacts import (  # noqa: E402
    atomic_gzip_json_write, atomic_json_write, atomic_text_write, canonical_json_hash,
    sha256_file,
)
from pipeline.label_binning import bin_hypergraph_labels  # noqa: E402
from pipeline.preprocessing import verify_preprocessing  # noqa: E402
from pipeline.provenance import capture_provenance, require_unchanged_source  # noqa: E402
from pipeline.splits import load_and_validate_split_manifest  # noqa: E402
from scripts.baselines.hypergraph_wl import (  # noqa: E402
    HELD_OUT_NORMS, KERNELS, N_ITER, HypergraphWLKernel, feng_input, stored_label_input,
)

logger = logging.getLogger("hypergraph_wl")

MAIN_DATASETS = ("imdb_dir_form", "imdb_dir_genre", "steam_player",
                 "twitter_friend", "rhg_10", "rhg_3")
LABELS = ("exact", "log2")
LABEL_MODE = "composite"
SPLIT_PROTOCOL = "dhgbench_torch_2_2_2"
SOURCE_FILES = ("scripts/baselines/hypergraph_wl.py", "scripts/baselines/run_hypergraph_wl.py")
REFERENCE = {
    "reference": "Feng, Han, Ying, Gao. Hypergraph Isomorphism Computation. IEEE TPAMI 46(5), 2024",
    "code": "https://github.com/iMoonLab/HIC",
    "code_commit": "179e420d50e05b380072f59b0646cf7972c03b44",
}
CSV_FIELDS = ("repeat", "vocabulary_size", "val_accuracy", "val_macro_f1",
              "accuracy", "macro_f1", "kernel_fit_seconds", "kernel_transform_seconds",
              "svm_seconds", "process_peak_rss_mb")


def run_id(kernel, labels, held_out_norm):
    suffix = "" if held_out_norm == "seen" else f"_NORM{held_out_norm.upper()}"
    return f"HGWL_{kernel.upper()}_{labels.upper()}{suffix}"


def source_hashes():
    return {name: sha256_file(REPO / name) for name in SOURCE_FILES}


def load_dataset(dataset):
    """Graphs and labels with their verified preprocessing receipt."""
    domain = DATASET_INFO[dataset]["domain"]
    hypergraph_file = str(REPO / "data" / dataset / LABEL_MODE / f"{domain}.txt")
    labels_file = str(REPO / "data" / dataset / "hg_label.txt")
    graphs, labels = load_hypergraphs_from_file(hypergraph_file), load_labels(labels_file)
    config = {"_dataset": dataset, "_label_mode": LABEL_MODE,
              "num_classes": DATASET_INFO[dataset]["num_classes"]}
    receipt = verify_preprocessing(graphs, labels, config, hypergraph_file, labels_file)
    return domain, hypergraph_file, labels_file, graphs, labels, receipt


def kernel_inputs(graphs, labels):
    if labels == "exact":
        return [feng_input(graph) for graph in graphs]
    binned = bin_hypergraph_labels(graphs, {"label_binning": "log2"})
    return [stored_label_input(graph) for graph in binned]


def metrics(truth, predicted, classes):
    return (float(np.mean(predicted == truth) * 100),
            float(f1_score(truth, predicted, average="macro", labels=classes,
                           zero_division=0) * 100))


def run_repeat(inputs, labels, split, kernel_kind, held_out_norm, classes):
    """Fit on the training graphs; score validation and test against them."""
    def subset(indices):
        return [inputs[i] for i in indices], labels[indices]

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = (
        subset(split.train), subset(split.validation), subset(split.test))
    kernel = HypergraphWLKernel(kernel_kind, N_ITER, held_out_norm)
    start = time.time()
    k_train = kernel.fit_transform(train_x)
    fit_seconds = time.time() - start
    start = time.time()
    k_val, k_test = kernel.transform(val_x), kernel.transform(test_x)
    transform_seconds = time.time() - start
    start = time.time()
    svm = SVC(kernel="precomputed").fit(k_train, train_y)
    pred_val, pred_test = svm.predict(k_val), svm.predict(k_test)
    svm_seconds = time.time() - start
    val_accuracy, val_macro_f1 = metrics(val_y, pred_val, classes)
    accuracy, macro_f1 = metrics(test_y, pred_test, classes)
    record = {
        "repeat": split.repeat, "vocabulary_size": kernel.vocabulary_size,
        "val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1,
        "accuracy": accuracy, "macro_f1": macro_f1,
        "kernel_fit_seconds": fit_seconds, "kernel_transform_seconds": transform_seconds,
        "svm_seconds": svm_seconds,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    predictions = {
        name: {"graph_ids": [int(i) for i in indices], "predicted": [int(p) for p in pred],
               "truth": [int(t) for t in truth]}
        for name, indices, pred, truth in (
            ("validation", split.validation, pred_val, val_y),
            ("test", split.test, pred_test, test_y))
    }
    return record, predictions


def aggregate(records):
    result = {}
    for key in ("val_accuracy", "val_macro_f1", "accuracy", "macro_f1"):
        values = [record[key] for record in records]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    return result


def write_repeats_csv(path, records):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record[key] for key in CSV_FIELDS})
    atomic_text_write(path, output.getvalue())


def run_dataset(dataset, kernels, label_variants, norms, repeats, out_root, provenance,
                overwrite):
    domain, hypergraph_file, labels_file, graphs, labels, receipt = load_dataset(dataset)
    classes = list(range(DATASET_INFO[dataset]["num_classes"]))
    split_dir = REPO / "data" / dataset / "splits" / SPLIT_PROTOCOL / LABEL_MODE
    splits = [load_and_validate_split_manifest(
        split_dir / f"repeat_{repeat}.json", dataset=dataset, label_mode=LABEL_MODE,
        hypergraph_file=hypergraph_file, labels_file=labels_file,
        num_graphs=len(graphs), expected_repeat=repeat, expected_domain=domain,
    ) for repeat in repeats]
    for label_variant in label_variants:
        inputs = kernel_inputs(graphs, label_variant)
        for kernel_kind, held_out_norm in ((k, n) for k in kernels for n in norms):
            name = run_id(kernel_kind, label_variant, held_out_norm)
            root = Path(out_root) / dataset / LABEL_MODE / "baselines" / name
            if (root / "experiment.json").exists() and not overwrite:
                logger.info("%s %s: record exists, skipped (--overwrite to replace)",
                            dataset, name)
                continue
            started = datetime.now(timezone.utc).isoformat()
            records = []
            for split in splits:
                record, predictions = run_repeat(inputs, labels, split, kernel_kind,
                                                 held_out_norm, classes)
                atomic_gzip_json_write(
                    str(root / f"repeat_{split.repeat}" / "predictions.json.gz"), predictions)
                records.append(record)
                logger.info("%s %s r%d: test acc %.2f F1 %.2f (val %.2f / %.2f), %d colours",
                            dataset, name, split.repeat, record["accuracy"],
                            record["macro_f1"], record["val_accuracy"],
                            record["val_macro_f1"], record["vocabulary_size"])
            require_unchanged_source(provenance)
            if source_hashes() != provenance["baseline_sources"]:
                raise RuntimeError("Baseline source changed during this invocation")
            summary = aggregate(records)
            write_repeats_csv(str(root / "repeats.csv"), records)
            atomic_json_write(str(root / "experiment.json"), {
                "schema_version": 1, "run_id": name, "status": "complete",
                "baseline": "hypergraph_wl", "reference": REFERENCE,
                "dataset": dataset, "label_mode": LABEL_MODE,
                "settings": {
                    "kernel": kernel_kind, "labels": label_variant, "n_iter": N_ITER,
                    "normalize": "cosine", "held_out_norm": held_out_norm,
                    "classifier": "sklearn SVC(kernel='precomputed')",
                    "svm_C": 1.0, "merge_duplicate_hyperedges": label_variant == "exact",
                    "tuning": "none (authors' defaults); validation used for no choice",
                },
                "protocol": {
                    "split_protocol": SPLIT_PROTOCOL, "repeats": list(repeats),
                    "colours_learned_from": "train", "classifier_fit_on": "train",
                },
                "split_manifest_hashes": {str(split.repeat): split.manifest_hash
                                          for split in splits},
                "preprocessing_sha256": canonical_json_hash(receipt),
                "repeats_detail": records, "aggregate": summary,
                "provenance": provenance, "started_at": started,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            logger.info("%s %s: test Macro-F1 %.2f ± %.2f, acc %.2f ± %.2f -> %s",
                        dataset, name, summary["macro_f1_mean"], summary["macro_f1_std"],
                        summary["accuracy_mean"], summary["accuracy_std"], root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="+",
                        help=f"dataset names, or 'all' for {', '.join(MAIN_DATASETS)}")
    parser.add_argument("--kernel", nargs="+", choices=KERNELS, default=list(KERNELS))
    parser.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--held-out-norm", nargs="+", choices=HELD_OUT_NORMS,
                        default=list(HELD_OUT_NORMS))
    parser.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-root", default=str(REPO / "results"))
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permit uncommitted source (recorded in the provenance)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    datasets = list(MAIN_DATASETS) if args.datasets == ["all"] else args.datasets
    unknown = [name for name in datasets if name not in DATASET_INFO]
    if unknown:
        parser.error(f"unknown datasets: {unknown}")
    provenance = capture_provenance()
    provenance["baseline_sources"] = source_hashes()
    provenance["allow_dirty"] = args.allow_dirty
    if not args.allow_dirty and (not provenance["git_commit"] or
                                 provenance["git_source_dirty"] is not False):
        parser.error("source is not committed and clean; commit first or pass --allow-dirty")
    for dataset in datasets:
        run_dataset(dataset, args.kernel, args.labels, args.held_out_norm, args.repeats,
                    args.out_root, provenance, args.overwrite)


if __name__ == "__main__":
    main()
