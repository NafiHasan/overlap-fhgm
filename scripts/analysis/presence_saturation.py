"""Presence saturation of the mined vocabulary.

For every dataset and repeat of a run, reads the run's frozen vocabulary and
its binary coverage (which hypergraphs contain each pattern) from the mining
cache named in the run's ``repeats.csv``, and reports how close presence
features are to constant:

- the mean share of the vocabulary present in a hypergraph, over training and
  over test hypergraphs, overall and for patterns of one, two and three
  incidences;
- the share of patterns present in at least 90% of the training hypergraphs.

Pattern size is the number of incidences (extensions) of a pattern. Nothing is
fitted and no class labels are read. The caches hold the coverage files, so
run this where the caches of the run exist.

Writes ``results/presence_saturation/presence_saturation.{json,md}``.

Usage:
    uv run python scripts/analysis/presence_saturation.py --run RUN
    uv run python scripts/analysis/presence_saturation.py --run RUN \
        --run-root results/sanity/shuffled_labels/{dataset}
"""

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from dataset_stats import DATASETS, LABEL_MODE, REPEATS, SPLIT_PROTOCOL, load_dataset  # noqa: E402
from mining.io import load_coverage_file  # noqa: E402
from mining.pattern_io import load_pattern_vocabulary  # noqa: E402
from pipeline.artifacts import atomic_json_write, atomic_text_write  # noqa: E402
from pipeline.provenance import capture_provenance  # noqa: E402
from pipeline.splits import load_and_validate_split_manifest  # noqa: E402

OUTPUT = REPO / "results" / "presence_saturation"
SATURATED = 0.9
SIZES = (1, 2, 3)


def repeat_record(cache_dir, split, expected_patterns):
    patterns, _ = load_pattern_vocabulary(cache_dir / "patterns.json")
    if len(patterns) != expected_patterns:
        raise ValueError(f"{cache_dir}: {len(patterns)} patterns, run recorded {expected_patterns}")
    coverage = load_coverage_file(str(cache_dir / "cover_freq.txt"))
    sizes = [len(code.get_extensions()) for code in patterns]
    graphs = {pattern: set(coverage.get(pattern, {})) for pattern in range(len(patterns))}
    train, test = set(map(int, split.train)), set(map(int, split.test))
    record = {"repeat": split.repeat, "patterns": len(patterns),
              "patterns_by_size": {str(s): sizes.count(s) for s in SIZES}}
    for name, members in (("train", train), ("test", test)):
        present = Counter()
        present_by_size = {s: Counter() for s in SIZES}
        for pattern, ids in graphs.items():
            for graph in ids & members:
                present[graph] += 1
                present_by_size[sizes[pattern]][graph] += 1
        record[f"{name}_present_share"] = statistics.mean(
            present[g] / len(patterns) for g in members) * 100
        for s in SIZES:
            total = sizes.count(s)
            record[f"{name}_present_share_size{s}"] = (statistics.mean(
                present_by_size[s][g] / total for g in members) * 100) if total else None
    saturated = [len(ids & train) >= SATURATED * len(train) for ids in graphs.values()]
    record["saturated_share"] = sum(saturated) / len(patterns) * 100
    for s in SIZES:
        flags = [flag for flag, size in zip(saturated, sizes) if size == s]
        record[f"saturated_share_size{s}"] = sum(flags) / len(flags) * 100 if flags else None
    return record


def dataset_record(dataset, run, run_root):
    domain, hypergraph_file, labels_file, graphs, _ = load_dataset(dataset)
    split_dir = REPO / "data" / dataset / "splits" / SPLIT_PROTOCOL / LABEL_MODE
    with open(run_root / "inductive" / run / "repeats.csv", newline="") as handle:
        rows = {int(row["repeat"]): row for row in csv.DictReader(handle)}
    cache_root = REPO / "results" / dataset / "composite" / "inductive_mining_cache"
    repeats = []
    for repeat in REPEATS:
        split = load_and_validate_split_manifest(
            split_dir / f"repeat_{repeat}.json", dataset=dataset, label_mode=LABEL_MODE,
            hypergraph_file=hypergraph_file, labels_file=labels_file, num_graphs=len(graphs),
            expected_repeat=repeat, expected_domain=domain)
        row = rows[repeat]
        repeats.append({**repeat_record(cache_root / row["mining_hash"], split, int(row["patterns"])),
                        "mining_hash": row["mining_hash"]})
    aggregate = {}
    for key, value in repeats[0].items():
        if isinstance(value, float):
            data = [r[key] for r in repeats if r[key] is not None]
            aggregate[f"{key}_mean"] = statistics.mean(data)
            aggregate[f"{key}_std"] = statistics.pstdev(data)
    return {"run": run, "repeats": repeats, "aggregate": aggregate}


def markdown(records, run):
    def cell(a, key):
        return f"{a[key + '_mean']:.1f}" if key + "_mean" in a else "—"
    lines = [
        "# Presence saturation of the mined vocabulary",
        "",
        f"Generated by `scripts/analysis/presence_saturation.py` from run `{run}`; "
        "means over repeats 0-4 (population s.d. in the JSON).",
        "",
        "Present share: mean over hypergraphs of the share of vocabulary patterns they contain (%). "
        f"Saturated: share of patterns present in at least {SATURATED:.0%} of the training hypergraphs (%).",
        "",
        "| Dataset | Present, train | Present, test | Size 1 | Size 2 | Size 3 "
        "| Saturated | Saturated, size 1 | Size 2 | Size 3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, name, _ in DATASETS:
        a = records[dataset]["aggregate"]
        lines.append(
            f"| {name} | {cell(a, 'train_present_share')} | {cell(a, 'test_present_share')} "
            + " ".join(f"| {cell(a, f'train_present_share_size{s}')}" for s in SIZES)
            + f" | {cell(a, 'saturated_share')} "
            + " ".join(f"| {cell(a, f'saturated_share_size{s}')}" for s in SIZES) + " |")
    lines.append("")
    lines.append("Size columns under Present are over training hypergraphs.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run ID of a completed inductive run")
    parser.add_argument("--run-root", default="results/{dataset}/composite",
                        help="directory holding inductive/<run>/; {dataset} is substituted")
    args = parser.parse_args()
    provenance = capture_provenance()
    records = {}
    for dataset, _, _ in DATASETS:
        run_root = REPO / args.run_root.format(dataset=dataset)
        records[dataset] = dataset_record(dataset, args.run, run_root)
        print(dataset, "done", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json_write(OUTPUT / "presence_saturation.json",
                      {"schema_version": 1, "run": args.run, "saturation_threshold": SATURATED,
                       "provenance": provenance, "datasets": records})
    atomic_text_write(OUTPUT / "presence_saturation.md", markdown(records, args.run))
    print(markdown(records, args.run))


if __name__ == "__main__":
    main()
