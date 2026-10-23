"""Training-seed variance for the final profile.

Reruns the main run's recorded configuration with other classifier seeds,
keeping everything else fixed: the persisted partitions, the training-only
vocabularies (valid caches in the main run's cache directory are reused), and
all settings. The classifier seed controls weight initialization and the
shuffle order of training batches. The main run uses seed ``r`` in repeat
``r``; each extra run here uses seed ``offset + r``.

Comparing the spread across seeds within a repeat with the spread across
repeats separates initialization variance from partition variance.

Records go to ``results/sanity/training_seeds/<dataset>/inductive/<run_id>/``,
apart from the method's runs; ``--report`` writes the summary to
``results/sanity/training_seeds/summary.{json,md}``.

Usage:
    uv run python scripts/sanity/training_seeds.py [--dataset NAME] [--offset N] [--repeat R]
    uv run python scripts/sanity/training_seeds.py --report
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.logger import setup_logging  # noqa: E402

setup_logging("sanity_training_seeds")

import pipeline.inductive as inductive  # noqa: E402

DATASETS = ("imdb_dir_form", "imdb_dir_genre", "steam_player",
            "twitter_friend", "rhg_10", "rhg_3")
MAIN_RUN = "MAIN"
SEED_OFFSETS = (100, 200, 300, 400)  # with the main run: five seeds per repeat
OUTPUT = REPO / "results" / "sanity" / "training_seeds"
METRICS = ("macro_f1", "accuracy", "val_macro_f1", "val_accuracy")

_original_run_single_split = inductive._run_single_split


def run_id(offset):
    return f"SANITY_SEED_{offset}"


def _seeded(offset):
    def run_single_split(train_index, test_index, codes, labels, config, paths,
                         **kwargs):
        """Shift the classifier seed, then run the unchanged evaluation."""
        config = dict(config, random_state=offset + int(config["random_state"]))
        return _original_run_single_split(train_index, test_index, codes, labels,
                                          config, paths, **kwargs)
    return run_single_split


def _remap(value, old_root):
    if isinstance(value, dict):
        return {key: _remap(item, old_root) for key, item in value.items()}
    if isinstance(value, str) and value.startswith(old_root):
        return str(REPO) + value[len(old_root):]
    return value


def main_run_config(dataset):
    """The main run's recorded config, with its paths moved to this checkout."""
    record = REPO / "results" / dataset / "composite" / "inductive" / MAIN_RUN / "experiment.json"
    config = json.loads(record.read_text())["config"]
    old_root = config["paths"]["labels_file"].rsplit("/data/", 1)[0]
    return _remap(config, old_root)


def run(dataset, offset, repeat=None):
    config = main_run_config(dataset)
    main_results = Path(config["paths"]["coverage_file"]).parent
    config["run_id"] = run_id(offset)
    config["inductive_mining_cache_dir"] = str(main_results / "inductive_mining_cache")
    root = OUTPUT / dataset
    root.mkdir(parents=True, exist_ok=True)
    config["paths"]["coverage_file"] = str(root / "cover_freq.txt")
    config["sanity_check"] = {"classifier_seed": f"{offset} + repeat",
                              "main_run": MAIN_RUN}
    inductive._run_single_split = _seeded(offset)
    try:
        return inductive.run_fully_inductive_pipeline(config, repeat=repeat)
    finally:
        inductive._run_single_split = _original_run_single_split


def _repeats(path):
    with open(path, newline="") as handle:
        return {int(row["repeat"]): {metric: float(row[metric]) for metric in METRICS}
                for row in csv.DictReader(handle)}


def _complete(run):
    """Complete per the compact record (completion.json is not tracked in git)."""
    record = run / "experiment.json"
    return record.is_file() and json.loads(record.read_text()).get("status") == "complete"


def report():
    """Within-repeat (seed) and between-repeat (partition) spread per dataset."""
    summary = {}
    for dataset in DATASETS:
        runs = [REPO / "results" / dataset / "composite" / "inductive" / MAIN_RUN]
        runs += [OUTPUT / dataset / "inductive" / run_id(offset) for offset in SEED_OFFSETS]
        present = [path for path in runs if _complete(path)]
        if len(present) < 2:
            continue
        tables = [_repeats(path / "repeats.csv") for path in present]
        repeats = sorted(set.intersection(*(set(table) for table in tables)))
        entry = {"seeds": len(present), "repeats": repeats}
        for metric in METRICS:
            scores = np.array([[table[r][metric] for r in repeats] for table in tables])
            entry[metric] = {
                "mean": float(scores.mean()),
                # population SDs, as in the run reports
                "seed_sd_within_repeat": float(scores.std(axis=0).mean()),
                "repeat_sd_of_seed_means": float(scores.mean(axis=0).std()),
                "main_run_repeat_sd": float(scores[0].std()),
                "seed_range_of_dataset_mean": float(np.ptp(scores.mean(axis=1))),
            }
        summary[dataset] = entry
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = ["# Training-seed variance", "",
             f"Main run `{MAIN_RUN}` plus seed offsets {', '.join(map(str, SEED_OFFSETS))}; "
             "population SDs in points.", "",
             "| Dataset | Seeds | Metric | Mean | Seed SD (within repeat) | "
             "Repeat SD (seed means) | Main-run repeat SD | Range of dataset mean over seeds |",
             "|---|---|---|---|---|---|---|---|"]
    for dataset, entry in summary.items():
        for metric in METRICS:
            values = entry[metric]
            lines.append(
                f"| {dataset} | {entry['seeds']} | {metric} | {values['mean']:.2f} | "
                f"{values['seed_sd_within_repeat']:.2f} | {values['repeat_sd_of_seed_means']:.2f} | "
                f"{values['main_run_repeat_sd']:.2f} | {values['seed_range_of_dataset_mean']:.2f} |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--offset", type=int, choices=SEED_OFFSETS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(json.dumps(report(), indent=1))
        return
    for dataset in [args.dataset] if args.dataset else DATASETS:
        for offset in [args.offset] if args.offset else SEED_OFFSETS:
            result = run(dataset, offset, args.repeat)
            print(dataset, offset,
                  json.dumps(result.get("aggregate", {}) if isinstance(result, dict) else {},
                             indent=1))


if __name__ == "__main__":
    main()
