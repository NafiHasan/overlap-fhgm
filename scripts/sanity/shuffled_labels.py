"""Shuffled-label sanity check for the final profile.

Reruns the main run's recorded configuration, with the class labels of the
training hypergraphs permuted among themselves in every repeat. Validation
and test labels are unchanged, and scores are computed against them. Mining
never reads class labels, so the training-only vocabularies are the ones the
main configuration gives: valid caches in the usual cache directory are
reused, and missing ones are mined and cached there.

If no information about held-out labels reaches the classifier, test scores
should fall to the level of a classifier that ignores its input (compare with
the majority-class reference in ``results/dataset_stats/``).

Records go to ``results/sanity/shuffled_labels/<dataset>/inductive/<run_id>/``,
apart from the method's runs.

Usage:
    uv run python scripts/sanity/shuffled_labels.py [--dataset NAME] [--repeat R]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.logger import setup_logging  # noqa: E402

setup_logging("sanity_shuffled_labels")

import pipeline.inductive as inductive  # noqa: E402

DATASETS = ("imdb_dir_form", "imdb_dir_genre", "steam_player",
            "twitter_friend", "rhg_10", "rhg_3")
MAIN_RUN = "MAIN"
RUN_ID = "SANITY_SHUFFLED_LABELS"
PERMUTATION_SEED = 1000  # repeat r uses seed 1000 + r
OUTPUT = REPO / "results" / "sanity" / "shuffled_labels"

_original_run_single_split = inductive._run_single_split


def _shuffled_run_single_split(train_index, test_index, codes, labels, config,
                               paths, **kwargs):
    """Permute the training labels, then run the unchanged evaluation."""
    rng = np.random.default_rng(PERMUTATION_SEED + int(config["random_state"]))
    train_index = np.asarray(train_index)
    shuffled = np.array(labels, copy=True)
    shuffled[train_index] = rng.permutation(shuffled[train_index])
    return _original_run_single_split(train_index, test_index, codes, shuffled,
                                      config, paths, **kwargs)


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


def run(dataset, repeat=None):
    config = main_run_config(dataset)
    config["run_id"] = RUN_ID
    root = OUTPUT / dataset
    root.mkdir(parents=True, exist_ok=True)
    config["paths"]["coverage_file"] = str(root / "cover_freq.txt")
    config["sanity_check"] = {"training_labels": "permuted",
                              "permutation_seed": f"{PERMUTATION_SEED} + repeat"}
    inductive._run_single_split = _shuffled_run_single_split
    try:
        return inductive.run_fully_inductive_pipeline(config, repeat=repeat)
    finally:
        inductive._run_single_split = _original_run_single_split


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    for dataset in [args.dataset] if args.dataset else DATASETS:
        result = run(dataset, args.repeat)
        print(dataset, json.dumps(result.get("aggregate", {}) if isinstance(result, dict) else {},
                                  indent=1))


if __name__ == "__main__":
    main()
