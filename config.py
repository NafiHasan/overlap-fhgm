"""
Datasets and file locations.

Directory layout:
    data/<dataset>/composite/<domain>.txt    hypergraphs; vertex label = degree,
                                             hyperedge label = cardinality
    data/<dataset>/hg_label.txt              class labels
    data/<dataset>/splits/dhgbench_torch_2_2_2/composite/repeat_<r>.json
                                             DHG-Bench partitions
    results/<dataset>/composite/             outputs

The method's settings are in ``pipeline/profiles.py``.
"""

import os

DATASETS = {
    "imdb_dir_form": {"domain": "IMDB_dir_form", "num_classes": 3},
    "imdb_dir_genre": {"domain": "IMDB_dir_genre", "num_classes": 3},
    "steam_player": {"domain": "steam_player", "num_classes": 2},
    "twitter_friend": {"domain": "twitter_friend", "num_classes": 2},
    "rhg_10": {"domain": "rhg_10", "num_classes": 10},
    "rhg_3": {"domain": "rhg_3", "num_classes": 3},
}

LABEL_MODE = "composite"
SPLIT_PROTOCOL = "dhgbench_torch_2_2_2"


def dataset_config(dataset, run_id):
    """Dataset identity and repository-relative paths for one run."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}")
    results_dir = os.path.join("results", dataset, LABEL_MODE)
    return {
        "run_id": run_id,
        "_dataset": dataset,
        "_label_mode": LABEL_MODE,
        "domains": [DATASETS[dataset]["domain"]],
        "num_classes": DATASETS[dataset]["num_classes"],
        "paths": {
            "hypergraph_data_dir": os.path.join("data", dataset, LABEL_MODE),
            "labels_file": os.path.join("data", dataset, "hg_label.txt"),
            "coverage_file": os.path.join(results_dir, "cover_freq.txt"),
        },
        "split_manifest_dir": os.path.join("data", dataset, "splits", SPLIT_PROTOCOL, LABEL_MODE),
        "inductive_mining_cache_dir": os.path.join(results_dir, "inductive_mining_cache"),
        "transductive_mining_cache_dir": os.path.join(results_dir, "transductive_mining_cache"),
        "random_state": 0,
    }
