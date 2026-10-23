"""Named settings profile of the method; change the name when changing the settings."""

import copy

FINAL_PROFILE_NAME = "FHGM_OVERLAP_COUNTS_V1"

# The method: overlap-aware mining with log2-binned degree, cardinality and
# overlap labels; every pattern of at most two incidences found in two or more
# training graphs, plus frequent patterns of up to three incidences mined by the
# phase schedule from support 0.1; per-skeleton relative occurrence frequencies
# as the encoder input.
FINAL_SETTINGS = {
    "base_sup": 0.1, "sup_discount": 0.02, "max_phases": 10,
    "min_sup_floor": 0.001, "min_coverage": 1, "max_depth": 3,
    "label_binning": "log2", "overlap_labels": "log2",
    "exhaustive_depth": 2, "exhaustive_min_graphs": 2,
    "mined_count_encoding": "relative",
    "supervised_input_scaling": "train_maxabs", "supervised_count_feature": "nonzero",
    "representation_type": "supervised_sum", "classifier_type": "nn",
    "min_cov": 15, "embedding_dim": 32,
    "supervised_pattern_dropout": 0.3, "supervised_encoder_l2": 0.05,
    "nn_layers": [64, 32], "nn_dropouts": [0.3, 0.2],
    "nn_l2_regs": [0.05, 0.03], "nn_learning_rate": 0.0003,
    "nn_epochs": 1000, "nn_batch_size": 16, "nn_patience": 20,
    "nn_lr_decay_factor": 0.5, "nn_lr_patience": 5, "nn_min_lr": 1e-5,
    "deterministic_training": True, "nn_shuffle": True,
    "svm_C": 1.0, "svm_max_iter": 5000,
    "pipeline_mode": "fully_inductive", "vocabulary_access": "train_only",
    "split_protocol": "dhgbench_torch_2_2_2",
    "train_ratio": 0.8, "validation_ratio": 0.1, "num_repeats": 5,
    "require_clean_source": True, "experiment_profile": FINAL_PROFILE_NAME,
}


def final_dataset_config(dataset, run_id):
    """Final settings for one registered dataset, with repo-relative paths."""
    from config import dataset_config

    config = dataset_config(dataset, run_id)
    config.update(copy.deepcopy(FINAL_SETTINGS))
    return config
