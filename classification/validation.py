"""Fail-fast validation of the run settings."""

import math


def number(config, key, *, minimum=0, maximum=None, integer=False, inclusive=True):
    value = config.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or (integer and type(value) is not int) or
            (value < minimum if inclusive else value <= minimum) or
            (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {key}: {value!r}")


def validate_head(config):
    names = ("nn_layers", "nn_dropouts", "nn_l2_regs")
    if any(not isinstance(config.get(name), (list, tuple)) for name in names):
        raise ValueError("NN layers, dropouts and L2 settings must be lists")
    if not len(config["nn_layers"]) == len(config["nn_dropouts"]) == len(config["nn_l2_regs"]):
        raise ValueError("NN layers, dropouts and L2 settings must have equal lengths")
    for width, dropout, l2 in zip(*(config[name] for name in names)):
        number({"width": width}, "width", integer=True, inclusive=False)
        number({"dropout": dropout}, "dropout", maximum=1)
        if dropout == 1:
            raise ValueError("NN dropout must be less than 1")
        number({"l2": l2}, "l2")


def validate_experiment_settings(config):
    for key in ("base_sup",):
        number(config, key, maximum=1, inclusive=False)
    for key in ("sup_discount", "min_sup_floor"):
        number(config, key)
    number(config, "max_phases", integer=True)
    if config.get("max_depth") is not None:
        number(config, "max_depth", integer=True, inclusive=False)
    number(config, "num_repeats", integer=True, minimum=1, maximum=5)
    number(config, "num_classes", integer=True, minimum=2)
    if config.get("split_protocol") != "dhgbench_torch_2_2_2":
        raise ValueError("Unsupported split_protocol")
    if config.get("mined_selection") not in {None, "none", "filter"}:
        raise ValueError("mined_selection must be 'none' or 'filter'")
    if config.get("mined_selection") == "filter":
        number(config, "min_cov", integer=True, inclusive=False)
    representation = config["representation_type"]
    if representation == "supervised_sum":
        number(config, "embedding_dim", integer=True, inclusive=False)
        number(config, "supervised_pattern_dropout", maximum=1)
        if config["supervised_pattern_dropout"] == 1:
            raise ValueError("supervised_pattern_dropout must be less than 1")
        number(config, "supervised_encoder_l2")
    classifier = config.get("classifier_type")
    if classifier not in {"nn", "svm"}:
        raise ValueError(f"Unknown classifier_type: {classifier!r}")
    if classifier == "nn":
        validate_head(config)
        for key in ("nn_epochs", "nn_batch_size", "nn_patience", "nn_lr_patience"):
            number(config, key, integer=True, inclusive=False)
        for key in ("nn_learning_rate", "nn_min_lr"):
            number(config, key, inclusive=False)
        number(config, "nn_lr_decay_factor", maximum=1, inclusive=False)
        if config["nn_lr_decay_factor"] == 1:
            raise ValueError("nn_lr_decay_factor must be less than 1")
        for key in ("deterministic_training", "nn_shuffle"):
            if type(config.get(key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
    else:
        number(config, "svm_C", inclusive=False)
        number(config, "svm_max_iter", integer=True, inclusive=False)
