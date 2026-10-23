"""
Classification of one train/validation/test split.

Encodes the pattern counts, optionally selects patterns, trains the classifier
and reports accuracy and Macro-F1 for the manifest-driven pipelines.
"""

import os
import time
import logging
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import f1_score

from .feature_selection import SelectedCodes, filter_codes
from .pattern_counts import encode_pattern_counts, resolve_count_options
from .classifiers import (
    NN,
    SupervisedPatternSetNN,
    LinearSVM,
    plot_training_history,
)

logger = logging.getLogger(__name__)

_REPRESENTATION_TYPES = {"raw", "supervised_sum"}


@dataclass
class SplitEvaluationResult:
    accuracy: float
    macro_f1: float
    filtering_time: float
    embedding_time: float
    classifier_time: float
    selected_pattern_count: int
    raw_zero_vector_rates: dict
    total_time: float = 0.0
    epochs_trained: int = 0
    best_epoch: int = 0
    best_validation_accuracy: float = 0.0
    val_accuracy: float = 0.0
    val_macro_f1: float = 0.0
    selected_pattern_ids: list = field(default_factory=list)
    predictions: dict = field(default_factory=dict)
    trainable_parameters: int | None = None
    training_history: dict = field(default_factory=dict)


def resolve_representation_type(config):
    """Resolve and validate the graph representation.

    ``supervised_sum`` is the learned pattern encoder with its network head;
    ``raw`` feeds the scaled relative frequencies straight to the head
    (``classifier_type`` "nn") or to a linear SVM ("svm").
    """
    representation_type = config.get("representation_type", "supervised_sum")
    if representation_type not in _REPRESENTATION_TYPES:
        choices = ", ".join(sorted(_REPRESENTATION_TYPES))
        raise ValueError(
            f"Unknown representation_type {representation_type!r}. Use {choices}."
        )
    return representation_type


def _run_single_split(train_index, test_index, codes, labels, config, paths,
                      validation_index, pattern_counts, fold_label="", plot_dir=None):
    """
    Encode the pattern counts, train the classifier and evaluate one split.

    Args:
        train_index: Array of training sample indices. All learned preprocessing
            is fitted exclusively on this partition.
        test_index: Array of test sample indices.
        codes: Binary pattern coverage {pattern_id: {graph_id: 1}}; read only
            by the optional pattern selection.
        labels: Ground-truth label array.
        config: Classification configuration dict.
        paths: Paths configuration dict.
        validation_index: Validation sample indices.
        pattern_counts: ``(counts, totals, depths)`` of the frozen vocabulary
            (``classification.pattern_counts.pattern_count_matrix``).
        fold_label: Label for logging (e.g., "Repeat 0").
        plot_dir: Directory to save training plots. If None, no plot is saved.

    Returns:
        SplitEvaluationResult.
    """
    split_start = time.time()
    total_graphs = len(labels)
    train_index = np.asarray(train_index)
    test_index = np.asarray(test_index)
    validation_index = np.asarray(validation_index)

    split_size = len(train_index) + len(validation_index) + len(test_index)
    combined = np.concatenate([train_index, validation_index, test_index])
    if len(np.unique(combined)) != split_size:
        raise ValueError("Train, validation, and test indices must be disjoint")

    if fold_label:
        logger.info(f"{fold_label}")
    logger.info(
        f"  Model split: {len(train_index)} train / "
        f"{len(validation_index)} val / {len(test_index)} test"
    )

    # ----- Pattern counts and optional selection -----
    filter_start = time.time()
    mined_selection = config.get("mined_selection") or "none"
    if mined_selection not in {"none", "filter"}:
        raise ValueError("mined_selection must be 'none' or 'filter'")
    # Every mined pattern seen in enough training graphs enters the encoder.
    count_encoding, min_depth, min_train_graphs = resolve_count_options(config)
    counts, totals, depths = pattern_counts
    if counts.shape[0] != total_graphs:
        raise ValueError("pattern_counts row count must equal the graph count")
    count_values, count_columns = encode_pattern_counts(
        counts, totals, depths, train_index, encoding=count_encoding,
        min_train_graphs=min_train_graphs, min_depth=min_depth)
    if mined_selection == "filter":
        # Selection ablation: the entropy-based ranking and min_cov coverage
        # selection of feature_selection.py, on training labels, choose which
        # count columns enter the encoder.
        selected = filter_codes(
            {int(column): codes.get(int(column), {}) for column in count_columns},
            labels[train_index],
            train_index,
            total_graphs,
            min_cov=config.get("min_cov", 15),
        )
        keep = np.isin(count_columns, selected.pattern_ids)
        count_values, count_columns = count_values[:, keep], count_columns[keep]
    t_codes = SelectedCodes()
    t_codes.pattern_ids = count_columns.tolist()
    patterns_after_filter = len(count_columns)
    logger.info("  Mined pattern counts (%s): %d of %d patterns used",
                count_encoding, patterns_after_filter, counts.shape[1])
    filter_time = time.time() - filter_start

    # ----- Feature scaling -----
    representation_type = resolve_representation_type(config)
    if representation_type == "supervised_sum" and config.get("classifier_type") != "nn":
        logger.info("  supervised_sum uses its joint NN head; normalizing classifier_type to 'nn'")
        config["classifier_type"] = "nn"
    input_scaling = config.get("supervised_input_scaling") or "none"
    if input_scaling not in {"none", "train_maxabs", "train_std"}:
        raise ValueError("supervised_input_scaling must be 'none', 'train_maxabs' or 'train_std'")
    embed_start = time.time()
    X = count_values
    if input_scaling != "none":
        # Divide by a training-fitted column scale without centering, so
        # absent patterns stay exactly zero; all-zero or constant columns
        # keep scale 1.
        if input_scaling == "train_maxabs":
            scale = np.abs(X[train_index]).max(axis=0)
        else:
            scale = X[train_index].std(axis=0)
        scale[scale == 0] = 1.0
        X = X / scale
    embed_time = time.time() - embed_start
    logger.info(
        "  Feature matrix: %d graphs x %d patterns (%.2fs)",
        X.shape[0], X.shape[1], embed_time,
    )

    # ----- Classification -----
    X_train = X[train_index]
    X_val = X[validation_index]
    X_test = X[test_index]
    y_train = labels[train_index]
    y_val = labels[validation_index]

    def zero_vector_rate(values):
        if len(values) == 0:
            return 0.0
        return np.mean(np.all(values == 0, axis=1)) * 100

    raw_zero_rates = {"train": float(zero_vector_rate(X_train)),
                      "validation": float(zero_vector_rate(X_val)),
                      "test": float(zero_vector_rate(X_test))}
    logger.info(
        f"  Zero representations: train={raw_zero_rates['train']:.1f}%, "
        f"val={zero_vector_rate(X_val):.1f}%, test={zero_vector_rate(X_test):.1f}%"
    )

    # Select classifier based on config
    classifier_type = config.get("classifier_type", "nn")
    if representation_type == "supervised_sum":
        logger.info("  Joint supervised pattern encoder and NN head selected")
        clf = SupervisedPatternSetNN(config, input_dim=X_train.shape[1])
    elif classifier_type == "svm":
        clf = LinearSVM(config)
    elif classifier_type == "nn":
        clf = NN(config)
    else:
        raise ValueError(
            f"Unknown classifier_type '{classifier_type}'. "
            "Use 'nn' or 'svm'."
        )

    logger.debug("Training classifier...")
    clf_start = time.time()
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    predict_test = clf.predict(X_test)
    predict_val = clf.predict(X_val)
    clf_time = time.time() - clf_start

    # Test labels are deliberately materialized only after fitting and test
    # prediction.  They are used solely for the final evaluation metrics.
    y_test = labels[test_index]

    # ----- Training Plot -----
    if plot_dir is not None:
        if hasattr(clf, "history") and clf.history is not None:
            plot_path = os.path.join(plot_dir, "training_history.png")
            try:
                plot_training_history(clf.history, plot_path)
            except Exception as e:
                logger.warning(f"Could not save training plot: {e}")

    # ----- Metrics -----
    accuracy = np.mean(np.array(predict_test) == y_test) * 100
    metric_labels = list(range(config["num_classes"])) if "num_classes" in config else None
    macro_f1 = f1_score(y_test, predict_test, average="macro",
                        labels=metric_labels, zero_division=0) * 100
    val_accuracy = np.mean(np.array(predict_val) == y_val) * 100
    val_macro_f1 = f1_score(y_val, predict_val, average="macro",
                            labels=metric_labels, zero_division=0) * 100

    logger.info(f"  Accuracy: {accuracy:.2f}%, Macro-F1: {macro_f1:.2f}% "
                f"(Val Acc: {val_accuracy:.2f}%, {patterns_after_filter} features)")

    history = getattr(clf, "history", None)
    epochs_trained = 0
    best_epoch = 0
    best_validation_accuracy = 0.0
    if history is not None and history.history.get("val_loss"):
        epochs_trained = len(history.history.get("loss", []))
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        best_validation_accuracy = float(history.history["val_accuracy"][best_epoch - 1])
    return SplitEvaluationResult(
        accuracy, macro_f1, filter_time, embed_time, clf_time,
        patterns_after_filter, raw_zero_rates,
        total_time=time.time() - split_start,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        best_validation_accuracy=best_validation_accuracy,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        selected_pattern_ids=list(map(int, getattr(t_codes, "pattern_ids", []))),
        predictions={
            name: {"graph_ids": list(map(int, indices)),
                   "truth": list(map(int, truth)),
                   "predicted": list(map(int, predicted))}
            for name, indices, truth, predicted in (
                ("validation", validation_index, y_val, predict_val),
                ("test", test_index, y_test, predict_test),
            )
        },
        trainable_parameters=(sum(int(np.prod(weight.shape)) for weight in
                                  clf.model.trainable_weights)
                              if hasattr(getattr(clf, "model", None), "trainable_weights")
                              else None),
        training_history=({key: list(map(float, values)) for key, values in history.history.items()}
                          if history is not None else {}),
    )
