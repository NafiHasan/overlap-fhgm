"""Mining and classification runs under the two vocabulary-access protocols.

Fully inductive: the vocabulary is mined from the training hypergraphs of each
repeat and frozen; validation and test hypergraphs are only matched against it.
Transductive vocabulary: the unlabeled validation and test hypergraphs also
take part in mining. In both, feature scaling and the classifier are fitted on
the training hypergraphs only. Runs are resumable, and mining results are
cached by configuration.
"""

import contextlib
import copy
import csv
import fcntl
import io
import json
import logging
import os
import platform
import resource
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

from classification.evaluate import _run_single_split, resolve_representation_type
from classification.validation import validate_experiment_settings
from classification.pattern_counts import (
    compute_pattern_tallies, pattern_count_matrix, resolve_count_options, tally_depth,
    tally_overlap_binning,
)
from mining.fhgm import FHGMMiner, resolve_pattern_options
from mining.io import load_coverage_file, load_hypergraphs_from_file, load_labels
from mining.pattern_io import load_pattern_vocabulary, save_pattern_vocabulary
from mining.transform import match_pattern_vocabulary, merge_binary_coverage, vocabulary_signature
from .artifacts import (
    CACHE_ARTIFACTS,
    REPEAT_ARTIFACTS,
    TRANSDUCTIVE_CACHE_ARTIFACTS,
    artifact_hashes,
    atomic_gzip_json_write,
    atomic_json_write,
    atomic_text_write,
    canonical_json_hash,
    file_lock,
    load_gzip_json,
    mining_config_hash,
    sha256_file,
    valid_repeat_cache,
    valid_completion_marker,
    write_completion_marker,
)
from .splits import load_and_validate_split_manifest
from .label_binning import bin_hypergraph_labels, effective_label_mode, resolve_label_binning
from .preprocessing import verify_preprocessing
from .provenance import capture_provenance, implementation_identity, require_unchanged_source

logger = logging.getLogger(__name__)

_SUPPORTED = {
    "imdb_dir_form", "imdb_dir_genre", "steam_player",
    "twitter_friend", "rhg_10", "rhg_3",
}
_VOCABULARY_ACCESS_MODES = {"train_only", "all_unlabeled"}
_MINING_CONFIG_KEYS = (
    "base_sup", "sup_discount", "max_phases", "min_sup_floor",
    "min_coverage", "max_depth", "label_binning",
    "overlap_labels", "exhaustive_depth", "exhaustive_min_graphs",
)
_CLASSIFICATION_CONFIG_KEYS = (
    "representation_type", "classifier_type", "min_cov", "embedding_dim",
    "supervised_pattern_dropout", "supervised_encoder_l2",
    "nn_layers",
    "nn_dropouts", "nn_l2_regs", "nn_learning_rate", "nn_epochs",
    "nn_batch_size", "nn_patience", "nn_lr_decay_factor", "nn_lr_patience",
    "nn_min_lr", "num_classes", "deterministic_training", "nn_shuffle",
    "svm_C", "svm_max_iter",
    "supervised_input_scaling",
    "supervised_count_feature", "supervised_layer_norm",
    "mined_count_encoding", "mined_min_depth", "mined_min_train_graphs",
    "mined_selection",
)
_REPEAT_CSV_FIELDS = (
    "repeat", "mining_hash", "cache_reused", "patterns", "phase_patterns",
    "fallback_patterns", "selected_patterns",
    "train_zero_rate", "validation_zero_rate", "test_zero_rate",
    "val_accuracy", "val_macro_f1", "mining_seconds", "classification_seconds",
    "accuracy", "macro_f1",
    "validation_matching_seconds", "test_matching_seconds", "end_to_end_seconds",
    "trainable_parameters", "process_peak_rss_mb",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _protocol_contract(config):
    """Resolve protocol-specific paths and immutable artifact stage names."""
    access = config.get("vocabulary_access", "train_only")
    pipeline_mode = config.get("pipeline_mode")
    if access == "train_only":
        return {
            "access": access,
            "display_name": "Inductive",
            "result_subdir": "inductive",
            "cache_dir_key": "inductive_mining_cache_dir",
            "cache_dir_name": "inductive_mining_cache",
            "cache_stage": "inductive_mining_cache",
            "repeat_stage": "inductive_classification_repeat",
            "run_stage": "fully_inductive_run",
            "coverage_artifact": "train_coverage.json.gz",
            "cache_artifacts": CACHE_ARTIFACTS,
            "report_name": "inductive_report.md",
        }
    if access == "all_unlabeled" and pipeline_mode == "transductive_vocabulary":
        return {
            "access": access,
            "display_name": "Transductive-vocabulary",
            "result_subdir": "transductive",
            "cache_dir_key": "transductive_mining_cache_dir",
            "cache_dir_name": "transductive_mining_cache",
            "cache_stage": "transductive_mining_cache",
            "repeat_stage": "transductive_classification_repeat",
            "run_stage": "transductive_vocabulary_run",
            "coverage_artifact": "discovery_coverage.json.gz",
            "cache_artifacts": TRANSDUCTIVE_CACHE_ARTIFACTS,
            "report_name": "transductive_report.md",
        }
    raise ValueError(
        "Unsupported vocabulary_access/pipeline_mode combination"
    )


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _selected_config(config, keys):
    return {key: config.get(key) for key in keys}


def _effective_config_sections(config):
    """Return the effective settings used for hashing and provenance.

    Settings that the chosen representation or classifier does not read are
    left out, so they cannot change a run's identity.
    """
    classification_config = _selected_config(
        config, _CLASSIFICATION_CONFIG_KEYS
    )
    representation = resolve_representation_type(config)
    classification_config["representation_type"] = representation
    if representation == "supervised_sum":
        classification_config["classifier_type"] = "nn"
    classifier = classification_config.get("classifier_type") or "nn"
    classification_config["classifier_type"] = classifier
    for prefix, active in (("nn_", classifier == "nn"),
                           ("svm_", classifier == "svm")):
        if not active:
            classification_config = {key: value for key, value in classification_config.items()
                                     if not key.startswith(prefix)}
    if representation == "raw":
        for key in ("embedding_dim", "supervised_pattern_dropout",
                    "supervised_encoder_l2", "supervised_layer_norm"):
            classification_config.pop(key, None)
    # Unset and true both mean the default architecture with layer normalization.
    if config.get("supervised_layer_norm") in (None, True):
        classification_config.pop("supervised_layer_norm", None)
    # Unset and "none" both mean no pattern selection; min_cov is read only
    # by the selection.
    if config.get("mined_selection") in (None, "none"):
        classification_config.pop("mined_selection", None)
        classification_config.pop("min_cov", None)
    return {
        "mining_config": _selected_config(config, _MINING_CONFIG_KEYS),
        "classification_config": classification_config,
    }


def _run_config_hash(config):
    sections = _effective_config_sections(config)
    payload = {
        "implementation": implementation_identity()["run"],
        "require_clean_source": config.get("require_clean_source", False),
        "experiment_profile": config.get("experiment_profile"),
        "dataset": config.get("_dataset"),
        "label_mode": config.get("_label_mode"),
        "domains": config.get("domains"),
        "pipeline_mode": config.get("pipeline_mode"),
        "vocabulary_access": config.get("vocabulary_access", "train_only"),
        "split_protocol": config.get("split_protocol"),
        "train_ratio": config.get("train_ratio"),
        "validation_ratio": config.get("validation_ratio"),
        "num_repeats": config.get("num_repeats"),
        "mining": sections["mining_config"],
        "classification": sections["classification_config"],
    }
    return canonical_json_hash(payload)


def _validate_config(config, repeat):
    if not isinstance(config.get("run_id"), str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["run_id"]):
        raise ValueError("run_id must be a filename-safe identifier")
    config["representation_type"] = resolve_representation_type(config)
    if config["representation_type"] == "supervised_sum":
        config["classifier_type"] = "nn"
    vocabulary_access = config.setdefault("vocabulary_access", "train_only")
    if vocabulary_access not in _VOCABULARY_ACCESS_MODES:
        raise ValueError(
            "vocabulary_access must be 'train_only' or 'all_unlabeled'"
        )
    expected_mode = ("fully_inductive" if vocabulary_access == "train_only"
                     else "transductive_vocabulary")
    if config.get("pipeline_mode") != expected_mode:
        raise ValueError(
            f"pipeline_mode must be {expected_mode!r} when "
            f"vocabulary_access={vocabulary_access}"
        )
    if len(config.get("domains", [])) != 1:
        raise ValueError("A run reads exactly one hypergraph file")
    if config.get("_dataset") not in _SUPPORTED:
        raise ValueError("Dataset is not supported")
    if config.get("train_ratio") != .8 or config.get("validation_ratio") != .1:
        raise ValueError("Runs require the DHG-Bench 80/10/10 ratios")
    if config.get("num_repeats") not in range(1, 6):
        raise ValueError("num_repeats must be in 1..5")
    if repeat is not None and repeat not in range(5):
        raise ValueError("Requested repeat must be in 0..4")
    if repeat is not None and repeat >= config["num_repeats"]:
        raise ValueError("Requested repeat is outside the configured repeat set")
    if repeat is not None and type(repeat) is not int:
        raise ValueError("repeat must be an integer")
    resolve_label_binning(config)
    resolve_pattern_options(config)
    resolve_count_options(config)
    validate_experiment_settings(config)


def _write_coverage(path, patterns, coverage):
    handle = io.StringIO()
    for pattern_id, pattern in enumerate(patterns):
        handle.write("#" + str(pattern))
        for graph_id in sorted(coverage[pattern_id]):
            handle.write(f"{graph_id} 1\n")
        handle.write("\n")
    atomic_text_write(path, handle.getvalue())


def _coverage_is_valid(cache_dir, vocabulary_hash, split_hash,
                       vocabulary_access="train_only"):
    path = os.path.join(cache_dir, "coverage_metadata.json")
    coverage_path = os.path.join(cache_dir, "cover_freq.txt")
    if not os.path.isfile(path) or not os.path.isfile(coverage_path):
        return False
    try:
        metadata = _read_json(path)
        recorded_split = metadata.get("split_manifest_hash")
        return (
            metadata.get("schema_version") == 2 and
            metadata.get("coverage_semantics") == "binary_presence" and
            metadata.get("vocabulary_access", "train_only") == vocabulary_access and
            metadata.get("vocabulary_hash") == vocabulary_hash and
            recorded_split == split_hash
        )
    except (OSError, ValueError, TypeError):
        return False


def _serialize_train_coverage(coverage):
    return {str(pattern_id): sorted(map(int, graph_ids))
            for pattern_id, graph_ids in coverage.items()}


def _deserialize_train_coverage(value):
    return {int(pattern_id): {int(graph_id): 1 for graph_id in graph_ids}
            for pattern_id, graph_ids in value.items()}


def _cache_artifact_names(config, cache_dir, schema_version):
    if schema_version != 2:
        raise ValueError(f"Unsupported mining cache schema {schema_version}; rebuild it")
    return list(_protocol_contract(config)["cache_artifacts"])


def _prepare_cache(config, *, mode, force_remine, cache_root, dataset,
                   label_mode, domain, hypergraphs, manifest_path, split,
                   source_hash):
    contract = _protocol_contract(config)
    vocabulary_access = contract["access"]
    mining_hash = mining_config_hash(
        config, dataset=dataset, label_mode=effective_label_mode(label_mode, config),
        manifest_hash=split.manifest_hash, source_hash=source_hash,
    )
    cache_dir = os.path.join(cache_root, mining_hash)
    os.makedirs(cache_root, exist_ok=True)

    with file_lock(cache_dir + ".lock"):
        cache_valid = valid_repeat_cache(
            cache_dir, mining_hash=mining_hash,
            manifest_hash=split.manifest_hash, source_hash=source_hash,
            vocabulary_access=vocabulary_access,
        )
        if mode == "classification" and (force_remine or not cache_valid):
            raise FileNotFoundError(
                f"No valid {contract['display_name'].lower()} mining cache "
                f"for repeat {split.repeat}. Run the matching mining stage first."
            )
        if not cache_valid and os.path.isfile(os.path.join(cache_dir, "completion.json")):
            raise ValueError(
                f"Completed cache is invalid and will not be overwritten: {cache_dir}. "
                "Preserve it for inspection and use a separate cache directory."
            )

        cache_reused = cache_valid and not force_remine
        if cache_valid and force_remine:
            raise ValueError(
                "Completed mining caches are immutable. Use a separate cache directory "
                "for a forced reproducibility check."
            )
        os.makedirs(cache_dir, exist_ok=True)
        patterns_path = os.path.join(cache_dir, "patterns.json")
        mining_metadata_path = os.path.join(cache_dir, "mining_metadata.json")

        if cache_reused:
            patterns, vocabulary_metadata = load_pattern_vocabulary(patterns_path)
            mining_metadata = _read_json(mining_metadata_path)
            schema_version = mining_metadata.get("schema_version", 1)
            discovery_coverage = _deserialize_train_coverage(
                load_gzip_json(os.path.join(
                    cache_dir, contract["coverage_artifact"]
                ))
            )
            logger.info("Repeat %d: reusing mining cache %s", split.repeat, mining_hash[:12])
        else:
            completion_path = os.path.join(cache_dir, "completion.json")
            if os.path.isfile(completion_path):
                os.unlink(completion_path)
            shutil.copyfile(manifest_path, os.path.join(cache_dir, "split.json"))
            discovery_ids = (
                split.train.tolist()
                if vocabulary_access == "train_only"
                else list(range(len(hypergraphs)))
            )
            discovery_graphs = [hypergraphs[index] for index in discovery_ids]
            logger.info(
                "Repeat %d: mining %d %s graphs (cache %s)",
                split.repeat,
                len(discovery_graphs),
                "training" if vocabulary_access == "train_only" else "unlabeled-access",
                mining_hash[:12],
            )
            miner = FHGMMiner()
            result = miner.mine_hypergraphs(
                discovery_graphs, discovery_ids, config,
                domain_name=domain,
                pattern_text_path=os.path.join(cache_dir, "patterns.txt"),
            )
            patterns, discovery_coverage = result.patterns, result.train_coverage
            vocabulary_metadata = {
                "source_hypergraph_hash": source_hash,
                "split_manifest_hash": split.manifest_hash,
                "mining_config_hash": mining_hash,
                "vocabulary_access": vocabulary_access,
            }
            save_pattern_vocabulary(patterns_path, patterns, vocabulary_metadata)
            atomic_gzip_json_write(
                os.path.join(cache_dir, contract["coverage_artifact"]),
                _serialize_train_coverage(discovery_coverage),
            )
            mining_metadata = {
                "schema_version": 2,
                "require_clean_source": config.get("require_clean_source", False),
                "implementation": config["_invocation_provenance"]["implementation"]["mining"],
                **vocabulary_metadata,
                "repeat": split.repeat,
                "discovery_graph_ids": list(map(int, discovery_ids)),
                "supervised_graph_ids": list(map(int, split.train)),
                "statistics": result.statistics,
            }
            atomic_json_write(mining_metadata_path, mining_metadata)
            atomic_text_write(os.path.join(cache_dir, "mining_report.md"), result.report_markdown)
            atomic_json_write(os.path.join(cache_dir, "provenance.json"),
                              config["_invocation_provenance"])
            atomic_json_write(os.path.join(cache_dir, "preprocessing.json"),
                              config["_preprocessing"])
            schema_version = 2

        vocabulary_hash = canonical_json_hash(
            [list(item) for item in vocabulary_signature(patterns)]
        )
        coverage_valid = _coverage_is_valid(
            cache_dir,
            vocabulary_hash,
            split.manifest_hash,
            vocabulary_access,
        )
        coverage_reused = coverage_valid and not force_remine
        if coverage_reused:
            coverage = load_coverage_file(os.path.join(cache_dir, "cover_freq.txt"))
        elif vocabulary_access == "train_only":
            validation_start = time.time()
            validation_coverage = match_pattern_vocabulary(
                patterns, [hypergraphs[index] for index in split.validation],
                split.validation.tolist(), partition_name="validation",
            )
            validation_seconds = time.time() - validation_start
            test_start = time.time()
            test_coverage = match_pattern_vocabulary(
                patterns, [hypergraphs[index] for index in split.test],
                split.test.tolist(), partition_name="test",
            )
            test_seconds = time.time() - test_start
            if vocabulary_hash != canonical_json_hash(
                    [list(item) for item in vocabulary_signature(patterns)]):
                raise RuntimeError("Vocabulary changed while matching held-out data")
            coverage = merge_binary_coverage(
                patterns,
                (
                    ("train", split.train, discovery_coverage),
                    ("validation", split.validation, validation_coverage),
                    ("test", split.test, test_coverage),
                ),
                known_graph_ids=range(len(hypergraphs)),
            )
            _write_coverage(os.path.join(cache_dir, "cover_freq.txt"), patterns, coverage)
            atomic_json_write(os.path.join(cache_dir, "coverage_metadata.json"), {
                "schema_version": 2,
                "coverage_semantics": "binary_presence",
                "vocabulary_access": vocabulary_access,
                "coverage_source": "training_mining_plus_frozen_heldout_matching",
                "vocabulary_hash": vocabulary_hash,
                "split_manifest_hash": split.manifest_hash,
                "validation_matching_seconds": validation_seconds,
                "test_matching_seconds": test_seconds,
            })
        else:
            coverage = discovery_coverage
            _write_coverage(
                os.path.join(cache_dir, "cover_freq.txt"), patterns, coverage
            )
            atomic_json_write(os.path.join(cache_dir, "coverage_metadata.json"), {
                "schema_version": 2,
                "coverage_semantics": "binary_presence",
                "vocabulary_access": vocabulary_access,
                "coverage_source": "all_graph_structure_mining",
                "vocabulary_hash": vocabulary_hash,
                "split_manifest_hash": split.manifest_hash,
                "validation_matching_seconds": 0.0,
                "test_matching_seconds": 0.0,
            })

        require_unchanged_source(config["_invocation_provenance"])
        if schema_version == 2 and (not cache_reused or not coverage_reused):
            names = _cache_artifact_names(config, cache_dir, schema_version)
            write_completion_marker(
                cache_dir,
                stage=contract["cache_stage"],
                identity={"mining_hash": mining_hash, "repeat": split.repeat,
                          "vocabulary_access": vocabulary_access},
                artifact_names=names,
                completed_at=_utc_now(),
            )

        artifact_names = _cache_artifact_names(config, cache_dir, schema_version)
        if schema_version == 2 and os.path.isfile(os.path.join(cache_dir, "completion.json")):
            cache_hashes = dict(
                _read_json(os.path.join(cache_dir, "completion.json"))["artifacts"]
            )
            cache_hashes["completion.json"] = sha256_file(
                os.path.join(cache_dir, "completion.json")
            )
        else:
            cache_hashes = artifact_hashes(cache_dir, artifact_names)
        statistics = mining_metadata.get("statistics", {})
        if not valid_repeat_cache(
                cache_dir, mining_hash=mining_hash, manifest_hash=split.manifest_hash,
                source_hash=source_hash, vocabulary_access=vocabulary_access):
            raise RuntimeError("Mining cache failed validation after preparation")
        coverage_metadata = _read_json(os.path.join(cache_dir, "coverage_metadata.json"))
        return patterns, coverage, {
            "repeat": split.repeat,
            "vocabulary_access": vocabulary_access,
            "cache_stage": contract["cache_stage"],
            "mining_hash": mining_hash,
            "cache_dir": cache_dir,
            "cache_reused": cache_reused,
            "coverage_reused": coverage_reused,
            "pattern_count": len(patterns),
            "phase_pattern_count": statistics.get("phase_patterns", ""),
            "fallback_pattern_count": statistics.get("fallback_patterns", ""),
            "mining_seconds": statistics.get("total_seconds", ""),
            "validation_matching_seconds": coverage_metadata["validation_matching_seconds"],
            "test_matching_seconds": coverage_metadata["test_matching_seconds"],
            "producer_provenance": _read_json(os.path.join(cache_dir, "provenance.json")),
            "coverage_statistics": {
                key: statistics.get(key, {})
                for key in (
                    "post_phase_coverage",
                    "post_fallback_working_coverage",
                    "matched_train_coverage",
                )
            },
            "schema_version": schema_version,
            "vocabulary_hash": vocabulary_hash,
            "artifacts": cache_hashes,
        }


@contextlib.contextmanager
def _shared_file_lock(path):
    """Wait for, then hold, a lock on a file shared by concurrent dataset runs."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _update_cache_index(cache_root, dataset, label_mode, records):
    path = os.path.join(cache_root, "cache_index.json")
    with _shared_file_lock(path):
        _update_cache_index_locked(path, dataset, label_mode, records)


def _update_cache_index_locked(path, dataset, label_mode, records):
    indexed = {}
    if os.path.isfile(path):
        try:
            indexed = {item["mining_hash"]: item for item in
                       _read_json(path).get("artifacts", [])}
        except (OSError, ValueError, KeyError, TypeError):
            indexed = {}
    for record in records:
        indexed[record["mining_hash"]] = {
            key: record[key] for key in
            ("repeat", "mining_hash", "cache_dir", "pattern_count", "schema_version")
        }
    atomic_json_write(path, {
        "schema_version": 2,
        "dataset": dataset,
        "label_mode": label_mode,
        "updated_at": _utc_now(),
        "artifacts": sorted(indexed.values(), key=lambda item: (item["repeat"], item["mining_hash"])),
    })


def _write_repeat_csv(root, metrics):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_REPEAT_CSV_FIELDS)
    writer.writeheader()
    for record in sorted(metrics, key=lambda item: item["repeat"]):
        writer.writerow({key: record.get(key, "") for key in _REPEAT_CSV_FIELDS})
    atomic_text_write(os.path.join(root, "repeats.csv"), output.getvalue())


def _environment_metadata(provenance):
    try:
        import tensorflow as tf
        tensorflow_version = tf.__version__
        devices = [device.name for device in tf.config.list_physical_devices()]
    except (ImportError, AttributeError):
        tensorflow_version, devices = None, []
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tensorflow_version,
        "platform": platform.platform(),
        "devices": devices,
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "git_source_dirty": provenance["git_source_dirty"],
    }


def _active_log_files():
    paths = []
    for handler in logging.getLogger().handlers:
        path = getattr(handler, "baseFilename", None)
        if path:
            paths.append(os.path.relpath(path))
    return sorted(set(paths))


def _aggregate(metrics):
    accuracies = [record["accuracy"] for record in metrics]
    macro_f1s = [record["macro_f1"] for record in metrics]
    val_accuracies = [record.get("val_accuracy", record.get("best_validation_accuracy", 0.0))
                      for record in metrics]
    val_macro_f1s = [record.get("val_macro_f1", 0.0) for record in metrics]
    result = {
        "val_accuracy_mean": float(np.mean(val_accuracies)),
        "val_accuracy_std": float(np.std(val_accuracies)),
        "val_macro_f1_mean": float(np.mean(val_macro_f1s)),
        "val_macro_f1_std": float(np.std(val_macro_f1s)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
    }
    return result


def _render_report(config, metrics, aggregate, active_seconds):
    reused = sum(bool(record["cache_reused"]) for record in metrics)
    contract = _protocol_contract(config)
    lines = [
        f"# {contract['display_name']} Run {config['run_id']}", "",
        "Status: **complete**  ",
        f"Dataset: `{config['_dataset']}` / `{config['_label_mode']}`  ",
        f"Protocol: `{config['split_protocol']}`  ",
        f"Pipeline mode: `{config['pipeline_mode']}`  ",
        f"Vocabulary access: `{contract['access']}`  ",
        f"Representation: `{resolve_representation_type(config)}`  ",
        f"Active time: {active_seconds:.1f}s  ",
        f"Mining caches reused: {reused}/{len(metrics)}", "",
        "| Metric | Mean | Population std |",
        "|---|---:|---:|",
        f"| Validation Accuracy | {aggregate['val_accuracy_mean']:.2f}% | {aggregate['val_accuracy_std']:.2f}% |",
        f"| Validation Macro-F1 | {aggregate['val_macro_f1_mean']:.2f}% | {aggregate['val_macro_f1_std']:.2f}% |",
        f"| Accuracy | {aggregate['accuracy_mean']:.2f}% | {aggregate['accuracy_std']:.2f}% |",
        f"| Macro-F1 | {aggregate['macro_f1_mean']:.2f}% | {aggregate['macro_f1_std']:.2f}% |",
    ]
    lines.extend([
        "", "| Repeat | Patterns | Phase / fallback | Selected | Zero train / val / test | Val Acc / F1 | Test Acc / F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in sorted(metrics, key=lambda item: item["repeat"]):
        lines.append(
            f"| {record['repeat']} | {record['patterns']} | "
            f"{record.get('phase_patterns', '')} / "
            f"{record.get('fallback_patterns', '')} | "
            f"{record['selected_patterns']} | "
            f"{record.get('train_zero_rate', 0.0):.2f}% / "
            f"{record.get('validation_zero_rate', 0.0):.2f}% / "
            f"{record['test_zero_rate']:.2f}% | "
            f"{record.get('val_accuracy', 0.0):.2f}% / "
            f"{record.get('val_macro_f1', 0.0):.2f}% | "
            f"{record['accuracy']:.2f}% / {record['macro_f1']:.2f}% |"
        )
    lines.extend([
        "", "| Repeat | Mining s | Val/test matching s | Classification s | Full stage cost s | Trainable parameters | Process peak RSS MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in metrics:
        lines.append(
            f"| {record['repeat']} | {record.get('mining_seconds', '')} | "
            f"{record.get('validation_matching_seconds', '')} / {record.get('test_matching_seconds', '')} | "
            f"{record.get('classification_seconds', '')} | {record.get('end_to_end_seconds', '')} | "
            f"{record.get('trainable_parameters', '')} | {record.get('process_peak_rss_mb', '')} |"
        )
    lines.extend([
        "", "Full stage cost includes the mining and matching time recorded when each cache was built; active time measures this run's invocations.",
        "RSS is the process-lifetime high-water mark, not an isolated per-repeat measurement.",
        "", "Detailed provenance: `experiment.json`  ",
        "Repeat data: `repeats.csv`  ",
        "Per-repeat evidence: `predictions.json.gz`, `selected_patterns.json`, and `provenance.json`  ",
        "Authoritative completion: `completion.json`", "",
    ])
    return "\n".join(lines)


def _load_completed_metrics(root, expected_repeats,
                            repeat_stage="inductive_classification_repeat"):
    metrics = []
    for current in expected_repeats:
        repeat_dir = os.path.join(root, f"repeat_{current}")
        if not valid_completion_marker(
                repeat_dir, stage=repeat_stage,
                identity={"repeat": current}):
            continue
        metrics.append(_read_json(os.path.join(repeat_dir, "metrics.json")))
    return metrics


def _remove_stale_staging(root):
    """Remove only pipeline-owned, unpublished repeat staging directories."""
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.startswith(".repeat_") and os.path.isdir(path):
            shutil.rmtree(path)


_TALLY_CACHE = {}
_COUNTER_SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "classification", "pattern_counts.py")


def _pattern_tallies(hypergraph_file, hypergraphs, config):
    """Per-graph pattern tallies, reused across runs of one process."""
    depth, binning = tally_depth(config), tally_overlap_binning(config)
    key = (sha256_file(hypergraph_file), resolve_label_binning(config), depth, binning)
    if key not in _TALLY_CACHE:
        _TALLY_CACHE.clear()
        _TALLY_CACHE[key] = compute_pattern_tallies(hypergraphs, depth, binning)
    return _TALLY_CACHE[key]


def _cached_count_matrix(results_root, cache_record, config, patterns, hypergraphs):
    """Per-repeat count matrix, stored once per frozen vocabulary and counter.

    Tallying per repeat (exact labels, four-incidence overlap keys) takes
    minutes on large graphs, so classification-only reruns load the stored
    matrix. The key binds the mining cache, the vocabulary, the tally
    settings and the counter source; the caller still checks every count
    against the frozen matched presence.
    """
    key = canonical_json_hash({
        "schema_version": 1,
        "mining_hash": cache_record["mining_hash"],
        "vocabulary_hash": cache_record["vocabulary_hash"],
        "label_binning": resolve_label_binning(config),
        "tally_depth": tally_depth(config),
        "overlap_binning": tally_overlap_binning(config),
        "counter": sha256_file(_COUNTER_SOURCE),
    })
    cache_dir = os.path.join(results_root, "count_matrix_cache")
    path = os.path.join(cache_dir, key + ".npz")
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as stored:
            result = (stored["counts"], stored["totals"], stored["depths"])
        if result[0].shape != (len(hypergraphs), len(patterns)):
            raise ValueError(f"Stored count matrix has the wrong shape: {path}")
        logger.info("Pattern counts loaded from %s", path)
        return result
    start = time.time()
    result = pattern_count_matrix(patterns, hypergraphs=hypergraphs)
    os.makedirs(cache_dir, exist_ok=True)
    handle, staging = tempfile.mkstemp(suffix=".npz", dir=cache_dir)
    os.close(handle)
    try:
        np.savez_compressed(staging, counts=result[0], totals=result[1], depths=result[2])
        os.replace(staging, path)
    finally:
        if os.path.exists(staging):
            os.unlink(staging)
    logger.info("Pattern counts computed in %.1fs and stored in %s", time.time() - start, path)
    return result


def _check_counts_match_presence(counts, coverage):
    """A count must be positive exactly where frozen matching found the pattern."""
    for pattern_id in range(counts.shape[1]):
        counted = set(np.flatnonzero(counts[:, pattern_id]).tolist())
        if counted != set(map(int, coverage.get(pattern_id, {}))):
            raise RuntimeError(f"Pattern {pattern_id}: counts disagree with matched presence")


def _run_vocabulary_access_pipeline(config, *, split_dir=None, repeat=None,
                                    force_remine=False, mode="full"):
    """Run validated vocabulary mining and classification with safe resumption.

    The public inductive and transductive runners select the effective behavior
    through ``vocabulary_access`` and its matching ``pipeline_mode``.
    """
    config = copy.deepcopy(config)
    provenance = config.pop("_invocation_provenance", None) or capture_provenance()
    require_unchanged_source(provenance)
    environment = _environment_metadata(provenance)
    _validate_config(config, repeat)
    contract = _protocol_contract(config)
    if config.get("require_clean_source", False) and (
            not provenance["git_commit"] or provenance["git_source_dirty"] is not False):
        raise ValueError("require_clean_source needs committed, clean source files")
    config["_invocation_provenance"] = provenance
    if mode not in {"full", "mining", "classification"}:
        raise ValueError("mode must be 'full', 'mining', or 'classification'")

    invocation_start = time.time()
    paths, dataset, label_mode = config["paths"], config["_dataset"], config["_label_mode"]
    domain = config["domains"][0]
    hypergraph_file = os.path.join(paths["hypergraph_data_dir"], domain + ".txt")
    labels_file = paths["labels_file"]
    hypergraphs, labels = load_hypergraphs_from_file(hypergraph_file), load_labels(labels_file)
    if len(hypergraphs) != len(labels):
        raise ValueError("Hypergraph and label counts differ")
    preprocessing = verify_preprocessing(
        hypergraphs, labels, config, hypergraph_file, labels_file
    )
    # Only mining sees binned labels.
    hypergraphs = bin_hypergraph_labels(hypergraphs, config)
    config["_preprocessing"] = preprocessing
    # Pattern-count tallies use the labels the miner sees; no class labels and
    # no fitting. Binned labels give small tallies, cached for the process;
    # exact labels, and four-incidence overlap keys (about 10**7 per dataset),
    # are tallied per repeat, keeping only the vocabulary's keys.
    counts_enabled = mode != "mining"
    pattern_tallies = None
    if (counts_enabled and resolve_label_binning(config) != "none"
            and tally_overlap_binning(config) is None):
        tally_start = time.time()
        pattern_tallies = _pattern_tallies(hypergraph_file, hypergraphs, config)
        logger.info("Pattern-count tallies: %d graphs (%.1fs)",
                    len(pattern_tallies), time.time() - tally_start)

    expected_repeats = list(range(config["num_repeats"]))
    selected_repeats = [repeat] if repeat is not None else expected_repeats
    logger.info(
        "%s %s: dataset=%s, label_mode=%s, run_id=%s, repeats=%s",
        contract["display_name"],
        mode,
        dataset,
        label_mode,
        config["run_id"],
        ",".join(map(str, selected_repeats)),
    )
    manifests = split_dir or config["split_manifest_dir"]
    validated = {}
    for current in selected_repeats:
        manifest_path = os.path.join(manifests, f"repeat_{current}.json")
        try:
            split = load_and_validate_split_manifest(
                manifest_path, dataset=dataset, label_mode=label_mode,
                hypergraph_file=hypergraph_file, labels_file=labels_file,
                num_graphs=len(hypergraphs), expected_repeat=current,
                expected_domain=domain,
            )
            validated[current] = (manifest_path, split)
            if set(map(int, labels[split.train])) != set(range(config["num_classes"])):
                raise ValueError("Training partition must contain all configured classes")
        except (FileNotFoundError, ValueError) as exc:
            command = (
                "cd tools/dhgbench_split_generator && uv run generate_splits.py "
                f"--repo-root ../.. --dataset {dataset} --label-mode {label_mode} "
                f"--repeats {current}"
            )
            raise ValueError(
                f"A valid DHG-Bench manifest is required before mining: {exc}\n"
                f"Generate it with: {command}"
            ) from exc

    results_root = os.path.dirname(paths["coverage_file"])
    cache_root = config.get(
        contract["cache_dir_key"],
        os.path.join(results_root, contract["cache_dir_name"]),
    )
    source_hash = sha256_file(hypergraph_file)
    labels_hash = sha256_file(labels_file)

    if mode == "mining":
        records = []
        for current in selected_repeats:
            manifest_path, split = validated[current]
            _, _, record = _prepare_cache(
                config, mode=mode, force_remine=force_remine,
                cache_root=cache_root, dataset=dataset, label_mode=label_mode,
                domain=domain, hypergraphs=hypergraphs,
                manifest_path=manifest_path, split=split,
                source_hash=source_hash,
            )
            records.append(record)
        _update_cache_index(cache_root, dataset, label_mode, records)
        logger.info("MINING CACHE COMPLETE: %d repeat cache(s) in %s",
                    len(records), cache_root)
        return {"schema_version": 2, "mode": "mining", "artifacts": records}

    schema_version = 2

    root = os.path.join(
        results_root, contract["result_subdir"], config["run_id"]
    )
    os.makedirs(os.path.dirname(root), exist_ok=True)
    run_hash = _run_config_hash(config)

    with file_lock(root + ".lock"):
        completion_path = os.path.join(root, "completion.json")
        state_path = os.path.join(root, "run_state.json")
        if os.path.isfile(completion_path):
            raise FileExistsError(
                f"Run {config['run_id']} is complete and immutable; choose a new run ID"
            )
        if os.path.isdir(root) and not os.path.isfile(state_path):
            raise FileExistsError(
                f"Run directory {root} has no run state and will not be overwritten"
            )
        os.makedirs(root, exist_ok=True)

        if os.path.isfile(state_path):
            state = _read_json(state_path)
            if state.get("run_config_hash") != run_hash:
                raise ValueError("Cannot resume: configuration, implementation, or dependencies changed")
            initial_provenance = _read_json(os.path.join(root, "provenance.json"))
            if initial_provenance["implementation"]["run"] != provenance["implementation"]["run"]:
                raise ValueError("Cannot resume under a different implementation")
            expected_sources = {"hypergraphs": source_hash, "labels": labels_hash}
            if state.get("source_hashes") != expected_sources:
                raise ValueError("Cannot resume: the dataset source files have changed")
            active_before = float(state.get("active_seconds", 0.0))
            started_at = state.get("started_at", _utc_now())
        else:
            active_before = 0.0
            started_at = _utc_now()
            state = {}
            initial_provenance = provenance
            atomic_json_write(os.path.join(root, "provenance.json"), provenance)
            atomic_json_write(os.path.join(root, "preprocessing.json"), preprocessing)
        _remove_stale_staging(root)
        state.pop("error", None)
        state.pop("missing_repeats", None)
        invocation_modes = list(state.get("invocation_modes", []))
        if mode not in invocation_modes:
            invocation_modes.append(mode)
        state.update({
            "schema_version": schema_version,
            "status": "running",
            "run_id": config["run_id"],
            "pipeline_mode": config["pipeline_mode"],
            "vocabulary_access": contract["access"],
            "run_config_hash": run_hash,
            **_effective_config_sections(config),
            "expected_repeats": expected_repeats,
            "completed_repeats": [record["repeat"] for record in
                                  _load_completed_metrics(
                                      root, expected_repeats,
                                      contract["repeat_stage"],
                                  )],
            "started_at": started_at,
            "updated_at": _utc_now(),
            "active_seconds": active_before,
            "last_mode": mode,
            "invocation_modes": invocation_modes,
            "source_hashes": {"hypergraphs": source_hash, "labels": labels_hash},
            "invocations": [*state.get("invocations", []), {
                "started_at": _utc_now(), "mode": mode,
                "repeats": selected_repeats, "provenance": provenance,
                "environment": environment,
            }],
        })
        atomic_json_write(state_path, state)

        cache_records = []
        mined_during_run = bool(state.get("mined_during_run", False))
        try:
            for current in selected_repeats:
                repeat_dir = os.path.join(root, f"repeat_{current}")
                if valid_completion_marker(
                        repeat_dir, stage=contract["repeat_stage"],
                        identity={"run_id": config["run_id"], "repeat": current,
                                  "run_config_hash": run_hash}):
                    logger.info("Repeat %d already complete; skipping", current)
                    continue
                if os.path.exists(repeat_dir):
                    raise FileExistsError(
                        f"Incomplete repeat directory cannot be overwritten: {repeat_dir}"
                    )

                manifest_path, split = validated[current]
                patterns, coverage, cache_record = _prepare_cache(
                    config, mode=mode, force_remine=force_remine,
                    cache_root=cache_root, dataset=dataset, label_mode=label_mode,
                    domain=domain, hypergraphs=hypergraphs,
                    manifest_path=manifest_path, split=split,
                    source_hash=source_hash,
                )
                cache_records.append(cache_record)
                mined_during_run = mined_during_run or not cache_record["cache_reused"]
                pattern_counts = (
                    pattern_count_matrix(patterns, pattern_tallies)
                    if pattern_tallies is not None else
                    _cached_count_matrix(results_root, cache_record, config,
                                         patterns, hypergraphs))
                _check_counts_match_presence(pattern_counts[0], coverage)

                staging = tempfile.mkdtemp(prefix=f".repeat_{current}_", dir=root)
                shutil.copyfile(manifest_path, os.path.join(staging, "split.json"))
                atomic_json_write(os.path.join(staging, "mining_reference.json"), {
                    "schema_version": 2,
                    "vocabulary_access": contract["access"],
                    "cache_stage": contract["cache_stage"],
                    "mining_hash": cache_record["mining_hash"],
                    "cache_dir": os.path.relpath(cache_record["cache_dir"], repeat_dir),
                    "vocabulary_hash": cache_record["vocabulary_hash"],
                    "artifacts": cache_record["artifacts"],
                })

                repeat_config = copy.deepcopy(config)
                repeat_config["random_state"] = current
                repeat_paths = copy.deepcopy(paths)
                repeat_paths["coverage_file"] = os.path.join(
                    cache_record["cache_dir"], "cover_freq.txt"
                )
                evaluation = _run_single_split(
                    split.train, split.test, coverage, labels,
                    repeat_config, repeat_paths,
                    validation_index=split.validation,
                    pattern_counts=pattern_counts,
                    fold_label=f"Repeat {current}",
                    plot_dir=os.path.join(staging, "plots"),
                )
                record = {
                    "schema_version": schema_version,
                    "pipeline_mode": config["pipeline_mode"],
                    "vocabulary_access": contract["access"],
                    "run_config_hash": run_hash,
                    "implementation": provenance["implementation"]["run"],
                    "num_classes": config["num_classes"],
                    "repeat": current,
                    "model_seed": split.model_seed,
                    "split_seed": split.split_seed,
                    "split_manifest_hash": split.manifest_hash,
                    "mining_hash": cache_record["mining_hash"],
                    "cache_reused": cache_record["cache_reused"],
                    "patterns": len(patterns),
                    "phase_patterns": cache_record["phase_pattern_count"],
                    "fallback_patterns": cache_record["fallback_pattern_count"],
                    "selected_patterns": evaluation.selected_pattern_count,
                    "train_zero_rate": evaluation.raw_zero_vector_rates["train"],
                    "validation_zero_rate": evaluation.raw_zero_vector_rates["validation"],
                    "test_zero_rate": evaluation.raw_zero_vector_rates["test"],
                    "val_accuracy": evaluation.val_accuracy,
                    "val_macro_f1": evaluation.val_macro_f1,
                    "mining_seconds": cache_record["mining_seconds"],
                    "classification_seconds": evaluation.total_time,
                    "validation_matching_seconds": cache_record["validation_matching_seconds"],
                    "test_matching_seconds": cache_record["test_matching_seconds"],
                    "end_to_end_seconds": (
                        float(cache_record["mining_seconds"]) +
                        cache_record["validation_matching_seconds"] +
                        cache_record["test_matching_seconds"] + evaluation.total_time
                    ),
                    "filtering_seconds": evaluation.filtering_time,
                    "embedding_seconds": evaluation.embedding_time,
                    "classifier_seconds": evaluation.classifier_time,
                    "trainable_parameters": evaluation.trainable_parameters,
                    "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                        1024**2 if platform.system() == "Darwin" else 1024
                    ),
                    "memory_scope": "process lifetime high-water RSS, not isolated repeat peak",
                    "accuracy": evaluation.accuracy,
                    "macro_f1": evaluation.macro_f1,
                    "epochs_trained": evaluation.epochs_trained,
                    "best_epoch": evaluation.best_epoch,
                    "best_validation_accuracy": evaluation.best_validation_accuracy,
                }
                if evaluation.training_history:
                    atomic_json_write(os.path.join(staging, "training_history.json"), evaluation.training_history)
                atomic_json_write(os.path.join(staging, "metrics.json"), record)
                atomic_json_write(os.path.join(staging, "provenance.json"), provenance)
                atomic_gzip_json_write(os.path.join(staging, "predictions.json.gz"),
                                       evaluation.predictions)
                atomic_json_write(os.path.join(staging, "selected_patterns.json"), {
                    "vocabulary_hash": cache_record["vocabulary_hash"],
                    "pattern_ids": evaluation.selected_pattern_ids,
                })
                require_unchanged_source(provenance)
                repeat_artifacts = list(REPEAT_ARTIFACTS)
                if evaluation.training_history:
                    repeat_artifacts.append("training_history.json")
                plot_path = os.path.join(staging, "plots", "training_history.png")
                if os.path.isfile(plot_path):
                    repeat_artifacts.append("plots/training_history.png")
                write_completion_marker(
                    staging,
                    stage=contract["repeat_stage"],
                    identity={"run_id": config["run_id"], "repeat": current,
                              "run_config_hash": run_hash},
                    artifact_names=repeat_artifacts,
                    completed_at=_utc_now(),
                    schema_version=schema_version,
                )
                os.replace(staging, repeat_dir)

                state["completed_repeats"] = sorted(
                    set(state.get("completed_repeats", [])) | {current}
                )
                state["mined_during_run"] = mined_during_run
                state["updated_at"] = _utc_now()
                state["active_seconds"] = active_before + time.time() - invocation_start
                atomic_json_write(state_path, state)

            if cache_records:
                _update_cache_index(cache_root, dataset, label_mode, cache_records)

            metrics = _load_completed_metrics(
                root, expected_repeats, contract["repeat_stage"]
            )
            completed = {record["repeat"] for record in metrics}
            missing = [current for current in expected_repeats if current not in completed]
            active_seconds = active_before + time.time() - invocation_start
            if missing:
                state.update({
                    "status": "partial",
                    "completed_repeats": sorted(completed),
                    "missing_repeats": missing,
                    "updated_at": _utc_now(),
                    "active_seconds": active_seconds,
                    "mined_during_run": mined_during_run,
                })
                atomic_json_write(state_path, state)
                logger.info(
                    "RUN PARTIAL: missing repeats %s; rerun the original %s "
                    "command with identical options to resume",
                    missing,
                    mode,
                )
                return {"schema_version": schema_version, "status": "partial", "missing_repeats": missing}

            # A run is finalized only after every currently installed manifest
            # validates and still matches the repeat artifact created from it.
            for record in metrics:
                current = record["repeat"]
                if current not in validated:
                    manifest_path = os.path.join(manifests, f"repeat_{current}.json")
                    current_split = load_and_validate_split_manifest(
                        manifest_path, dataset=dataset, label_mode=label_mode,
                        hypergraph_file=hypergraph_file, labels_file=labels_file,
                        num_graphs=len(hypergraphs), expected_repeat=current,
                        expected_domain=domain,
                    )
                    validated[current] = (manifest_path, current_split)
                if validated[current][1].manifest_hash != record["split_manifest_hash"]:
                    raise ValueError(
                        f"Cannot finalize: split manifest changed for repeat {current}"
                    )

            aggregate = _aggregate(metrics)
            require_unchanged_source(provenance)
            _write_repeat_csv(root, metrics)
            ended_at = _utc_now()
            split_hashes = {str(record["repeat"]): record["split_manifest_hash"]
                            for record in metrics}
            cache_hashes = {str(record["repeat"]): record["mining_hash"]
                            for record in metrics}
            experiment = {
                "schema_version": schema_version,
                "status": "complete",
                "run_id": config["run_id"],
                "vocabulary_access": contract["access"],
                "mode": "full" if "full" in invocation_modes else "classification",
                "protocol": config["split_protocol"],
                "config": {key: value for key, value in config.items()
                           if key not in {"_invocation_provenance", "_preprocessing"}},
                "run_config_hash": run_hash,
                "source_hashes": {"hypergraphs": source_hash, "labels": labels_hash},
                "split_manifest_hashes": split_hashes,
                "mining_hashes": cache_hashes,
                "environment": state["invocations"][0]["environment"],
                "provenance": initial_provenance,
                "invocations": state["invocations"],
                "log_files": _active_log_files(),
                "started_at": started_at,
                "completed_at": ended_at,
                "active_seconds": active_seconds,
                "aggregate": aggregate,
                "artifacts": {
                    "repeat_metrics": "repeats.csv",
                    "report": contract["report_name"],
                    "state": "run_state.json",
                },
            }
            atomic_json_write(os.path.join(root, "experiment.json"), experiment)
            atomic_text_write(
                os.path.join(root, contract["report_name"]),
                _render_report(config, metrics, aggregate, active_seconds),
            )
            state.update({
                "status": "complete",
                "completed_repeats": expected_repeats,
                "missing_repeats": [],
                "completed_at": ended_at,
                "updated_at": ended_at,
                "active_seconds": active_seconds,
                "mined_during_run": mined_during_run,
            })
            atomic_json_write(state_path, state)

            root_artifacts = [
                "run_state.json", "experiment.json", "repeats.csv",
                contract["report_name"],
                "provenance.json", "preprocessing.json",
                *[f"repeat_{current}/completion.json" for current in expected_repeats],
            ]
            write_completion_marker(
                root,
                stage=contract["run_stage"],
                identity={"run_id": config["run_id"], "run_config_hash": run_hash},
                artifact_names=root_artifacts,
                completed_at=ended_at,
                schema_version=schema_version,
            )
            if not valid_completion_marker(root, stage=contract["run_stage"]):
                os.unlink(completion_path)
                raise RuntimeError("Final artifact chain failed validation")
            logger.info(
                "RUN COMPLETE: Acc=%.2f%% +/- %.2f%%, F1=%.2f%% +/- %.2f%%; "
                "report=%s; marker=%s",
                aggregate["accuracy_mean"], aggregate["accuracy_std"],
                aggregate["macro_f1_mean"], aggregate["macro_f1_std"],
                os.path.join(root, contract["report_name"]), completion_path,
            )
            return {"schema_version": schema_version, "status": "complete", **aggregate}
        except BaseException as exc:
            _remove_stale_staging(root)
            state.update({
                "status": "failed",
                "updated_at": _utc_now(),
                "active_seconds": active_before + time.time() - invocation_start,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "mined_during_run": mined_during_run,
            })
            atomic_json_write(state_path, state)
            logger.error("RUN FAILED: state=%s; error=%s", state_path, exc)
            raise


def run_fully_inductive_pipeline(config, *, split_dir=None, repeat=None,
                                 force_remine=False, mode="full"):
    """Run the fully inductive protocol: mining sees training hypergraphs only."""
    if config.get("vocabulary_access", "train_only") != "train_only":
        raise ValueError(
            "run_fully_inductive_pipeline requires vocabulary_access=train_only"
        )
    return _run_vocabulary_access_pipeline(
        config, split_dir=split_dir, repeat=repeat,
        force_remine=force_remine, mode=mode,
    )


def run_transductive_pipeline(config, *, split_dir=None, repeat=None,
                              force_remine=False, mode="full"):
    """Mine all hypergraphs without labels; fit the classifier on training hypergraphs."""
    config = copy.deepcopy(config)
    config["vocabulary_access"] = "all_unlabeled"
    config["pipeline_mode"] = "transductive_vocabulary"
    return _run_vocabulary_access_pipeline(
        config, split_dir=split_dir, repeat=repeat,
        force_remine=force_remine, mode=mode,
    )
