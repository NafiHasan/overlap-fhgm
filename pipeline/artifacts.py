"""Deterministic artifacts for validated vocabulary-access protocols."""

import hashlib
import gzip
import json
import os
import math
import statistics
import tempfile
from pathlib import Path
from contextlib import contextmanager

import fcntl


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compact_json(value):
    """Return stable compact JSON suitable for a CSV cell or hash input."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def mining_config_hash(config, *, dataset, label_mode, manifest_hash,
                       source_hash):
    """Cache key of one repeat's mining: settings, data, partition and miner source."""
    keys = ("base_sup", "sup_discount", "max_phases", "min_sup_floor",
            "min_coverage", "max_depth")
    from .provenance import implementation_identity
    payload = {key: config.get(key) for key in keys}
    payload["vocabulary_access"] = config.get("vocabulary_access", "train_only")
    payload["implementation"] = implementation_identity()["mining"]
    payload["require_clean_source"] = config.get("require_clean_source", False)
    # Normalized, so an unset option and its default give the same key.
    from mining.fhgm import resolve_pattern_options
    overlap_binning, exhaustive_depth, exhaustive_min_graphs = resolve_pattern_options(config)
    payload["overlap_labels"] = overlap_binning or "none"
    payload["exhaustive_depth"] = exhaustive_depth
    payload["exhaustive_min_graphs"] = exhaustive_min_graphs
    payload.update(dataset=dataset, label_mode=label_mode,
                   train_manifest_hash=manifest_hash, source_hypergraph_hash=source_hash)
    return canonical_json_hash(payload)


def atomic_json_write(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path) or ".", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_text_write(path, value):
    """Write text through a sibling temporary file followed by replacement."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path) or ".", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_gzip_json_write(path, value):
    """Atomically store deterministic gzip-compressed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        encoded = (compact_json(value) + "\n").encode("utf-8")
        with open(temp_path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(encoded)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def file_lock(path):
    """Hold an exclusive process lock for a run or content-addressed cache."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is already writing this artifact: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def artifact_hashes(directory, names):
    """Hash an explicit artifact list relative to one directory."""
    hashes = {}
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required artifact is missing: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def write_completion_marker(directory, *, stage, identity, artifact_names, completed_at,
                            schema_version=2):
    """Write the authoritative success marker after every artifact exists."""
    marker = {
        "schema_version": schema_version,
        "status": "complete",
        "stage": stage,
        "identity": identity,
        "completed_at": completed_at,
        "artifacts": artifact_hashes(directory, artifact_names),
    }
    atomic_json_write(os.path.join(directory, "completion.json"), marker)
    return marker


CACHE_ARTIFACTS = (
    "split.json", "patterns.json", "patterns.txt", "mining_report.md",
    "mining_metadata.json", "cover_freq.txt", "coverage_metadata.json",
    "train_coverage.json.gz", "provenance.json", "preprocessing.json",
)
TRANSDUCTIVE_CACHE_ARTIFACTS = (
    "split.json", "patterns.json", "patterns.txt", "mining_report.md",
    "mining_metadata.json", "cover_freq.txt", "coverage_metadata.json",
    "discovery_coverage.json.gz", "provenance.json", "preprocessing.json",
)
REPEAT_ARTIFACTS = (
    "split.json", "mining_reference.json", "metrics.json", "provenance.json",
    "predictions.json.gz", "selected_patterns.json",
)
RUN_ARTIFACTS = (
    "run_state.json", "experiment.json", "repeats.csv", "inductive_report.md",
    "provenance.json", "preprocessing.json",
)
TRANSDUCTIVE_RUN_ARTIFACTS = (
    "run_state.json", "experiment.json", "repeats.csv",
    "transductive_report.md", "provenance.json", "preprocessing.json",
)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def valid_completion_marker(directory, *, stage=None, identity=None):
    """Validate fixed stage contracts and the complete run -> repeat -> cache chain."""
    path = os.path.join(directory, "completion.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            marker = json.load(handle)
        if marker.get("schema_version") != 2 or marker.get("status") != "complete":
            return False
        if stage is not None and marker.get("stage") != stage:
            return False
        if identity is not None:
            recorded = marker.get("identity", {})
            if any(recorded.get(key) != value for key, value in identity.items()):
                return False
        hashes = marker.get("artifacts")
        if not isinstance(hashes, dict) or not hashes:
            return False
        if any(not isinstance(name, str) or Path(name).is_absolute() or
               ".." in Path(name).parts for name in hashes):
            return False
        required = {
            "inductive_mining_cache": CACHE_ARTIFACTS,
            "transductive_mining_cache": TRANSDUCTIVE_CACHE_ARTIFACTS,
            "inductive_classification_repeat": REPEAT_ARTIFACTS,
            "transductive_classification_repeat": REPEAT_ARTIFACTS,
            "fully_inductive_run": RUN_ARTIFACTS,
            "transductive_vocabulary_run": TRANSDUCTIVE_RUN_ARTIFACTS,
        }.get(marker.get("stage"), ())
        if not set(required).issubset(hashes):
            return False
        if not all(
            os.path.isfile(os.path.join(directory, name)) and
            sha256_file(os.path.join(directory, name)) == expected_hash
            for name, expected_hash in hashes.items()
        ):
            return False
        return _valid_stage_links(Path(directory), marker)
    except (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError):
        return False


def _valid_stage_links(root, marker):
    stage, identity = marker["stage"], marker["identity"]
    if stage in {"inductive_mining_cache", "transductive_mining_cache"}:
        expected_access = (
            "train_only" if stage == "inductive_mining_cache"
            else "all_unlabeled"
        )
        metadata = _read_json(root / "mining_metadata.json")
        provenance = _read_json(root / "provenance.json")
        split = _read_json(root / "split.json")
        coverage = _read_json(root / "coverage_metadata.json")
        vocabulary = _read_json(root / "patterns.json")
        preprocessing = _read_json(root / "preprocessing.json")
        if metadata.get("require_clean_source") and (
                not provenance.get("git_commit") or provenance.get("git_source_dirty") is not False):
            return False
        return (
            metadata.get("schema_version") == 2 and
            metadata.get("vocabulary_access", "train_only") == expected_access and
            metadata["mining_config_hash"] == identity["mining_hash"] and
            metadata["implementation"] == provenance["implementation"]["mining"] and
            metadata["split_manifest_hash"] == canonical_json_hash(split) and
            metadata["source_hypergraph_hash"] == split["source"]["hypergraph_sha256"] and
            preprocessing["hypergraph_sha256"] == split["source"]["hypergraph_sha256"] and
            preprocessing["labels_sha256"] == split["source"]["labels_sha256"] and
            split["repeat"] == identity["repeat"] and
            vocabulary["metadata"]["mining_config_hash"] == identity["mining_hash"] and
            coverage.get("schema_version") == 2 and
            coverage.get("coverage_semantics") == "binary_presence" and
            coverage.get("vocabulary_access", "train_only") == expected_access and
            coverage["split_manifest_hash"] == metadata["split_manifest_hash"] and
            coverage["vocabulary_hash"] == canonical_json_hash([
                record["extensions"] for record in vocabulary["patterns"]
            ])
        )
    if stage in {
            "inductive_classification_repeat",
            "transductive_classification_repeat",
    }:
        expected_access = (
            "train_only" if stage == "inductive_classification_repeat"
            else "all_unlabeled"
        )
        expected_cache_stage = (
            "inductive_mining_cache" if expected_access == "train_only"
            else "transductive_mining_cache"
        )
        metrics = _read_json(root / "metrics.json")
        if metrics.get("schema_version") != 2:
            return False
        reference = _read_json(root / "mining_reference.json")
        split = _read_json(root / "split.json")
        provenance = _read_json(root / "provenance.json")
        cache = root / reference["cache_dir"]
        cache_stage = reference.get("cache_stage", "inductive_mining_cache")
        if cache_stage != expected_cache_stage or not valid_completion_marker(
                cache, stage=cache_stage,
                identity={"mining_hash": reference["mining_hash"],
                          "repeat": identity["repeat"]}):
            return False
        cache_marker = _read_json(cache / "completion.json")
        cache_metadata = _read_json(cache / "mining_metadata.json")
        expected = {**cache_marker["artifacts"],
                    "completion.json": sha256_file(cache / "completion.json")}
        selected = _read_json(root / "selected_patterns.json")
        predictions = load_gzip_json(root / "predictions.json.gz")
        pattern_ids = selected["pattern_ids"]
        if (not isinstance(pattern_ids, list) or len(set(pattern_ids)) != len(pattern_ids) or
                len(pattern_ids) != metrics["selected_patterns"] or
                any(type(value) is not int or not 0 <= value < metrics["patterns"]
                    for value in pattern_ids) or
                selected["vocabulary_hash"] != reference["vocabulary_hash"]):
            return False
        for partition in ("validation", "test"):
            prediction = predictions[partition]
            if (prediction["graph_ids"] != split["indices"][partition] or
                    len(prediction["truth"]) != len(prediction["graph_ids"]) or
                    len(prediction["predicted"]) != len(prediction["graph_ids"])):
                return False
            truth, predicted = prediction["truth"], prediction["predicted"]
            if not truth:
                return False
            classes = metrics["num_classes"]
            if type(classes) is not int or classes < 2 or any(
                    type(value) is not int or not 0 <= value < classes
                    for value in truth + predicted):
                return False
            accuracy = 100 * sum(a == b for a, b in zip(truth, predicted)) / len(truth)
            key = "val_accuracy" if partition == "validation" else "accuracy"
            if not math.isclose(accuracy, metrics[key], abs_tol=1e-8):
                return False
            f1s = []
            for label in range(classes):
                true_positive = sum(a == b == label for a, b in zip(truth, predicted))
                denominator = truth.count(label) + predicted.count(label)
                f1s.append(2 * true_positive / denominator if denominator else 0)
            key = "val_macro_f1" if partition == "validation" else "macro_f1"
            if not math.isclose(100 * sum(f1s) / classes, metrics[key], abs_tol=1e-8):
                return False
        return (
            metrics.get("vocabulary_access", "train_only") == expected_access and
            reference.get("vocabulary_access", "train_only") == expected_access and
            reference["artifacts"] == expected and
            metrics["repeat"] == identity["repeat"] and
            metrics["run_config_hash"] == identity["run_config_hash"] and
            metrics["implementation"] == provenance["implementation"]["run"] and
            metrics["mining_hash"] == reference["mining_hash"] and
            metrics["split_manifest_hash"] == canonical_json_hash(split) ==
            cache_metadata["split_manifest_hash"] and
            cache_metadata["implementation"] == provenance["implementation"]["mining"]
        )
    if stage in {
            "fully_inductive_run", "transductive_vocabulary_run",
    }:
        expected_access = (
            "train_only" if stage == "fully_inductive_run"
            else "all_unlabeled"
        )
        repeat_stage = (
            "inductive_classification_repeat" if stage == "fully_inductive_run"
            else "transductive_classification_repeat"
        )
        experiment = _read_json(root / "experiment.json")
        state = _read_json(root / "run_state.json")
        provenance = _read_json(root / "provenance.json")
        preprocessing = _read_json(root / "preprocessing.json")
        if experiment.get("schema_version") != 2 or state.get("schema_version") != 2:
            return False
        if experiment["config"].get("require_clean_source"):
            invocations = experiment["invocations"]
            if not invocations or any(
                    not invocation["provenance"].get("git_commit") or
                    invocation["provenance"].get("git_source_dirty") is not False
                    for invocation in invocations):
                return False
        count = experiment["config"]["num_repeats"]
        if type(count) is not int or not 1 <= count <= 5:
            return False
        repeats = list(range(count))
        if (experiment["status"] != "complete" or state["status"] != "complete" or
                experiment.get("vocabulary_access", "train_only") != expected_access or
                experiment["config"].get("vocabulary_access", "train_only") != expected_access or
                state.get("vocabulary_access", "train_only") != expected_access or
                experiment["run_id"] != identity["run_id"] or
                experiment["run_config_hash"] != identity["run_config_hash"] or
                state["run_config_hash"] != identity["run_config_hash"] or
                state["completed_repeats"] != repeats or
                experiment["provenance"] != provenance or
                experiment["source_hashes"] != state["source_hashes"] or
                experiment["source_hashes"] != {
                    "hypergraphs": preprocessing["hypergraph_sha256"],
                    "labels": preprocessing["labels_sha256"],
                }):
            return False
        records = []
        for repeat in repeats:
            name = f"repeat_{repeat}/completion.json"
            if name not in marker["artifacts"] or not valid_completion_marker(
                    root / f"repeat_{repeat}",
                    stage=repeat_stage,
                    identity={**identity, "repeat": repeat}):
                return False
            metrics = _read_json(root / f"repeat_{repeat}" / "metrics.json")
            split = _read_json(root / f"repeat_{repeat}" / "split.json")
            records.append(metrics)
            if (metrics["implementation"] != provenance["implementation"]["run"] or
                    metrics["split_manifest_hash"] != experiment["split_manifest_hashes"][str(repeat)] or
                    metrics["mining_hash"] != experiment["mining_hashes"][str(repeat)] or
                    split["dataset"] != experiment["config"]["_dataset"] or
                    split["label_mode"] != experiment["config"]["_label_mode"] or
                    split["source"]["hypergraph_sha256"] != experiment["source_hashes"]["hypergraphs"] or
                    split["source"]["labels_sha256"] != experiment["source_hashes"]["labels"]):
                return False
        for metric in ("accuracy", "macro_f1", "val_accuracy", "val_macro_f1"):
            values = [record[metric] for record in records]
            if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
                return False
            for suffix, value in (("mean", statistics.mean(values)),
                                  ("std", statistics.pstdev(values))):
                if not math.isclose(value, experiment["aggregate"][f"{metric}_{suffix}"], abs_tol=1e-8):
                    return False
        return True
    return True


def valid_repeat_cache(repeat_dir, *, mining_hash, manifest_hash, source_hash,
                       vocabulary_access="train_only"):
    metadata_path = os.path.join(repeat_dir, "mining_metadata.json")
    required = ("mining_metadata.json", "patterns.json", "patterns.txt", "mining_report.md")
    if not all(os.path.isfile(os.path.join(repeat_dir, name)) for name in required):
        return False
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        common_valid = (
            metadata.get("mining_config_hash") == mining_hash and
            metadata.get("split_manifest_hash") == manifest_hash and
            metadata.get("source_hypergraph_hash") == source_hash and
            metadata.get("vocabulary_access", "train_only") == vocabulary_access
        )
        if not common_valid:
            return False
        if metadata.get("schema_version") != 2:
            return False
        return valid_completion_marker(
            repeat_dir,
            stage=("inductive_mining_cache"
                   if vocabulary_access == "train_only"
                   else "transductive_mining_cache"),
            identity={"mining_hash": mining_hash},
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False
