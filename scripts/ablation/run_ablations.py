"""Run frozen one-factor-at-a-time ablations around a completed base run.

A suite in ``scripts/ablation/suites.json`` names a completed base campaign
(the main run) and studies whose variants each change one setting.
Every variant reruns the validated inductive pipeline with its own run ID, so
mining caches are reused whenever the mining settings are unchanged.

    python scripts/ablation/run_ablations.py plan   --suite SUITE
    python scripts/ablation/run_ablations.py init   --suite SUITE
    python scripts/ablation/run_ablations.py run    --suite SUITE --dataset rhg_3
    python scripts/ablation/run_ablations.py status --suite SUITE

Use ``scripts/ablation/run_ablations.sh``, which matches the execution
environment of the base campaign so its mining caches stay valid. Run one
process per dataset; variants of one dataset share caches and run in order.
"""
import argparse
import copy
import csv
import json
import logging
from pathlib import Path
import re
import statistics
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SPEC_PATH = REPO_ROOT / "scripts" / "ablation" / "suites.json"
LABEL_MODE = "composite"
SPLIT_PROTOCOL_DIR = "dhgbench_torch_2_2_2"
SUPPORTED_DATASETS = (
    "imdb_dir_form", "imdb_dir_genre", "steam_player",
    "twitter_friend", "rhg_10", "rhg_3",
)
# Settings that define the protocol, data or bookkeeping rather than a factor.
PROTECTED_SETTINGS = {
    "run_id", "pipeline_mode", "vocabulary_access",
    "paths", "_dataset", "_label_mode", "domains", "num_classes",
    "split_manifest_dir", "split_protocol", "train_ratio", "validation_ratio",
    "num_repeats", "random_state", "require_clean_source",
    "inductive_mining_cache_dir", "transductive_mining_cache_dir",
}
PRIVATE_KEYS = ("_invocation_provenance", "_preprocessing", "experiment_profile")
ID_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9_]*")
STUDY_TYPES = {"component", "sweep"}
# Settings that the base configuration leaves unset (at their default) and a
# variant may still change.
OPTIONAL_KEYS = {"mined_min_depth", "mined_min_train_graphs", "mined_selection",
                 "supervised_layer_norm"}
TIER_ORDER = {"control": 0, "classification": 1, "mining": 2}

logger = logging.getLogger(__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------

def load_suite(suite_id, spec_path=SPEC_PATH):
    """Return the validated suite specification for ``suite_id``."""
    spec = read_json(spec_path)
    if spec.get("schema_version") != 1 or not isinstance(spec.get("suites"), dict):
        raise ValueError(f"{spec_path} must contain schema_version 1 and a suites mapping")
    if suite_id not in spec["suites"]:
        raise ValueError(f"Unknown suite {suite_id!r}; defined: {sorted(spec['suites'])}")
    suite = copy.deepcopy(spec["suites"][suite_id])
    validate_suite(suite_id, suite)
    return suite


def validate_suite(suite_id, suite):
    """Reject malformed or ambiguous suites before any data is touched."""
    if not ID_PATTERN.fullmatch(suite_id):
        raise ValueError("Suite IDs use upper-case letters, digits and underscores")
    for key in ("base_run", "datasets", "repeats", "studies"):
        if key not in suite:
            raise ValueError(f"Suite {suite_id} is missing {key!r}")
    if suite.get("protocol", "inductive") != "inductive":
        raise ValueError("Ablations support the inductive protocol only")
    datasets = suite["datasets"]
    if (not datasets or len(set(datasets)) != len(datasets) or
            any(dataset not in SUPPORTED_DATASETS for dataset in datasets)):
        raise ValueError("Choose unique supported datasets")
    repeats = suite["repeats"]
    if repeats != list(range(len(repeats))) or not 1 <= len(repeats) <= 5:
        raise ValueError("Repeats must be a prefix of 0..4")
    study_ids = set()
    for study in suite["studies"]:
        study_id = study.get("id", "")
        if not ID_PATTERN.fullmatch(study_id) or study_id in study_ids or study_id == "CONTROL":
            raise ValueError(f"Invalid or duplicate study ID {study_id!r}")
        study_ids.add(study_id)
        if study.get("type") not in STUDY_TYPES:
            raise ValueError(f"Study {study_id}: type must be component or sweep")
        if not study.get("title"):
            raise ValueError(f"Study {study_id}: title is required")
        variants = study.get("variants", [])
        variant_ids = [variant.get("id", "") for variant in variants]
        if len(variants) < 2 or len(set(variant_ids)) != len(variant_ids):
            raise ValueError(f"Study {study_id}: needs at least two uniquely named variants")
        controls = [variant for variant in variants if variant.get("control")]
        if len(controls) != 1:
            raise ValueError(f"Study {study_id}: exactly one variant must be the control")
        for variant in variants:
            name = f"{study_id}/{variant.get('id')}"
            if not ID_PATTERN.fullmatch(variant.get("id", "")):
                raise ValueError(f"{name}: invalid variant ID")
            overrides = variant.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError(f"{name}: overrides must be an object")
            if variant.get("control") and overrides:
                raise ValueError(f"{name}: the control must be the unchanged base run")
            if not variant.get("control") and not overrides:
                raise ValueError(f"{name}: a non-control variant must change a setting")
            protected = PROTECTED_SETTINGS.intersection(overrides)
            if protected:
                raise ValueError(f"{name}: may not change {sorted(protected)}")
            if study["type"] == "sweep":
                if "x" not in variant:
                    raise ValueError(f"{name}: sweep variants need an x value")
                parameter = study.get("parameter")
                if not parameter:
                    raise ValueError(f"Study {study_id}: sweeps need a parameter")
                if parameter in overrides and overrides[parameter] != variant["x"]:
                    raise ValueError(f"{name}: x must equal the swept {parameter} value")
            elif not variant.get("label"):
                raise ValueError(f"{name}: component variants need a label")


def run_id_for(suite_id, study_id, variant):
    """Variants share one control run; every other variant owns a run ID."""
    if variant.get("control"):
        return f"{suite_id}_CONTROL"
    return f"{suite_id}_{study_id}_{variant['id']}"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def result_base(results_dir, dataset):
    return Path(results_dir) / dataset / LABEL_MODE


def base_record_path(results_dir, dataset, base_run):
    return result_base(results_dir, dataset) / "inductive" / base_run / "experiment.json"


def localize(config, dataset, repo_root, results_dir):
    """Point a recorded configuration at this checkout's data and results."""
    repo_root, results = Path(repo_root).resolve(), result_base(results_dir, dataset).resolve()
    data = repo_root / "data" / dataset
    config["paths"] = {
        "hypergraph_data_dir": str(data / LABEL_MODE),
        "labels_file": str(data / "hg_label.txt"),
        "coverage_file": str(results / "cover_freq.txt"),
    }
    config["split_manifest_dir"] = str(data / "splits" / SPLIT_PROTOCOL_DIR / LABEL_MODE)
    config["inductive_mining_cache_dir"] = str(results / "inductive_mining_cache")
    config["transductive_mining_cache_dir"] = str(results / "transductive_mining_cache")
    return config


def base_config(suite, dataset, repo_root, results_dir):
    """Recorded base configuration, localized and checked, plus its record."""
    path = base_record_path(results_dir, dataset, suite["base_run"])
    record = read_json(path)
    if record.get("status") != "complete":
        raise ValueError(f"Base run is not complete: {path}")
    config = copy.deepcopy(record["config"])
    for key in PRIVATE_KEYS:
        config.pop(key, None)
    if (config.get("_dataset") != dataset or config.get("_label_mode") != LABEL_MODE or
            config.get("pipeline_mode") != "fully_inductive" or
            config.get("vocabulary_access", "train_only") != "train_only"):
        raise ValueError(f"Base record is not a fully inductive {dataset} run: {path}")
    config["num_repeats"] = len(suite["repeats"])
    config["random_state"] = 0
    return localize(config, dataset, repo_root, results_dir), record, path


def variant_config(base, suite_id, study, variant):
    """Apply one variant to the base configuration and normalize it."""
    from pipeline.inductive import _validate_config

    unknown = set(variant.get("overrides", {})) - set(base) - OPTIONAL_KEYS
    if unknown:
        raise ValueError(
            f"{study['id']}/{variant['id']}: settings absent from the base record: {sorted(unknown)}"
        )
    config = copy.deepcopy(base)
    config.update(copy.deepcopy(variant.get("overrides", {})))
    config["run_id"] = run_id_for(suite_id, study["id"], variant)
    _validate_config(config, None)
    return config


def tier_of(base, config):
    from pipeline.inductive import _effective_config_sections

    same = (_effective_config_sections(base)["mining_config"] ==
            _effective_config_sections(config)["mining_config"])
    return "classification" if same else "mining"


def _without_paths(config):
    return {key: value for key, value in config.items()
            if key not in {"paths", "split_manifest_dir", "inductive_mining_cache_dir",
                           "transductive_mining_cache_dir"}}


def resolve_runs(suite_id, suite, dataset, repo_root, results_dir):
    """Map every run ID of one dataset to its config, tier and study uses."""
    from pipeline.inductive import _run_config_hash, _validate_config

    base, record, path = base_config(suite, dataset, repo_root, results_dir)
    normalized_base = copy.deepcopy(base)
    normalized_base["run_id"] = f"{suite_id}_CONTROL"
    _validate_config(normalized_base, None)
    runs = {}
    for study in suite["studies"]:
        for variant in study["variants"]:
            run_id = run_id_for(suite_id, study["id"], variant)
            if run_id in runs:
                runs[run_id]["uses"].append([study["id"], variant["id"]])
                continue
            config = variant_config(base, suite_id, study, variant)
            tier = "control" if variant.get("control") else tier_of(normalized_base, config)
            runs[run_id] = {
                "tier": tier,
                "config": config,
                "run_config_hash": _run_config_hash(config),
                "uses": [[study["id"], variant["id"]]],
            }
    seen = {}
    for run_id, run in runs.items():
        key = json.dumps(_without_paths({**run["config"], "run_id": None}), sort_keys=True)
        if key in seen:
            raise ValueError(f"{dataset}: {run_id} duplicates {seen[key]}; merge the variants")
        seen[key] = run_id
    return runs, record, path


def ordered_run_ids(runs):
    return sorted(runs, key=lambda run_id: (TIER_ORDER[runs[run_id]["tier"]], run_id))


# --------------------------------------------------------------------------
# Frozen manifest
# --------------------------------------------------------------------------

def manifest_path(results_dir, suite_id):
    return Path(results_dir) / "ablation" / suite_id / "suite.json"


def build_manifest(suite_id, suite, *, repo_root=REPO_ROOT, results_dir=None,
                   require_clean_source=False):
    """Resolve and audit every run of the suite without training anything."""
    from pipeline.artifacts import canonical_json_hash, sha256_file
    from pipeline.provenance import capture_provenance
    from scripts.run_campaign import validated_splits

    results_dir = Path(results_dir or repo_root / "results").resolve()
    provenance = capture_provenance()
    if require_clean_source and provenance["git_source_dirty"] is not False:
        raise ValueError("Committed, clean source is required for this suite")
    entries = {}
    for dataset in suite["datasets"]:
        runs, record, path = resolve_runs(suite_id, suite, dataset, repo_root, results_dir)
        control = runs[f"{suite_id}_CONTROL"]["config"]
        graphs, _labels, splits = validated_splits(control, suite["repeats"])
        split_hashes = {str(r): split.manifest_hash for r, split in splits.items()}
        recorded = {str(k): v for k, v in record.get("split_manifest_hashes", {}).items()}
        if any(recorded.get(key) != value for key, value in split_hashes.items()):
            raise ValueError(f"{dataset}: installed split manifests differ from the base run")
        entries[dataset] = {
            "base_record": str(path),
            "base_record_sha256": sha256_file(path),
            "base_run_config_hash": record.get("run_config_hash"),
            "base_implementation": record.get("provenance", {}).get("implementation", {}),
            "graph_count": len(graphs),
            "source_hashes": {
                "hypergraphs": sha256_file(Path(control["paths"]["hypergraph_data_dir"]) /
                                           (control["domains"][0] + ".txt")),
                "labels": sha256_file(control["paths"]["labels_file"]),
            },
            "split_hashes": split_hashes,
            "runs": runs,
        }
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "spec": suite,
        "spec_sha256": canonical_json_hash(suite),
        "base_run": suite["base_run"],
        "protocol": "fully_inductive",
        "datasets": list(suite["datasets"]),
        "repeats": list(suite["repeats"]),
        "implementation": provenance["implementation"],
        "initial_provenance": provenance,
        "entries": entries,
        "created_at": _utc_now(),
        "selection_policy": ("One factor at a time around a frozen base; validation "
                             "metrics only for any choice; test metrics descriptive."),
    }


def identity_report(manifest):
    """Explain whether base caches and base results should be reproduced."""
    current = manifest["implementation"]
    lines = []
    for dataset, entry in manifest["entries"].items():
        base = entry.get("base_implementation", {})
        lines.append(
            f"  {dataset:15s} mining caches reusable: "
            f"{'yes' if base.get('mining') == current['mining'] else 'NO (will re-mine)'}; "
            f"control reproduces base exactly: "
            f"{'expected' if base.get('run') == current['run'] else 'not guaranteed'}"
        )
    return "\n".join(lines)


def validate_manifest(manifest, suite, *, repo_root=REPO_ROOT, results_dir=None):
    """Bind a resumed run to the frozen spec, base records, splits and code."""
    from pipeline.artifacts import canonical_json_hash, sha256_file
    from pipeline.provenance import require_unchanged_source

    if canonical_json_hash(suite) != manifest["spec_sha256"]:
        raise ValueError(
            f"Suite {manifest['suite_id']} changed in suites.json since init; "
            "define a new suite ID (completed runs are kept)"
        )
    require_unchanged_source({"implementation": manifest["implementation"]})
    results_dir = Path(results_dir or repo_root / "results").resolve()
    current = {}
    for dataset, entry in manifest["entries"].items():
        runs, _record, path = resolve_runs(
            manifest["suite_id"], suite, dataset, repo_root, results_dir)
        if sha256_file(path) != entry["base_record_sha256"]:
            raise ValueError(f"Base record changed: {path}")
        for run_id, frozen in entry["runs"].items():
            run = runs.get(run_id)
            if (run is None or run["run_config_hash"] != frozen["run_config_hash"] or
                    _without_paths(run["config"]) != _without_paths(frozen["config"])):
                raise ValueError(f"{dataset}/{run_id}: effective configuration changed since init")
        current[dataset] = runs
    return current


# --------------------------------------------------------------------------
# Execution and status
# --------------------------------------------------------------------------

def run_directory(config):
    return Path(config["paths"]["coverage_file"]).parent / "inductive" / config["run_id"]


def run_state(run):
    """Return (state, completed repeats) for a pipeline run."""
    from pipeline.artifacts import valid_completion_marker

    root = run_directory(run["config"])
    if (root / "completion.json").exists():
        ok = valid_completion_marker(
            str(root), stage="fully_inductive_run",
            identity={"run_id": run["config"]["run_id"],
                      "run_config_hash": run["run_config_hash"]})
        return ("complete" if ok else "invalid"), list(range(run["config"]["num_repeats"]))
    state_path = root / "run_state.json"
    if state_path.exists():
        return "partial", read_json(state_path).get("completed_repeats", [])
    return "missing", []


def execute_run(manifest, dataset, run_id, run, repeat=None, no_mining=False):
    from pipeline.inductive import run_fully_inductive_pipeline

    config = copy.deepcopy(run["config"])
    # "classification" fails on a missing mining cache instead of mining.
    run_fully_inductive_pipeline(config, mode="classification" if no_mining else "full",
                                 repeat=repeat)


def select_runs(runs, *, studies=None, variants=None, tiers=None):
    wanted = []
    for run_id in ordered_run_ids(runs):
        run = runs[run_id]
        uses = run["uses"]
        if studies and not any(study in studies for study, _ in uses):
            if run["tier"] != "control":
                continue
        if variants and not any(variant in variants for _, variant in uses):
            if run["tier"] != "control":
                continue
        if tiers and run["tier"] not in tiers and run["tier"] != "control":
            continue
        wanted.append(run_id)
    return wanted


def estimate_seconds(record_path, runs):
    """Rough cost from the base run's recorded per-repeat medians."""
    repeats_csv = Path(record_path).parent / "repeats.csv"
    if not repeats_csv.exists():
        return None
    with repeats_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mining = statistics.median(float(row["mining_seconds"]) for row in rows)
    classify = statistics.median(float(row["classification_seconds"]) for row in rows)
    total = 0.0
    for run in runs.values():
        per_repeat = classify + (mining if run["tier"] == "mining" else 0.0)
        total += per_repeat * run["config"]["num_repeats"]
    return total


def print_plan(suite_id, suite, *, repo_root, results_dir):
    print(f"Suite {suite_id}: base run {suite['base_run']}, repeats {suite['repeats']}")
    for dataset in suite["datasets"]:
        runs, _record, path = resolve_runs(suite_id, suite, dataset, repo_root, results_dir)
        chosen = {run_id: runs[run_id] for run_id in select_runs(runs)}
        estimate = estimate_seconds(path, chosen)
        hours = "unknown" if estimate is None else f"{estimate / 3600:.1f} h"
        print(f"\n{dataset}: {len(chosen)} runs, base-cost estimate {hours} "
              "(mining variants can cost much more than the base)")
        for run_id in ordered_run_ids(runs):
            run = runs[run_id]
            uses = ", ".join(f"{study}/{variant}" for study, variant in run["uses"])
            print(f"  {run['tier']:14s} {run_id:28s} {uses}")


def print_status(manifest, runs_by_dataset):
    for dataset, runs in runs_by_dataset.items():
        print(f"\n{dataset}")
        for run_id in ordered_run_ids(runs):
            state, done = run_state(runs[run_id])
            print(f"  {state:9s} {len(done)}/{len(manifest['repeats'])} "
                  f"{runs[run_id]['tier']:14s} {run_id}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("plan", "init", "run", "status"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS,
                        help="Required for run; one process per dataset")
    parser.add_argument("--study", nargs="+", help="Limit to these study IDs (control always included)")
    parser.add_argument("--variant", nargs="+", help="Limit to these variant IDs")
    parser.add_argument("--tier", nargs="+", choices=("classification", "mining"),
                        help="Limit to these cost tiers (control always included)")
    parser.add_argument("--repeat", type=int, choices=range(5),
                        help="Run one repeat only; rerun later without it to finish")
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument("--no-mining", action="store_true",
                        help="Fail instead of mining when a repeat's mining cache is missing")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from pipeline.artifacts import atomic_json_write, file_lock

    suite = load_suite(args.suite, args.spec)
    results_dir = args.results_dir.resolve()
    path = manifest_path(results_dir, args.suite)

    if args.stage == "plan":
        print_plan(args.suite, suite, repo_root=REPO_ROOT, results_dir=results_dir)
        return 0

    if args.stage == "init":
        if path.exists():
            raise FileExistsError(f"Suite already initialized: {path}")
        manifest = build_manifest(args.suite, suite, repo_root=REPO_ROOT,
                                  results_dir=results_dir,
                                  require_clean_source=args.require_clean_source)
        for dataset, entry in manifest["entries"].items():
            for run_id, run in entry["runs"].items():
                if run_directory(run["config"]).exists():
                    raise FileExistsError(f"Output already exists for a new suite: "
                                          f"{run_directory(run['config'])}")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, manifest)
        print(f"Initialized {path}")
        print(identity_report(manifest))
        return 0

    if not path.exists():
        raise FileNotFoundError(f"Suite {args.suite} is not initialized; run the init stage first")
    manifest = read_json(path)
    if manifest.get("suite_id") != args.suite:
        raise ValueError("Suite identity mismatch")
    runs_by_dataset = validate_manifest(manifest, suite, repo_root=REPO_ROOT,
                                        results_dir=results_dir)

    if args.stage == "status":
        print_status(manifest, runs_by_dataset)
        return 0

    if args.dataset is None:
        raise ValueError("run needs --dataset; start one process per dataset")
    if args.dataset not in runs_by_dataset:
        raise ValueError(f"{args.dataset} is not part of suite {args.suite}")
    from utils.logger import setup_logging
    log_path = setup_logging("ablation", dataset=args.dataset, run_id=args.suite)
    logger.info("Ablation suite %s on %s; detailed log: %s", args.suite, args.dataset, log_path)
    runs = runs_by_dataset[args.dataset]
    selected = select_runs(runs, studies=args.study, variants=args.variant,
                           tiers=args.tier)
    lock = path.parent / f"{args.dataset}.lock"
    with file_lock(str(lock)):
        for number, run_id in enumerate(selected, 1):
            run = runs[run_id]
            state, done = run_state(run)
            if state == "complete" or (args.repeat is not None and args.repeat in done):
                logger.info("[%d/%d] %s: already complete; skipping", number, len(selected), run_id)
                continue
            if state == "invalid":
                raise ValueError(f"Invalid completed run, inspect before continuing: "
                                 f"{run_directory(run['config'])}")
            logger.info("[%d/%d] %s (%s): starting", number, len(selected), run_id, run["tier"])
            execute_run(manifest, args.dataset, run_id, run, repeat=args.repeat,
                        no_mining=args.no_mining)
    logger.info("Ablation suite %s on %s: requested runs finished", args.suite, args.dataset)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
