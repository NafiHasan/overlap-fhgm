"""Run a frozen grid of inductive configurations and select on validation only.

A grid in ``scripts/grid_search/grids.json`` names a completed base campaign,
fixed overrides, mining settings and classification axes.
Every combination is a validated fully inductive run with its own run ID, so
mining caches are reused for every combination that shares mining settings.

    python scripts/grid_search/run_grid.py plan   --grid GRID
    python scripts/grid_search/run_grid.py init   --grid GRID
    python scripts/grid_search/run_grid.py run    --grid GRID --dataset rhg_3
    python scripts/grid_search/run_grid.py status --grid GRID
    python scripts/grid_search/run_grid.py report --grid GRID

Use ``scripts/grid_search/run_grid.sh``, which matches the execution
environment of the base campaign so its mining caches stay valid; its
``launch`` stage starts one tmux session per dataset. A missing mining cache
is mined once per repeat and then reused. Runs start in a fixed pseudo-random
order, the same on every dataset. The report selects one configuration for all
datasets by mean validation Macro-F1 (then validation accuracy); test values
are shown only for the selected configurations.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import logging
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation import run_ablations as ablations  # noqa: E402

SPEC_PATH = REPO_ROOT / "scripts" / "grid_search" / "grids.json"
LABEL_MODE = ablations.LABEL_MODE
SUPPORTED_DATASETS = ablations.SUPPORTED_DATASETS
ID_PATTERN = ablations.ID_PATTERN
logger = logging.getLogger(__name__)
METRICS = ("val_macro_f1", "val_accuracy", "test_macro_f1", "test_accuracy")
CSV_COLUMNS = {"val_macro_f1": "val_macro_f1", "val_accuracy": "val_accuracy",
               "test_macro_f1": "macro_f1", "test_accuracy": "accuracy"}

logger = logging.getLogger(__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------

def load_grid(grid_id, spec_path=SPEC_PATH):
    """Return the validated grid specification for ``grid_id``."""
    spec = read_json(spec_path)
    if spec.get("schema_version") != 1 or not isinstance(spec.get("grids"), dict):
        raise ValueError(f"{spec_path} must contain schema_version 1 and a grids mapping")
    if grid_id not in spec["grids"]:
        raise ValueError(f"Unknown grid {grid_id!r}; defined: {sorted(spec['grids'])}")
    grid = copy.deepcopy(spec["grids"][grid_id])
    validate_grid(grid_id, grid)
    return grid


def _check_keys(name, overrides, allowed):
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"{name}: overrides must be a non-empty object")
    protected = ablations.PROTECTED_SETTINGS.intersection(overrides)
    if protected:
        raise ValueError(f"{name}: may not change {sorted(protected)}")
    unknown = set(overrides) - set(allowed)
    if unknown:
        raise ValueError(f"{name}: settings outside this group: {sorted(unknown)}")


def validate_grid(grid_id, grid):
    """Reject malformed or ambiguous grids before any data is touched."""
    from pipeline.inductive import _CLASSIFICATION_CONFIG_KEYS, _MINING_CONFIG_KEYS

    if not ID_PATTERN.fullmatch(grid_id):
        raise ValueError("Grid IDs use upper-case letters, digits and underscores")
    for key in ("base_run", "datasets", "repeats", "mining", "axes"):
        if key not in grid:
            raise ValueError(f"Grid {grid_id} is missing {key!r}")
    datasets = grid["datasets"]
    if (not datasets or len(set(datasets)) != len(datasets) or
            any(dataset not in SUPPORTED_DATASETS for dataset in datasets)):
        raise ValueError("Choose unique supported datasets")
    repeats = grid["repeats"]
    if repeats != list(range(len(repeats))) or not 1 <= len(repeats) <= 5:
        raise ValueError("Repeats must be a prefix of 0..4")
    classification_keys = set(_CLASSIFICATION_CONFIG_KEYS) - {"num_classes"}
    if grid.get("fixed"):
        _check_keys("fixed", grid["fixed"], classification_keys)
    groups = [("mining", grid["mining"], set(_MINING_CONFIG_KEYS))]
    groups += [(axis.get("name", "?"), axis.get("options", []), classification_keys)
               for axis in grid["axes"]]
    tags, names = set(), set()
    for name, options, allowed in groups:
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate axis name {name!r}")
        names.add(name)
        if not options:
            raise ValueError(f"{name}: needs at least one option")
        keys = None
        for option in options:
            tag = option.get("tag", "")
            if not ID_PATTERN.fullmatch(tag) or tag in tags:
                raise ValueError(f"{name}: invalid or duplicate tag {tag!r}")
            tags.add(tag)
            _check_keys(f"{name}/{tag}", option.get("overrides"), allowed)
            if keys is None:
                keys = set(option["overrides"])
            elif set(option["overrides"]) != keys:
                raise ValueError(f"{name}: every option must set the same settings")
    axis_keys = [set(options[0]["overrides"]) for _name, options, _allowed in groups]
    for first, second in itertools.combinations(axis_keys, 2):
        if first & second:
            raise ValueError(f"Axes overlap on {sorted(first & second)}")
    if any(set(grid.get("fixed", {})) & keys for keys in axis_keys):
        raise ValueError("Fixed overrides may not overlap an axis")
def combinations(grid):
    """Return [(tags, overrides)] for the grid points to run, mining first."""
    groups = [grid["mining"]] + [axis["options"] for axis in grid["axes"]]
    points = list(itertools.product(*groups))
    result = []
    for options in points:
        overrides = copy.deepcopy(grid.get("fixed", {}))
        for option in options:
            overrides.update(copy.deepcopy(option["overrides"]))
        result.append((tuple(option["tag"] for option in options), overrides))
    return result


def run_order_key(run_id):
    """Fixed pseudo-random order, identical on every dataset."""
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()


def run_id_for(grid_id, tags):
    return "_".join((grid_id,) + tuple(tags))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def base_config(grid, dataset, results_dir):
    return ablations.base_config(
        {"base_run": grid["base_run"], "repeats": grid["repeats"]},
        dataset, REPO_ROOT, results_dir)


def resolve_runs(grid_id, grid, dataset, results_dir):
    """Map every run ID of one dataset to its tags, config and hash."""
    from pipeline.inductive import _run_config_hash, _validate_config

    base, record, path = base_config(grid, dataset, results_dir)
    runs, seen = {}, {}
    for tags, overrides in combinations(grid):
        unknown = set(overrides) - set(base)
        if unknown:
            raise ValueError(f"{tags}: settings absent from the base record: {sorted(unknown)}")
        config = copy.deepcopy(base)
        config.update(overrides)
        config["run_id"] = run_id_for(grid_id, tags)
        _validate_config(config, None)
        key = json.dumps(ablations._without_paths({**config, "run_id": None}), sort_keys=True)
        if key in seen:
            raise ValueError(f"{dataset}: {config['run_id']} duplicates {seen[key]}")
        seen[key] = config["run_id"]
        runs[config["run_id"]] = {"tags": list(tags), "config": config,
                                  "run_config_hash": _run_config_hash(config)}
    return runs, record, path


def mining_cache_status(grid, dataset, results_dir):
    """Per mining tag: repeats whose mining cache already exists here.

    The cache key includes the implementation identity and execution
    settings, so run this through ``run_grid.sh`` on the machine that mined.
    """
    from pipeline.artifacts import mining_config_hash, sha256_file
    from pipeline.inductive import _validate_config
    from pipeline.label_binning import effective_label_mode
    from scripts.run_campaign import validated_splits

    base, _record, _path = base_config(grid, dataset, results_dir)
    _graphs, _labels, splits = validated_splits(base, grid["repeats"])
    source_hash = sha256_file(Path(base["paths"]["hypergraph_data_dir"]) /
                              (base["domains"][0] + ".txt"))
    status = {}
    for mining in grid["mining"]:
        config = {**copy.deepcopy(base), **mining["overrides"], "run_id": "cache_check"}
        _validate_config(config, None)
        cached = []
        for repeat, split in splits.items():
            mining_hash = mining_config_hash(
                config, dataset=dataset,
                label_mode=effective_label_mode(LABEL_MODE, config),
                manifest_hash=split.manifest_hash, source_hash=source_hash)
            if (Path(config["inductive_mining_cache_dir"]) / mining_hash /
                    "completion.json").is_file():
                cached.append(repeat)
        status[mining["tag"]] = cached
    return status


# --------------------------------------------------------------------------
# Frozen manifest
# --------------------------------------------------------------------------

def manifest_path(results_dir, grid_id):
    return Path(results_dir) / "grid_search" / grid_id / "grid.json"


def build_manifest(grid_id, grid, *, results_dir):
    """Resolve and audit every run of the grid without training anything."""
    from pipeline.artifacts import canonical_json_hash, sha256_file
    from pipeline.provenance import capture_provenance
    from scripts.run_campaign import validated_splits

    provenance = capture_provenance()
    entries = {}
    for dataset in grid["datasets"]:
        runs, record, path = resolve_runs(grid_id, grid, dataset, results_dir)
        first = next(iter(runs.values()))["config"]
        _graphs, _labels, splits = validated_splits(first, grid["repeats"])
        split_hashes = {str(r): split.manifest_hash for r, split in splits.items()}
        recorded = {str(k): v for k, v in record.get("split_manifest_hashes", {}).items()}
        if any(recorded.get(key) != value for key, value in split_hashes.items()):
            raise ValueError(f"{dataset}: installed split manifests differ from the base run")
        # Configurations are rebuilt from the spec and the pinned base record.
        entries[dataset] = {
            "base_record": str(path),
            "base_record_sha256": sha256_file(path),
            "split_hashes": split_hashes,
            "runs": {run_id: {"tags": run["tags"], "run_config_hash": run["run_config_hash"]}
                     for run_id, run in runs.items()},
        }
    return {
        "schema_version": 1,
        "grid_id": grid_id,
        "spec": grid,
        "spec_sha256": canonical_json_hash(grid),
        "base_run": grid["base_run"],
        "protocol": "fully_inductive",
        "datasets": list(grid["datasets"]),
        "repeats": list(grid["repeats"]),
        "implementation": provenance["implementation"],
        "initial_provenance": provenance,
        "entries": entries,
        "created_at": _utc_now(),
        "selection_policy": ("Validation Macro-F1, then validation accuracy; test "
                             "metrics are reported only for the selected configurations."),
    }


def validate_manifest(manifest, grid, *, results_dir):
    """Bind a resumed run to the frozen spec, base records and code."""
    from pipeline.artifacts import canonical_json_hash, sha256_file
    from pipeline.provenance import require_unchanged_source

    if canonical_json_hash(grid) != manifest["spec_sha256"]:
        raise ValueError(f"Grid {manifest['grid_id']} changed in grids.json since init; "
                         "define a new grid ID (completed runs are kept)")
    require_unchanged_source({"implementation": manifest["implementation"]})
    current = {}
    for dataset, entry in manifest["entries"].items():
        runs, _record, path = resolve_runs(manifest["grid_id"], grid, dataset, results_dir)
        if sha256_file(path) != entry["base_record_sha256"]:
            raise ValueError(f"Base record changed: {path}")
        if set(runs) != set(entry["runs"]):
            raise ValueError(f"{dataset}: run set changed since init")
        for run_id, frozen in entry["runs"].items():
            if runs[run_id]["run_config_hash"] != frozen["run_config_hash"]:
                raise ValueError(f"{dataset}/{run_id}: effective configuration changed since init")
        current[dataset] = runs
    return current


# --------------------------------------------------------------------------
# Execution and status
# --------------------------------------------------------------------------

def select_runs(runs, *, mining=None, limit=None):
    wanted = [run_id for run_id in sorted(runs, key=run_order_key)
              if not mining or runs[run_id]["tags"][0] in mining]
    return wanted[:limit] if limit else wanted


def print_plan(grid_id, grid, *, results_dir, check_caches):
    points = len(combinations(grid))
    print(f"Grid {grid_id}: base run {grid['base_run']}, repeats {grid['repeats']}, "
          f"{points} configurations per dataset")
    for axis in [{"name": "mining", "options": grid["mining"]}] + grid["axes"]:
        print(f"  {axis['name']:18s} " + "  ".join(
            f"{option['tag']}={json.dumps(option['overrides'])}" for option in axis["options"]))
    for dataset in grid["datasets"]:
        runs, _record, path = resolve_runs(grid_id, grid, dataset, results_dir)
        estimate = ablations.estimate_seconds(
            path, {run_id: {**run, "tier": "classification"} for run_id, run in runs.items()})
        hours = "unknown" if estimate is None else f"{estimate / 3600:.1f} h"
        print(f"\n{dataset}: {len(runs)} runs, classification estimate {hours} "
              "(from the base run's per-repeat median)")
        if check_caches:
            for tag, cached in mining_cache_status(grid, dataset, results_dir).items():
                note = ("" if len(cached) == len(grid["repeats"]) else
                        "  <- the first run mines the missing repeats, later runs reuse them")
                print(f"  mining {tag}: cached repeats {cached}{note}")


def print_status(manifest, runs_by_dataset):
    for dataset, runs in runs_by_dataset.items():
        counts = {}
        for run in runs.values():
            state, _done = ablations.run_state(run)
            counts[state] = counts.get(state, 0) + 1
        print(f"{dataset:15s} " + "  ".join(f"{state} {count}"
                                             for state, count in sorted(counts.items())))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def load_run_metrics(results_dir, dataset, run_id, repeats):
    """Per-repeat metrics of a complete run, or None while it is incomplete."""
    root = ablations.result_base(results_dir, dataset) / "inductive" / run_id
    experiment, table = root / "experiment.json", root / "repeats.csv"
    if not experiment.is_file() or not table.is_file():
        return None
    if read_json(experiment).get("status") != "complete":
        return None
    with table.open(encoding="utf-8") as handle:
        rows = {int(row["repeat"]): row for row in csv.DictReader(handle)}
    if sorted(rows) != list(repeats):
        return None
    values = {metric: [float(rows[r][column]) for r in repeats]
              for metric, column in CSV_COLUMNS.items()}
    values["mined"] = sum(rows[r].get("cache_reused") != "True" for r in repeats)
    return values


def _mean(values):
    return statistics.fmean(values)


def _rank_key(run_id, val_f1, val_acc):
    return (-val_f1, -val_acc, run_id)


def build_report(manifest, runs_by_dataset, *, results_dir):
    """Return (rows, markdown) for the grid's completed runs."""

    repeats, datasets = manifest["repeats"], manifest["datasets"]
    grid = manifest["spec"]
    metrics = {dataset: {} for dataset in datasets}
    rows = []
    for dataset in datasets:
        for run_id, run in runs_by_dataset[dataset].items():
            values = load_run_metrics(results_dir, dataset, run_id, repeats)
            if values is None:
                continue
            metrics[dataset][run_id] = values
            rows.append({"run_id": run_id, "dataset": dataset,
                         **{f"tag_{i}": tag for i, tag in enumerate(run["tags"])},
                         **{metric: round(_mean(values[metric]), 4) for metric in METRICS}})

    complete = sorted(set.intersection(*(set(metrics[d]) for d in datasets)))
    mined = sum(values["mined"] for dataset in datasets for values in metrics[dataset].values())
    total = len(next(iter(runs_by_dataset.values())))
    lines = [f"# Grid {manifest['grid_id']}", "",
             f"Base run {manifest['base_run']}; repeats {repeats}; fully inductive.",
             f"Complete on all datasets: {len(complete)} of {total} configurations.",
             f"Repeats that mined instead of reusing a cache: {mined}.",
             "", "Selection uses validation Macro-F1, then validation accuracy. Test "
             "values are shown only for selected configurations. Means are over repeats.", ""]

    def mean_over(run_id, metric):
        return _mean([_mean(metrics[d][run_id][metric]) for d in datasets])

    def selected_table(run_id):
        out = ["| Dataset | Val F1 | Val acc. | Test F1 | Test acc. |",
               "|---|---|---|---|---|"]
        for dataset in datasets:
            values = metrics[dataset][run_id]
            out.append(f"| {dataset} | {_mean(values['val_macro_f1']):.2f} | "
                       f"{_mean(values['val_accuracy']):.2f} | "
                       f"{_mean(values['test_macro_f1']):.2f} | "
                       f"{_mean(values['test_accuracy']):.2f} |")
        return out + [""]

    if complete:
        ranked = sorted(complete, key=lambda r: _rank_key(
            r, mean_over(r, "val_macro_f1"), mean_over(r, "val_accuracy")))
        lines += ["## Shared configuration (one setting for all datasets)", "",
                  f"Selected: `{ranked[0]}` (mean validation Macro-F1 "
                  f"{mean_over(ranked[0], 'val_macro_f1'):.2f}).", ""]
        lines += selected_table(ranked[0])
        for mining in grid["mining"]:
            within = [r for r in ranked if r.startswith(
                run_id_for(manifest["grid_id"], (mining["tag"],)) + "_")]
            if within:
                lines += [f"### Best with mining {mining['tag']}: `{within[0]}` "
                          f"(mean validation Macro-F1 {mean_over(within[0], 'val_macro_f1'):.2f})",
                          ""]
                lines += selected_table(within[0])
        lines += ["### Top 10 by mean validation Macro-F1 (validation only)", "",
                  "| Rank | Run | Val F1 | Val acc. |", "|---|---|---|---|"]
        lines += [f"| {rank} | {run_id} | {mean_over(run_id, 'val_macro_f1'):.2f} | "
                  f"{mean_over(run_id, 'val_accuracy'):.2f} |"
                  for rank, run_id in enumerate(ranked[:10], 1)]
        lines.append("")

    return rows, "\n".join(lines)


def write_report(manifest, runs_by_dataset, *, results_dir):
    rows, markdown = build_report(manifest, runs_by_dataset, results_dir=results_dir)
    out = manifest_path(results_dir, manifest["grid_id"]).parent / "report"
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row},
                    key=lambda k: (k not in ("run_id", "dataset"), not k.startswith("tag_"), k))
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "README.md").write_text(markdown + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("plan", "init", "run", "status", "report"))
    parser.add_argument("--grid", required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS,
                        help="Required for run; one process per dataset")
    parser.add_argument("--mining", nargs="+", help="Limit run to these mining tags")
    parser.add_argument("--limit", type=int, help="Run at most this many configurations")
    parser.add_argument("--repeat", type=int, choices=range(5),
                        help="Run one repeat only; rerun later without it to finish")
    parser.add_argument("--skip-cache-check", action="store_true",
                        help="plan: do not load splits to check mining caches")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from pipeline.artifacts import atomic_json_write, file_lock

    grid = load_grid(args.grid, args.spec)
    results_dir = args.results_dir.resolve()
    path = manifest_path(results_dir, args.grid)

    if args.stage == "plan":
        print_plan(args.grid, grid, results_dir=results_dir,
                   check_caches=not args.skip_cache_check)
        return 0

    if args.stage == "init":
        if path.exists():
            raise FileExistsError(f"Grid already initialized: {path}")
        manifest = build_manifest(args.grid, grid, results_dir=results_dir)
        for dataset, entry in manifest["entries"].items():
            for run_id in entry["runs"]:
                root = ablations.result_base(results_dir, dataset) / "inductive" / run_id
                if root.exists():
                    raise FileExistsError(f"Output already exists for a new grid: {root}")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, manifest)
        print(f"Initialized {path}")
        return 0

    if not path.exists():
        raise FileNotFoundError(f"Grid {args.grid} is not initialized; run the init stage first")
    manifest = read_json(path)
    if manifest.get("grid_id") != args.grid:
        raise ValueError("Grid identity mismatch")

    if args.stage == "report":
        # Reads compact records only, so it also works on a copy of the results.
        from pipeline.artifacts import canonical_json_hash
        if canonical_json_hash(grid) != manifest["spec_sha256"]:
            raise ValueError(f"Grid {args.grid} changed in grids.json since init")
        runs_by_dataset = {dataset: entry["runs"]
                           for dataset, entry in manifest["entries"].items()}
        out = write_report(manifest, runs_by_dataset, results_dir=results_dir)
        print(f"Report written to {out}")
        return 0

    runs_by_dataset = validate_manifest(manifest, grid, results_dir=results_dir)
    if args.stage == "status":
        print_status(manifest, runs_by_dataset)
        return 0

    if args.dataset is None:
        raise ValueError("run needs --dataset; start one process per dataset")
    if args.dataset not in runs_by_dataset:
        raise ValueError(f"{args.dataset} is not part of grid {args.grid}")
    from pipeline.inductive import run_fully_inductive_pipeline
    from utils.logger import setup_logging
    log_path = setup_logging("grid_search", dataset=args.dataset, run_id=args.grid)
    logger.info("Grid %s on %s; detailed log: %s", args.grid, args.dataset, log_path)
    runs = runs_by_dataset[args.dataset]
    selected = select_runs(runs, mining=args.mining, limit=args.limit)
    with file_lock(str(path.parent / f"{args.dataset}.lock")):
        for number, run_id in enumerate(selected, 1):
            run = runs[run_id]
            state, done = ablations.run_state(run)
            if state == "complete" or (args.repeat is not None and args.repeat in done):
                logger.info("[%d/%d] %s: already complete; skipping", number, len(selected), run_id)
                continue
            if state == "invalid":
                raise ValueError(f"Invalid completed run, inspect before continuing: "
                                 f"{ablations.run_directory(run['config'])}")
            logger.info("[%d/%d] %s: starting", number, len(selected), run_id)
            run_fully_inductive_pipeline(copy.deepcopy(run["config"]), mode="full",
                                         repeat=args.repeat)
    logger.info("Grid %s on %s: requested runs finished", args.grid, args.dataset)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
