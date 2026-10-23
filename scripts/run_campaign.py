"""Frozen, resumable campaigns of the method over datasets and repeats.

Every campaign uses the settings in pipeline/profiles.py; --protocol selects
fully inductive or transductive vocabulary access. --stage init freezes the
campaign manifest (settings, data and partition hashes, source identity);
--stage run executes datasets and repeats under that identity, and a resumed
run refuses any change. --dry-run checks everything without writing. The run
ID of each dataset's run is the campaign ID.
"""
import argparse
import copy
import json
import logging
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logger import setup_logging

from mining.io import load_hypergraphs_from_file, load_labels
from pipeline.artifacts import (
    atomic_json_write, canonical_json_hash, sha256_file, valid_completion_marker,
)
from pipeline.profiles import FINAL_PROFILE_NAME, FINAL_SETTINGS, final_dataset_config
from pipeline.provenance import capture_provenance, require_unchanged_source
from pipeline.splits import load_and_validate_split_manifest
DATASETS = ("imdb_dir_form", "imdb_dir_genre", "steam_player",
            "twitter_friend", "rhg_10", "rhg_3")
REPO_ROOT = Path(__file__).resolve().parents[1]
# Both protocols use the final settings; only vocabulary access differs.
PROTOCOLS = {
    "inductive": {"result_subdir": "inductive",
                  "pipeline_mode": "fully_inductive", "vocabulary_access": "train_only",
                  "cache_dir_key": "inductive_mining_cache_dir"},
    "transductive": {"result_subdir": "transductive",
                     "pipeline_mode": "transductive_vocabulary", "vocabulary_access": "all_unlabeled",
                     "cache_dir_key": "transductive_mining_cache_dir"},
}
logger = logging.getLogger(__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def campaign_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError("Use a fresh filename-safe campaign ID")
    return value


def run_directory(config):
    from pipeline.inductive import _protocol_contract
    return (Path(config["paths"]["coverage_file"]).parent /
            _protocol_contract(config)["result_subdir"] / config["run_id"])


def completion_stage(config):
    from pipeline.inductive import _protocol_contract
    return _protocol_contract(config)["run_stage"]


def validated_splits(config, repeats):
    graphs = load_hypergraphs_from_file(Path(config["paths"]["hypergraph_data_dir"]) / (config["domains"][0] + ".txt"))
    labels = load_labels(config["paths"]["labels_file"])
    from pipeline.preprocessing import verify_preprocessing
    verify_preprocessing(graphs, labels, config,
        Path(config["paths"]["hypergraph_data_dir"]) / (config["domains"][0] + ".txt"),
        config["paths"]["labels_file"])
    splits = {}
    for repeat in repeats:
        splits[repeat] = load_and_validate_split_manifest(
            Path(config["split_manifest_dir"]) / f"repeat_{repeat}.json",
            dataset=config["_dataset"], label_mode=config["_label_mode"],
            hypergraph_file=Path(config["paths"]["hypergraph_data_dir"]) / (config["domains"][0] + ".txt"),
            labels_file=config["paths"]["labels_file"], num_graphs=len(graphs),
            expected_repeat=repeat, expected_domain=config["domains"][0])
        if set(map(int, labels[splits[repeat].train])) != set(range(config["num_classes"])):
            raise ValueError("Every training split must include all configured classes")
    return graphs, labels, splits


def build_manifest(identifier, datasets, repeats, cache_policy, *,
                   repo_root=REPO_ROOT, results_dir=None, require_clean_source=False,
                   protocol="inductive"):
    """Audit the settings and installed inputs and resolve each dataset's frozen config."""
    from pipeline.inductive import _effective_config_sections, _run_config_hash, _validate_config
    campaign_id(identifier)
    if not datasets or len(set(datasets)) != len(datasets) or any(d not in DATASETS for d in datasets):
        raise ValueError("Choose unique supported datasets")
    if repeats != list(range(len(repeats))) or not 1 <= len(repeats) <= 5:
        raise ValueError("Declare a prefix of repeats: 0 for a pilot, or 0 1 2 3 4 for the study")
    if cache_policy not in {"build-missing", "require-existing"}:
        raise ValueError("Explicit cache policy is required")
    if protocol not in PROTOCOLS:
        raise ValueError("Choose a supported protocol")
    contract = PROTOCOLS[protocol]
    repo_root = Path(repo_root).resolve()
    results_dir = Path(results_dir or repo_root / "results").resolve()
    provenance = capture_provenance()
    if require_clean_source and provenance["git_source_dirty"] is not False:
        raise ValueError("Committed, clean source is required for this campaign")
    entries = {}
    shared = None
    settings_hash = canonical_json_hash(FINAL_SETTINGS)
    for dataset in datasets:
        cfg = final_dataset_config(dataset, identifier)
        cfg.pop("experiment_profile", None)
        cfg.update(pipeline_mode=contract["pipeline_mode"], vocabulary_access=contract["vocabulary_access"],
                   random_state=0, num_repeats=len(repeats), require_clean_source=require_clean_source,
                   classifier_type="nn", representation_type="supervised_sum")
        cfg["paths"] = {key: str((repo_root / value).resolve()) for key, value in cfg["paths"].items()}
        result_base = results_dir / dataset / "composite"
        cfg["paths"]["coverage_file"] = str(result_base / "cover_freq.txt")
        cfg["inductive_mining_cache_dir"] = str(result_base / "inductive_mining_cache")
        cfg["transductive_mining_cache_dir"] = str(result_base / "transductive_mining_cache")
        cfg["split_manifest_dir"] = str((repo_root / cfg["split_manifest_dir"]).resolve())
        _validate_config(cfg, None)
        settings = copy.deepcopy(_effective_config_sections(cfg))
        settings["classification_config"].pop("num_classes", None)
        if shared is not None and shared != settings:
            raise ValueError("Datasets do not share the same mining/training settings")
        shared = settings
        graphs, labels, splits = validated_splits(cfg, repeats)
        entries[dataset] = {
            "final_profile": FINAL_PROFILE_NAME, "final_settings_sha256": settings_hash,
            "graph_count": len(graphs),
            "source_hashes": {
                "hypergraphs": sha256_file(Path(cfg["paths"]["hypergraph_data_dir"]) / (cfg["domains"][0] + ".txt")),
                "labels": sha256_file(cfg["paths"]["labels_file"]),
            },
            "split_hashes": {str(r): split.manifest_hash for r, split in splits.items()},
            "config": cfg,
            "run_config_hash": _run_config_hash(cfg),
        }
    return {"schema_version": 1, "campaign_id": identifier, "datasets": list(datasets),
            "repeats": repeats, "cache_policy": cache_policy,
            "protocol": contract["pipeline_mode"],
            "implementation": provenance["implementation"], "initial_provenance": provenance,
            "shared_settings": shared, "entries": entries,
            "memory_scope": "process lifetime RSS; no isolated-repeat memory comparison"}


def validate_manifest(manifest):
    """Bind a resume to source, saved settings, data and manifests; never rewrite them."""
    from pipeline.inductive import _run_config_hash
    require_unchanged_source({"implementation": manifest["implementation"]})
    for dataset, entry in manifest["entries"].items():
        if canonical_json_hash(FINAL_SETTINGS) != entry["final_settings_sha256"]:
            raise ValueError(f"Final settings in pipeline/profiles.py changed: {dataset}")
        _, _, splits = validated_splits(entry["config"], manifest["repeats"])
        if {str(r): s.manifest_hash for r, s in splits.items()} != entry["split_hashes"]:
            raise ValueError("Campaign splits changed")
        if _run_config_hash(entry["config"]) != entry["run_config_hash"]:
            raise ValueError("Campaign effective configuration changed")


def execute_manifest(manifest, *, datasets=None, repeat=None):
    from pipeline.inductive import run_fully_inductive_pipeline, run_transductive_pipeline
    runners = {"fully_inductive": run_fully_inductive_pipeline,
               "transductive_vocabulary": run_transductive_pipeline}
    run_pipeline = runners[manifest["protocol"]]
    validate_manifest(manifest)
    datasets = manifest["datasets"] if datasets is None else datasets
    repeats = manifest["repeats"] if repeat is None else [repeat]
    if (not set(datasets).issubset(manifest["datasets"]) or
            not set(repeats).issubset(manifest["repeats"])):
        raise ValueError("Execution selection is outside the frozen campaign")
    for dataset in datasets:
        entry = manifest["entries"][dataset]
        config = entry["config"]
        root = run_directory(config)
        logger.info("%s: %s", dataset, root)
        if (root / "completion.json").exists():
            if not valid_completion_marker(root, stage=completion_stage(config),
                    identity={"run_id": config["run_id"], "run_config_hash": entry["run_config_hash"]}):
                raise ValueError(f"Invalid completed run: {root}")
            logger.info("%s: validated completed run; skipping", dataset)
            continue
        mode = "full" if manifest["cache_policy"] == "build-missing" else "classification"
        run_pipeline(config, mode=mode, repeat=repeat)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", type=campaign_id, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--repeats", nargs="+", type=int, choices=range(5))
    parser.add_argument("--repeat", type=int, choices=range(5), help="Execute only this declared repeat; resume others later")
    parser.add_argument("--cache-policy", choices=("build-missing", "require-existing"))
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS),
                        help="Vocabulary access (new campaigns default to inductive)")
    parser.add_argument("--stage", choices=("init", "run"), default="run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Campaign manifests live under this results folder.
    campaign_root = args.results_dir.resolve() / "campaigns" / args.campaign_id
    manifest_path = campaign_root / "campaign.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["campaign_id"] != args.campaign_id:
            raise ValueError("Campaign identity mismatch")
        if args.cache_policy is not None and args.cache_policy != manifest["cache_policy"]:
            raise ValueError("Cannot change frozen cache policy")
        if (args.protocol is not None and
                PROTOCOLS[args.protocol]["pipeline_mode"] != manifest["protocol"]):
            raise ValueError("Cannot change the frozen protocol; use a new campaign ID")
        if args.repeats is not None and args.repeats != manifest["repeats"]:
            raise ValueError("Cannot change declared repeats; use a new campaign ID")
        if args.require_clean_source and manifest["initial_provenance"]["git_source_dirty"] is not False:
            raise ValueError("Existing campaign was frozen from dirty source")
        validate_manifest(manifest)
    else:
        if campaign_root.exists():
            raise FileExistsError(f"Refusing unrelated campaign directory: {campaign_root}")
        if args.cache_policy is None:
            raise ValueError("--cache-policy is required for a new campaign")
        manifest = build_manifest(args.campaign_id, args.datasets or list(DATASETS),
            args.repeats or list(range(5)), args.cache_policy,
            results_dir=args.results_dir, require_clean_source=args.require_clean_source,
            protocol=args.protocol or "inductive")
        for entry in manifest["entries"].values():
            if run_directory(entry["config"]).exists():
                raise FileExistsError(
                    f"Refusing pre-existing output for a new campaign: {run_directory(entry['config'])}")
    if ((args.datasets and not set(args.datasets).issubset(manifest["datasets"])) or
            (args.repeat is not None and args.repeat not in manifest["repeats"])):
        raise ValueError("Selection outside frozen campaign")
    print(json.dumps({"campaign_id": manifest["campaign_id"], "datasets": args.datasets or manifest["datasets"],
                      "repeats": [args.repeat] if args.repeat is not None else manifest["repeats"],
                      "cache_policy": manifest["cache_policy"],
                      "protocol": manifest["protocol"],
                      "manifest": str(manifest_path),
                      "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return 0
    datasets = args.datasets or manifest["datasets"]
    log_path = setup_logging("campaign",
        dataset=datasets[0] if len(datasets) == 1 else "multi_dataset",
        run_id=args.campaign_id)
    logger.info("Campaign %s: stage=%s; detailed log: %s",
                args.campaign_id, args.stage, log_path)
    from pipeline.inductive import _shared_file_lock
    campaign_root.mkdir(parents=True, exist_ok=True)
    # Lock only the manifest write, so datasets of one campaign can run in
    # parallel; each run owns its directory and caches under their own locks.
    with _shared_file_lock(campaign_root):
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError("Concurrent campaign initialization disagrees")
        else:
            atomic_json_write(manifest_path, manifest)
    if args.stage == "run":
        execute_manifest(manifest, datasets=args.datasets, repeat=args.repeat)
    logger.info("Campaign %s: %s finished", args.campaign_id, args.stage)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
