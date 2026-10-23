"""Check every dataset and repeat of a campaign, then summarize its scores."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pipeline.artifacts import (
    atomic_json_write, atomic_text_write, load_gzip_json, sha256_file, valid_completion_marker,
)
from scripts.run_campaign import (
    completion_stage, read_json, run_directory, validated_splits,
)


def validate_run(root, entry, manifest, labels, splits):
    """Per-repeat records of one dataset's run, after checking it against the campaign."""
    config = entry["config"]
    if not valid_completion_marker(root, stage=completion_stage(config),
            identity={"run_id": config["run_id"], "run_config_hash": entry["run_config_hash"]}):
        raise ValueError(f"Incomplete or invalid run: {root}")
    experiment = read_json(root / "experiment.json")
    if (experiment["provenance"]["implementation"]["run"] != manifest["implementation"]["run"] or
            experiment["source_hashes"] != entry["source_hashes"] or
            experiment["split_manifest_hashes"] != entry["split_hashes"] or
            any(experiment["config"].get(k) != v for k, v in config.items())):
        raise ValueError("Run provenance or settings do not match the campaign")
    records = {}
    for repeat in manifest["repeats"]:
        directory = root / f"repeat_{repeat}"
        metrics = read_json(directory / "metrics.json")
        predictions = load_gzip_json(directory / "predictions.json.gz")
        for name, ids in (("validation", splits[repeat].validation), ("test", splits[repeat].test)):
            if predictions[name]["truth"] != labels[ids].tolist():
                raise ValueError("Prediction targets do not match installed labels")
        records[repeat] = {**metrics, "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]}
    return records


def summarize_records(rows, repeats):
    """Means and population SDs over the declared repeats."""
    if sorted(rows) != repeats:
        raise ValueError("Incomplete repeat set")
    result = {}
    for key in ("test_accuracy", "test_macro_f1", "val_accuracy", "val_macro_f1"):
        values = [float(rows[r][key]) for r in repeats]
        if not np.isfinite(values).all() or any(not 0 <= v <= 100 for v in values):
            raise ValueError("Invalid metric")
        result[key] = {"repeats": values, "mean": float(np.mean(values)),
                       "population_sd": float(np.std(values, ddof=0))}
    for key in ("patterns", "trainable_parameters", "classification_seconds", "epochs_trained",
                "best_epoch", "mining_seconds", "validation_matching_seconds",
                "test_matching_seconds", "end_to_end_seconds"):
        result[key] = [rows[r].get(key) for r in repeats]
    return result


def analyze_campaign(manifest, *, datasets=None):
    datasets = manifest["datasets"] if datasets is None else datasets
    if not datasets or not set(datasets).issubset(manifest["datasets"]):
        raise ValueError("Choose datasets declared in the campaign")
    report = {"campaign_id": manifest["campaign_id"], "repeats": manifest["repeats"],
              "protocol": manifest["protocol"], "datasets": {},
              "implementation": manifest["implementation"]}
    for dataset in datasets:
        entry = manifest["entries"][dataset]
        config = entry["config"]
        _, labels, splits = validated_splits(config, manifest["repeats"])
        source_hashes = {"hypergraphs": sha256_file(Path(config["paths"]["hypergraph_data_dir"]) / (config["domains"][0] + ".txt")),
                         "labels": sha256_file(config["paths"]["labels_file"])}
        if source_hashes != entry["source_hashes"] or {str(r): s.manifest_hash for r, s in splits.items()} != entry["split_hashes"]:
            raise ValueError("Installed data or split hashes do not match campaign")
        records = validate_run(run_directory(config), entry, manifest, labels, splits)
        report["datasets"][dataset] = summarize_records(records, manifest["repeats"])
    return report


def write_analysis(directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json_write(directory / "analysis.json", report)
    rows = []
    for dataset, values in report["datasets"].items():
        rows.append({"dataset": dataset,
                     **{key + "_" + statistic: value[statistic] for key, value in values.items()
                        if isinstance(value, dict) for statistic in ("mean", "population_sd")}})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(directory / "summary.csv", stream.getvalue())
    lines = ["# Campaign analysis", "", f"Campaign: {report['campaign_id']} ({report['protocol']})",
             "", "Every declared dataset and repeat passed the artifact and data checks.",
             "Means ± population standard deviations over the repeats.",
             "", "| Dataset | Test accuracy | Test Macro-F1 | Validation Macro-F1 |",
             "|---|---:|---:|---:|"]
    for dataset, values in report["datasets"].items():
        cells = [f"{values[k]['mean']:.2f} ± {values[k]['population_sd']:.2f}"
                 for k in ("test_accuracy", "test_macro_f1", "val_macro_f1")]
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.append("")
    atomic_text_write(directory / "analysis.md", "\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output-dir", type=Path, help="Write analysis.json, summary.csv and analysis.md")
    args = parser.parse_args(argv)
    report = analyze_campaign(read_json(args.manifest), datasets=args.datasets)
    if args.output_dir:
        write_analysis(args.output_dir, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
