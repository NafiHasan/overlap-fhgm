"""Collect a frozen ablation suite into tables and figures.

    python scripts/ablation/report_ablations.py --suite SUITE

Reads compact run records (``experiment.json`` and ``repeats.csv``, plus
per-repeat files when a run is still partial), so it also works when only
the compact records are available. Every run must match the frozen suite
manifest. Output goes to ``results/ablation/<suite>/report/``:

    README.md                 coverage, reproduction check, how to read
    records.csv, summary.csv  per-repeat and aggregated values
    <STUDY>.md                Markdown tables for each study
    <STUDY>_*.pdf / .png      figures

Means and population standard deviations are over repeats. Paired differences
are variant minus control within each repeat. Selection, where shown, uses
validation Macro-F1 only; test values are descriptive.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LABEL_MODE = "composite"
DISPLAY = {
    "imdb_dir_form": "IMDB-Dir-Form", "imdb_dir_genre": "IMDB-Dir-Genre",
    "steam_player": "Steam-Player", "twitter_friend": "Twitter-Friend",
    "rhg_10": "RHG-10", "rhg_3": "RHG-3",
}
METRICS = ("test_macro_f1", "test_accuracy", "val_macro_f1", "val_accuracy")
METRIC_NAMES = {"test_macro_f1": "Test Macro-F1", "test_accuracy": "Test accuracy",
                "val_macro_f1": "Validation Macro-F1", "val_accuracy": "Validation accuracy"}
COSTS = ("candidates", "selected", "mining_seconds", "classification_seconds",
         "trainable_parameters")
TIE_TOLERANCE = 0.005  # percentage points

# Figure palette: ink and series colours.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
SERIES = {"validation": "#2a78d6", "test": "#eb6834"}
MARKERS = {"validation": "o", "test": "s"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _pipeline_row(row):
    return {
        "repeat": int(row["repeat"]),
        "val_accuracy": _float(row.get("val_accuracy")),
        "val_macro_f1": _float(row.get("val_macro_f1")),
        "test_accuracy": _float(row.get("accuracy")),
        "test_macro_f1": _float(row.get("macro_f1")),
        "candidates": _float(row.get("patterns")),
        "selected": _float(row.get("selected_patterns")),
        "mining_seconds": _float(row.get("mining_seconds")),
        "classification_seconds": _float(row.get("classification_seconds")),
        "trainable_parameters": _float(row.get("trainable_parameters")),
    }


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run(root, run):
    """Return (status, rows, note) for one frozen run, verifying its identity."""
    root = Path(root)
    experiment_path = root / "experiment.json"
    if experiment_path.exists():
        experiment = read_json(experiment_path)
        if experiment.get("run_config_hash") != run["run_config_hash"]:
            return "mismatch", [], "run_config_hash differs from the suite"
        if experiment.get("status") == "complete" and (root / "repeats.csv").exists():
            note = "" if (root / "completion.json").exists() else "compact record only"
            return "complete", [_pipeline_row(r) for r in _read_csv(root / "repeats.csv")], note
    rows = []
    for metrics_path in sorted(root.glob("repeat_*/metrics.json")):
        if not (metrics_path.parent / "completion.json").exists():
            continue
        metrics = read_json(metrics_path)
        if metrics.get("run_config_hash") != run["run_config_hash"]:
            return "mismatch", [], "repeat run_config_hash differs from the suite"
        rows.append(_pipeline_row({**metrics, "selected_patterns": metrics.get("selected_patterns")}))
    return ("partial" if rows else "missing"), rows, ""


def load_suite_records(manifest, results_dir):
    """Per-repeat records for every dataset, study and variant of the suite."""
    records, coverage = [], []
    results_dir = Path(results_dir)
    for dataset, entry in manifest["entries"].items():
        for run_id, run in entry["runs"].items():
            root = results_dir / dataset / LABEL_MODE / "inductive" / run_id
            status, rows, note = load_run(root, run)
            coverage.append({"dataset": dataset, "run_id": run_id, "tier": run["tier"],
                             "status": status, "repeats": len(rows), "note": note})
            for study_id, variant_id in run["uses"]:
                for row in rows:
                    records.append({"dataset": dataset, "study": study_id,
                                    "variant": variant_id, "run_id": run_id,
                                    "tier": run["tier"], **row})
    return records, coverage


def reproduction_check(manifest, results_dir, records):
    """Compare the control with the base run, repeat by repeat."""
    checks = []
    control_id = f"{manifest['suite_id']}_CONTROL"
    for dataset in manifest["entries"]:
        base_csv = (Path(results_dir) / dataset / LABEL_MODE / "inductive" /
                    manifest["base_run"] / "repeats.csv")
        control = {r["repeat"]: r for r in records
                   if r["dataset"] == dataset and r["run_id"] == control_id}
        if not base_csv.exists() or not control:
            checks.append({"dataset": dataset, "compared": 0, "max_abs_diff": None})
            continue
        base = {int(row["repeat"]): _pipeline_row(row) for row in _read_csv(base_csv)}
        diffs = [abs(control[k][metric] - base[k][metric])
                 for k in control if k in base
                 for metric in ("test_macro_f1", "test_accuracy", "val_macro_f1")]
        checks.append({"dataset": dataset, "compared": len({k for k in control if k in base}),
                       "max_abs_diff": max(diffs) if diffs else None})
    return checks


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _mean_sd(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarize(manifest, records):
    """Aggregate by dataset/study/variant, with paired differences to control."""
    spec = manifest["spec"]
    summary = []
    for study in spec["studies"]:
        control_id = next(v["id"] for v in study["variants"] if v.get("control"))
        for dataset in manifest["entries"]:
            control_rows = {r["repeat"]: r for r in records if r["dataset"] == dataset
                            and r["study"] == study["id"] and r["variant"] == control_id}
            for variant in study["variants"]:
                rows = [r for r in records if r["dataset"] == dataset and
                        r["study"] == study["id"] and r["variant"] == variant["id"]]
                item = {"dataset": dataset, "study": study["id"], "variant": variant["id"],
                        "label": _variant_label(study, variant),
                        "x": variant.get("x"), "control": bool(variant.get("control")),
                        "n": len(rows)}
                for metric in METRICS:
                    item[f"{metric}_mean"], item[f"{metric}_sd"] = _mean_sd(r[metric] for r in rows)
                for cost in COSTS:
                    item[f"{cost}_median"] = _median(r[cost] for r in rows)
                for metric in ("test_macro_f1", "test_accuracy", "val_macro_f1"):
                    deltas = [r[metric] - control_rows[r["repeat"]][metric] for r in rows
                              if r["repeat"] in control_rows and r[metric] is not None
                              and control_rows[r["repeat"]][metric] is not None]
                    mean, sd = _mean_sd(deltas)
                    item[f"delta_{metric}_mean"], item[f"delta_{metric}_sd"] = mean, sd
                    item[f"delta_{metric}_n"] = len(deltas)
                    item[f"delta_{metric}_wins"] = sum(d > TIE_TOLERANCE for d in deltas)
                    item[f"delta_{metric}_losses"] = sum(d < -TIE_TOLERANCE for d in deltas)
                summary.append(item)
    return summary


def validation_choice(summary, study_id, dataset):
    """Variant with the highest mean validation Macro-F1 (ties: earlier in spec)."""
    best = None
    for item in summary:
        if item["study"] != study_id or item["dataset"] != dataset:
            continue
        if item["val_macro_f1_mean"] is None or item["n"] == 0:
            continue
        if best is None or item["val_macro_f1_mean"] > best["val_macro_f1_mean"] + 1e-12:
            best = item
    return best["variant"] if best else None


# --------------------------------------------------------------------------
# Text output
# --------------------------------------------------------------------------

def _fmt(mean, sd=None, digits=2):
    if mean is None:
        return "--"
    return f"{mean:.{digits}f}" if sd is None else f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _fmt_delta(item, metric):
    mean = item.get(f"delta_{metric}_mean")
    if item["control"]:
        return "ref."
    if mean is None:
        return "--"
    sd = item[f"delta_{metric}_sd"]
    wins, losses, n = (item[f"delta_{metric}_{key}"] for key in ("wins", "losses", "n"))
    return f"{mean:+.2f} ± {sd:.2f} ({wins}↑/{losses}↓ of {n})"


def _rows_for(summary, study, dataset):
    return {item["variant"]: item for item in summary
            if item["study"] == study["id"] and item["dataset"] == dataset}


def _variant_label(study, variant):
    return variant.get("label", str(variant.get("x", variant["id"])))


def _row_label(study, variant):
    if study["type"] == "sweep":
        return str(variant["x"]) + (" (base)" if variant.get("control") else "")
    return _variant_label(study, variant)


def _markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def study_markdown(manifest, study, summary):
    datasets = list(manifest["entries"])
    header = ["Variant"] + [DISPLAY.get(d, d) for d in datasets]
    parts = [f"# {study['id']}: {study['title']}", "",
             f"Type: {study['type']}"
             + (f"; parameter `{study['parameter']}`." if study["type"] == "sweep" else "."), ""]
    if study.get("note"):
        parts += [study["note"], ""]
    choices = {d: validation_choice(summary, study["id"], d) for d in datasets}
    for metric in METRICS:
        rows = []
        for variant in study["variants"]:
            cells = []
            for dataset in datasets:
                item = _rows_for(summary, study, dataset).get(variant["id"])
                cell = _fmt(item[f"{metric}_mean"], item[f"{metric}_sd"]) if item else "--"
                if study["type"] == "sweep" and choices[dataset] == variant["id"]:
                    cell += " ◆"
                cells.append(cell)
            rows.append([_row_label(study, variant)] + cells)
        parts += [f"## {METRIC_NAMES[metric]} (mean ± population SD over repeats)", "",
                  _markdown_table(header, rows), ""]
    if study["type"] == "sweep":
        parts += ["◆ marks the value with the highest mean validation Macro-F1 for that "
                  "dataset. Test columns are descriptive, not a selection criterion.", ""]
    for metric in ("test_macro_f1", "val_macro_f1"):
        rows = [[_row_label(study, variant)] +
                [_fmt_delta(_rows_for(summary, study, d).get(variant["id"],
                                                             {"control": False}), metric)
                 for d in datasets]
                for variant in study["variants"]]
        parts += [f"## Paired difference from the control, {METRIC_NAMES[metric]} "
                  "(points; repeats higher/lower)", "", _markdown_table(header, rows), ""]
    cost_rows = []
    for variant in study["variants"]:
        cells = []
        for dataset in datasets:
            item = _rows_for(summary, study, dataset).get(variant["id"])
            if not item or item["n"] == 0:
                cells.append("--")
                continue
            cand, sel = item["candidates_median"], item["selected_median"]
            mine = item["mining_seconds_median"]
            cells.append(
                f"{'--' if cand is None else f'{cand:,.0f}'} / "
                f"{'--' if sel is None else f'{sel:,.0f}'} / "
                f"{'--' if mine is None else f'{mine:,.0f} s'}")
        cost_rows.append([_row_label(study, variant)] + cells)
    parts += ["## Cost: median candidates / selected patterns / mining seconds per repeat", "",
              "Mining time is the time recorded when the cache was built; reused caches "
              "repeat the original value.", "", _markdown_table(header, cost_rows), ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
        "axes.titleweight": "semibold", "axes.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "text.color": INK, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def _style_axis(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def _grid(plt, count, height=2.0):
    columns = min(3, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.35 * columns, height * rows + 0.45),
                             squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, axes


def _save(fig, path_stem):
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(str(path_stem) + suffix, bbox_inches="tight", **kwargs)


def sweep_figure(manifest, study, summary, path_stem):
    """Validation and test Macro-F1 against the swept value, one panel per dataset."""
    plt = _pyplot()
    datasets = list(manifest["entries"])
    fig, axes = _grid(plt, len(datasets))
    positions = range(len(study["variants"]))
    labels = [str(v["x"]) for v in study["variants"]]
    control = next(i for i, v in enumerate(study["variants"]) if v.get("control"))
    for ax, dataset in zip(axes.flat, datasets):
        rows = _rows_for(summary, study, dataset)
        ax.axvspan(control - 0.4, control + 0.4, color=BAND, linewidth=0, zorder=0)
        for offset, (series, metric) in zip((-0.09, 0.09), (("validation", "val_macro_f1"),
                                                            ("test", "test_macro_f1"))):
            xs, ys, es = [], [], []
            for position, variant in zip(positions, study["variants"]):
                item = rows.get(variant["id"])
                if item and item["n"] and item[f"{metric}_mean"] is not None:
                    xs.append(position + offset)
                    ys.append(item[f"{metric}_mean"])
                    es.append(item[f"{metric}_sd"])
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=SERIES[series], marker=MARKERS[series],
                            markersize=4.5, markeredgecolor="white", markeredgewidth=0.8,
                            linewidth=1.5, elinewidth=0.8, capsize=2, label=series.title(),
                            solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.set_xticks(list(positions), labels)
        ax.set_title(DISPLAY.get(dataset, dataset), loc="left", color=INK)
        _style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1 (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel(study.get("x_label", study.get("parameter", "")))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{study['title']}: mean ± SD over repeats (shaded: base setting)",
                 x=0.0, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path_stem)
    plt.close(fig)


def sweep_cost_figure(manifest, study, summary, path_stem, *, field, ylabel, title):
    """One single-series panel per dataset; log scale for counts and seconds."""
    plt = _pyplot()
    datasets = list(manifest["entries"])
    fig, axes = _grid(plt, len(datasets), height=1.8)
    positions = list(range(len(study["variants"])))
    control = next(i for i, v in enumerate(study["variants"]) if v.get("control"))
    for ax, dataset in zip(axes.flat, datasets):
        rows = _rows_for(summary, study, dataset)
        ax.axvspan(control - 0.4, control + 0.4, color=BAND, linewidth=0, zorder=0)
        xs, ys = [], []
        for position, variant in zip(positions, study["variants"]):
            item = rows.get(variant["id"])
            value = item.get(f"{field}_median") if item else None
            if value:
                xs.append(position)
                ys.append(value)
        if xs:
            ax.plot(xs, ys, color=SERIES["validation"], marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, linewidth=1.5, zorder=3)
            ax.set_yscale("log")
        ax.set_xticks(positions, [str(v["x"]) for v in study["variants"]])
        ax.set_title(DISPLAY.get(dataset, dataset), loc="left", color=INK)
        _style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[-1, :]:
        ax.set_xlabel(study.get("x_label", study.get("parameter", "")))
    fig.suptitle(title, x=0.0, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path_stem)
    plt.close(fig)


def component_figure(manifest, study, summary, path_stem):
    """Paired test Macro-F1 difference from the control, one panel per dataset."""
    plt = _pyplot()
    datasets = list(manifest["entries"])
    variants = [v for v in study["variants"] if not v.get("control")]
    fig, axes = _grid(plt, len(datasets), height=0.32 * len(variants) + 0.9)
    for ax, dataset in zip(axes.flat, datasets):
        rows = _rows_for(summary, study, dataset)
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
        for position, variant in enumerate(variants):
            item = rows.get(variant["id"])
            if item and item.get("delta_test_macro_f1_mean") is not None:
                ax.errorbar(item["delta_test_macro_f1_mean"], position,
                            xerr=item["delta_test_macro_f1_sd"], color=SERIES["validation"],
                            marker="o", markersize=4.5, markeredgecolor="white",
                            markeredgewidth=0.8, elinewidth=0.9, capsize=2, zorder=3)
        ax.set_yticks(range(len(variants)), [_variant_label(study, v) for v in variants])
        ax.set_ylim(len(variants) - 0.5, -0.5)
        ax.margins(x=0.15)
        ax.set_title(DISPLAY.get(dataset, dataset), loc="left", color=INK)
        _style_axis(ax, grid_axis="x")
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    for row in axes:
        for ax in row[1:]:
            ax.tick_params(labelleft=False)
    fig.supxlabel("Paired difference in test Macro-F1 from the full model (points)",
                  fontsize=8, color=INK_2)
    fig.suptitle(f"{study['title']}: paired difference from the full model, mean ± SD",
                 x=0.0, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path_stem)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_text(buffer.getvalue(), encoding="utf-8")


def write_report(manifest, results_dir, output_dir, *, figures=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, coverage = load_suite_records(manifest, results_dir)
    summary = summarize(manifest, records)
    checks = reproduction_check(manifest, results_dir, records)
    _write_csv(output_dir / "records.csv", records)
    _write_csv(output_dir / "summary.csv", summary)
    spec = manifest["spec"]
    files = {}
    for study in spec["studies"]:
        outputs = [f"{study['id']}.md"]
        (output_dir / f"{study['id']}.md").write_text(
            study_markdown(manifest, study, summary), encoding="utf-8")
        if figures:
            stem = output_dir / study["id"]
            if study["type"] == "sweep":
                sweep_figure(manifest, study, summary, str(stem) + "_macro_f1")
                outputs += [f"{study['id']}_macro_f1.pdf", f"{study['id']}_macro_f1.png"]
                tiers = {run["tier"] for entry in manifest["entries"].values()
                         for run in entry["runs"].values()
                         if any(use[0] == study["id"] for use in run["uses"])}
                if "mining" in tiers:
                    sweep_cost_figure(manifest, study, summary, str(stem) + "_mining_time",
                                      field="mining_seconds", ylabel="Seconds per repeat",
                                      title=f"{study['title']}: median mining time per repeat")
                    sweep_cost_figure(manifest, study, summary, str(stem) + "_candidates",
                                      field="candidates", ylabel="Candidates",
                                      title=f"{study['title']}: median candidate patterns")
                    outputs += [f"{study['id']}_mining_time.pdf", f"{study['id']}_candidates.pdf"]
            else:
                component_figure(manifest, study, summary, str(stem) + "_delta")
                outputs += [f"{study['id']}_delta.pdf", f"{study['id']}_delta.png"]
        files[study["id"]] = outputs
    (output_dir / "README.md").write_text(
        readme(manifest, coverage, checks, files), encoding="utf-8")
    return {"records": len(records), "coverage": coverage, "checks": checks}


def readme(manifest, coverage, checks, files):
    expected = len(manifest["repeats"])
    lines = [f"# Ablation suite {manifest['suite_id']}", "",
             f"Base run `{manifest['base_run']}`; protocol `{manifest['protocol']}`; "
             f"repeats {manifest['repeats']}; created {manifest['created_at']}.", "",
             manifest.get("selection_policy", ""), "",
             "Each variant changes one factor of the base configuration. Paired differences "
             "are variant minus control within a repeat. Five overlapping repeats support "
             "descriptive comparison, not significance claims.", "",
             "## Run coverage", ""]
    complete = sum(item["status"] == "complete" for item in coverage)
    lines.append(f"{complete}/{len(coverage)} dataset runs complete.")
    problems = [item for item in coverage
                if item["status"] != "complete" or item["repeats"] != expected or item["note"]]
    if problems:
        lines += ["", _markdown_table(
            ["Dataset", "Run", "Tier", "Status", "Repeats", "Note"],
            [[item["dataset"], item["run_id"], item["tier"], item["status"],
              f"{item['repeats']}/{expected}", item["note"] or ""] for item in problems])]
    lines += ["", "## Control reproduces the base run", "",
              "Maximum absolute difference across repeats in test Macro-F1, test accuracy and "
              "validation Macro-F1 (points). Zero is expected when the implementation "
              "identity matches the base run.", "",
              _markdown_table(["Dataset", "Repeats compared", "Max absolute difference"],
                              [[DISPLAY.get(c["dataset"], c["dataset"]), str(c["compared"]),
                                "--" if c["max_abs_diff"] is None else f"{c['max_abs_diff']:.4f}"]
                               for c in checks]),
              "", "## Studies", ""]
    for study in manifest["spec"]["studies"]:
        lines.append(f"- **{study['id']}**: {study['title']} — "
                     + ", ".join(f"`{name}`" for name in files.get(study["id"], [])))
    lines += ["", "Regenerate with `python scripts/ablation/report_ablations.py --suite "
              f"{manifest['suite_id']}`.", ""]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <results-dir>/ablation/<suite>/report")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.resolve()
    manifest_file = results_dir / "ablation" / args.suite / "suite.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"No frozen suite manifest at {manifest_file}")
    manifest = read_json(manifest_file)
    output_dir = args.output_dir or manifest_file.parent / "report"
    result = write_report(manifest, results_dir, output_dir, figures=not args.no_figures)
    complete = sum(item["status"] == "complete" for item in result["coverage"])
    print(f"Report written to {output_dir} ({complete}/{len(result['coverage'])} runs complete, "
          f"{result['records']} repeat records)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
