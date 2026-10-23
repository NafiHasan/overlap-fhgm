"""Read-only validation of an archived run, repeat, or mining-cache bundle."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.artifacts import valid_completion_marker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        with (args.directory / "completion.json").open(encoding="utf-8") as handle:
            marker = json.load(handle)
        stage = marker["stage"]
        if stage not in {
                "fully_inductive_run", "inductive_classification_repeat",
                "inductive_mining_cache", "transductive_vocabulary_run",
                "transductive_classification_repeat",
                "transductive_mining_cache",
        }:
            raise ValueError(f"Unsupported stage: {stage}")
        if not valid_completion_marker(args.directory, stage=stage):
            raise ValueError("Missing, changed, or inconsistent artifacts in the completion chain")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"INVALID: {args.directory}: {exc}", file=sys.stderr)
        return 1
    print(f"VALID: {stage}: {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
