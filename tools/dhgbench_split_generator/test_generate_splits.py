"""Run with: uv run -m unittest test_generate_splits.py"""
from pathlib import Path
import unittest
from generate_splits import (
    DEFAULT_REPO_ROOT,
    build_parser,
    canonical_hash,
    format_completion_message,
)


class CanonicalHashTests(unittest.TestCase):
    def test_ordered_arrays_affect_hash(self):
        self.assertNotEqual(canonical_hash({"train": [1, 0], "validation": [], "test": []}),
                            canonical_hash({"train": [0, 1], "validation": [], "test": []}))


class CompletionMessageTests(unittest.TestCase):
    def test_completion_message_identifies_outputs_and_split_sizes(self):
        summary = {
            "destinations": [Path("repeat_0.json"), Path("repeat_1.json")],
            "output_root": Path("splits"),
            "total_graphs": 101,
            "train_size": 80,
            "validation_size": 10,
            "test_size": 11,
        }
        message = format_completion_message(
            "imdb_dir_genre", "composite", [0, 1], summary
        )
        self.assertIn("SPLIT GENERATION COMPLETE: 2 manifest(s)", message)
        self.assertIn("repeats: 0, 1", message)
        self.assertIn("train=80, validation=10, test=11 of 101 graphs", message)
        self.assertIn("Output directory:", message)


class CommandLineDefaultTests(unittest.TestCase):
    def test_project_defaults_require_only_the_dataset(self):
        args = build_parser().parse_args(["--dataset", "imdb_dir_genre"])
        self.assertEqual(Path(args.repo_root), DEFAULT_REPO_ROOT)
        self.assertEqual(args.label_mode, "composite")
        self.assertEqual(args.repeats, [0, 1, 2, 3, 4])

    def test_all_datasets_uses_the_same_project_defaults(self):
        args = build_parser().parse_args(["--all-datasets"])
        self.assertTrue(args.all_datasets)
        self.assertIsNone(args.dataset)
        self.assertEqual(args.label_mode, "composite")
        self.assertEqual(args.repeats, [0, 1, 2, 3, 4])
