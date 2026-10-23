"""
FHGM (Frequent HyperGraph Mining) with overlap labels.

Depth-first frequent sub-hypergraph mining with canonical codes (Alam et al.,
2021), extended with overlap labels and a support threshold that depends on
pattern size. Support is lowered in phases until every training hypergraph
contains enough patterns; hypergraphs still uncovered then get patterns of
their own (fallback).
"""

import copy
import math
import time
import logging
import threading
from collections import Counter
from dataclasses import dataclass

from .candidate import CandidateCode, get_extensions, extend_code, is_minimum

logger = logging.getLogger(__name__)


@dataclass
class MiningResult:
    patterns: list
    train_coverage: dict
    statistics: dict
    report_markdown: str

_HEARTBEAT_INTERVAL = 60     # seconds between progress logs
_EXTENSION_LOG_INTERVAL = 100


class _NullPatternWriter:
    """Discards the per-pattern text output when no file is requested."""
    def write(self, _text):
        return 0


def _elapsed(start):
    """Return a human-readable elapsed time string."""
    elapsed = time.time() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    elif elapsed < 3600:
        return f"{elapsed/60:.1f}m"
    else:
        return f"{elapsed/3600:.1f}h"


def _format_duration(elapsed):
    """Format an already-computed duration in seconds."""
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    if elapsed < 3600:
        return f"{elapsed / 60:.1f}m"
    return f"{elapsed / 3600:.1f}h"


OVERLAP_LABEL_MODES = ("none", "log2", "exact")


def resolve_pattern_options(config):
    """Overlap labels and the small-pattern support schedule.

    Returns ``(overlap_binning, exhaustive_depth, exhaustive_min_graphs)``.
    ``overlap_binning`` is None when ``overlap_labels`` is "none" (default).
    With ``exhaustive_depth`` k > 0, phase DFS patterns of at most k
    incidences need only ``exhaustive_min_graphs`` supporting graphs (or the
    phase threshold, if lower); deeper patterns keep the phase threshold.
    The threshold never decreases with depth, so support pruning stays exact.
    """
    overlap = config.get("overlap_labels") or "none"
    if overlap not in OVERLAP_LABEL_MODES:
        raise ValueError(f"overlap_labels must be one of {OVERLAP_LABEL_MODES}")
    depth = config.get("exhaustive_depth") or 0
    min_graphs = config.get("exhaustive_min_graphs", 2)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("exhaustive_depth must be a non-negative integer")
    if (isinstance(min_graphs, bool) or not isinstance(min_graphs, int)
            or min_graphs < 1):
        raise ValueError("exhaustive_min_graphs must be a positive integer")
    return (None if overlap == "none" else overlap), depth, min_graphs


def resolve_min_coverage(config):
    """Number of patterns each training hypergraph should contain."""
    value = config.get("min_coverage")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("min_coverage must be a positive integer")
    return value


def _coverage_summary(values, coverage_target):
    """Summarize a per-graph coverage vector without persisting every value."""
    values = list(values)
    distribution = Counter(int(value) for value in values)
    return {
        "graphs": len(values),
        "minimum": min(values) if values else 0,
        "maximum": max(values) if values else 0,
        "mean": (sum(values) / len(values)) if values else 0.0,
        "zero": distribution.get(0, 0),
        "below_target": sum(count for value, count in distribution.items()
                            if value < coverage_target),
        "at_or_above_target": sum(count for value, count in distribution.items()
                                  if value >= coverage_target),
        "distribution": {str(value): distribution[value]
                         for value in sorted(distribution)},
    }


def _matched_coverage_summary(train_coverage, graph_ids, coverage_target):
    """Summarize exact pattern hits for a fixed set of training graph IDs."""
    matched = {int(graph_id): 0 for graph_id in graph_ids}
    for hits in train_coverage.values():
        for graph_id in hits:
            matched[int(graph_id)] += 1
    return _coverage_summary(matched.values(), coverage_target)


class FHGMMiner:
    """
    The FHGM miner with its progress counters.

    Attributes:
        found (int): Number of frequent patterns found so far.
        candidates (int): Number of candidate patterns evaluated.
        start_time (float): Timestamp when mining started.
        out_file: File handle for writing frequent patterns.
    """

    def __init__(self):
        self.found = 0
        self.candidates = 0
        self.start_time = 0
        self.out_file = None
        self._depth = 0
        # Per-depth tracking for console summary
        self._depth_patterns = {}   # depth -> patterns found at this depth
        self._depth_time = {}       # depth -> cumulative seconds spent at this depth
        self._extension_calls = 0
        self._stage = "idle"
        self._phase = 0
        self._phase_count = 0
        self._phase_pool_size = 0
        self._phase_min_sup = 0
        self._phase_uncovered = 0
        self._phase_below_target = 0
        self._active_extension_call = 0
        self._active_extension_started = 0.0
        self._fallback_processed = 0
        self._fallback_total = 0
        self._fallback_candidates = 0
        self._matching_completed = 0
        self._matching_total = 0
        self._overlap_binning = None
        self._exhaustive = (0, 2)

    def _new_code(self):
        return CandidateCode(self._overlap_binning)

    def _required_support(self, depth, phase_min_sup):
        """Phase support threshold for a pattern with ``depth`` incidences."""
        exhaustive_depth, min_graphs = self._exhaustive
        if exhaustive_depth and depth <= exhaustive_depth:
            return min(min_graphs, phase_min_sup)
        return phase_min_sup

    def _match_training_coverage(self, patterns, hypergraphs, graph_ids):
        """Match a frozen pattern vocabulary against every training graph."""
        train_coverage = {}
        for pattern_id, pattern in enumerate(patterns):
            hits = {}
            # CandidateCode.get_hypergraph() is deterministic, and the matcher
            # does not mutate the pattern graph. Build it once per pattern
            # instead of once for every pattern/graph pair.
            pattern_graph = pattern.get_hypergraph()
            for local_index, graph in enumerate(hypergraphs):
                try:
                    if pattern_graph.is_subhypergraph_of(graph):
                        hits[int(graph_ids[local_index])] = 1
                except MemoryError as exc:
                    raise MemoryError(
                        "Training coverage exhausted memory: "
                        f"pattern={pattern_id}, graph={graph_ids[local_index]}"
                    ) from exc
            if hits:
                train_coverage[pattern_id] = hits
            self._matching_completed = pattern_id + 1
        return train_coverage

    def _watchdog_message(self, now=None):
        """Build one concise stage-aware progress heartbeat."""
        now = time.time() if now is None else now
        common = f"elapsed={_elapsed(self.start_time)}"
        if self._stage == "phase_dfs":
            active = ""
            if self._active_extension_call:
                active = (
                    f", extension_call={self._active_extension_call} "
                    f"({_format_duration(now - self._active_extension_started)})"
                )
            return (
                f"Mining progress: phase={self._phase}/{self._phase_count}, "
                f"stage=DFS, depth={self._depth}, pool={self._phase_pool_size}, "
                f"min_sup={self._phase_min_sup}, "
                f"uncovered={self._phase_uncovered}, "
                f"below_target={self._phase_below_target}, {common}{active}"
            )
        if self._stage == "fallback":
            return (
                f"Mining progress: stage=fallback, "
                f"graphs={self._fallback_processed}/{self._fallback_total}, "
                f"fallback_candidates={self._fallback_candidates}, {common}"
            )
        if self._stage == "train_matching":
            return (
                f"Mining progress: stage=train_matching, "
                f"patterns={self._matching_completed}/{self._matching_total}, "
                f"elapsed={_elapsed(self.start_time)}"
            )
        return f"Mining progress: stage={self._stage}, {common}"

    def _watchdog_loop(self, stop_event):
        """Emit progress independently of potentially blocking mining calls."""
        while not stop_event.wait(_HEARTBEAT_INTERVAL):
            logger.info(self._watchdog_message())

    def mine_hypergraphs(self, hypergraphs, graph_ids, config, *, domain_name,
                         pattern_text_path=None):
        """Mine a vocabulary from the supplied hypergraphs.

        The search works with local indices, which are translated to the
        original dataset IDs in the result. Mining uses graph structure only;
        it never sees class labels.
        """
        if len(hypergraphs) != len(graph_ids) or len(set(graph_ids)) != len(graph_ids):
            raise ValueError("hypergraphs and unique graph_ids must have equal lengths")
        if not hypergraphs:
            raise ValueError("Cannot mine an empty discovery population")
        self.found = self.candidates = self._depth = 0
        self._extension_calls = 0
        self.start_time = time.time()
        self._depth_patterns, self._depth_time = {}, {}
        self._phase_count = config["max_phases"]
        self._stage = "setup"
        self._phase_uncovered = self._phase_below_target = 0
        self._active_extension_call = 0
        self._active_extension_started = 0.0
        self._fallback_processed = self._fallback_total = 0
        self._fallback_candidates = 0
        self._matching_completed = self._matching_total = 0
        coverage, patterns, seen, phase_details = [0] * len(hypergraphs), [], set(), []
        phase_time = 0.0
        min_coverage = resolve_min_coverage(config)
        base_sup, n_total = config["base_sup"], len(hypergraphs)
        overlap_binning, exhaustive_depth, exhaustive_min_graphs = (
            resolve_pattern_options(config)
        )
        self._overlap_binning = overlap_binning
        self._exhaustive = (exhaustive_depth, exhaustive_min_graphs)
        handle = open(pattern_text_path, "w", encoding="utf-8") if pattern_text_path else None
        self.out_file = handle if handle is not None else _NullPatternWriter()
        watchdog_stop = threading.Event()
        watchdog = None
        try:
            watchdog = threading.Thread(
                target=self._watchdog_loop,
                args=(watchdog_stop,),
                name="fhgm-progress-watchdog",
                daemon=True,
            )
            watchdog.start()
            logger.info(
                "Mining start: %s (%d discovery graphs, base_sup=%.4f, floor=%s, "
                "max_depth=%s, min_coverage=%d)",
                domain_name,
                n_total,
                base_sup,
                config["min_sup_floor"],
                config.get("max_depth"),
                min_coverage,
            )
            if overlap_binning or exhaustive_depth:
                logger.info(
                    "Pattern options: overlap_labels=%s, exhaustive_depth=%d, "
                    "exhaustive_min_graphs=%d",
                    overlap_binning or "none", exhaustive_depth,
                    exhaustive_min_graphs,
                )
            last_phase_search = None
            for phase in range(config["max_phases"]):
                self._depth_patterns, self._depth_time = {}, {}
                subset_ids = [i for i, value in enumerate(coverage) if value < min_coverage]
                subset = [hypergraphs[i] for i in subset_ids]
                threshold = math.ceil(len(subset) * base_sup)
                floor = config["min_sup_floor"]
                floor = math.ceil(n_total * floor) if isinstance(floor, float) and floor < 1 else floor
                if not subset:
                    logger.info(
                        "Phase %d: all graphs reached the phase target; stopping",
                        phase + 1,
                    )
                    break
                if threshold < floor:
                    logger.info(
                        "Phase %d: min_sup=%d < floor=%d with pool=%d; stopping",
                        phase + 1, threshold, floor, len(subset),
                    )
                    break
                # Set phase number only after confirming this phase will run,
                # so the watchdog never reports a phase that exited immediately.
                self._phase = phase + 1
                self._stage = "phase_dfs"
                self._phase_pool_size = len(subset)
                self._phase_min_sup = threshold
                phase_coverage_before = _coverage_summary(coverage, min_coverage)
                self._phase_uncovered = phase_coverage_before["zero"]
                self._phase_below_target = phase_coverage_before["below_target"]
                logger.info(
                    "Phase %d/%d start: pool=%d/%d, min_sup=%d, floor=%d, "
                    "base_sup=%.4f, uncovered=%d, below_target=%d",
                    phase + 1,
                    config["max_phases"],
                    len(subset),
                    n_total,
                    threshold,
                    floor,
                    base_sup,
                    phase_coverage_before["zero"],
                    phase_coverage_before["below_target"],
                )
                phase_start, before = time.time(), len(patterns)
                # An unchanged cohort and integer threshold repeat the same
                # exhaustive search, even if its first run added patterns.
                # Still advance the schedule: later discounts can cross a
                # rounding plateau and make previously infrequent motifs valid.
                phase_search = (tuple(subset_ids), threshold)
                dfs_skipped = phase_search == last_phase_search
                if dfs_skipped:
                    logger.info(
                        "Phase %d: unchanged pool and min_sup=%d; skipping DFS",
                        phase + 1, threshold,
                    )
                    codes = []
                else:
                    codes = self._fhgm(self._new_code(), subset, threshold, config.get("max_depth"))
                    last_phase_search = phase_search
                for code, _, local_support in codes:
                    if code in seen:
                        continue
                    seen.add(code); patterns.append(code)
                    for local_index in local_support:
                        coverage[subset_ids[local_index]] += 1
                elapsed = time.time() - phase_start
                phase_time += elapsed
                covered = sum(c > 0 for c in coverage)
                phase_details.append({"phase": phase + 1, "min_sup": threshold,
                                      "dfs_skipped": dfs_skipped,
                                      "patterns_found": len(patterns) - before,
                                      "total_patterns": len(patterns),
                                      "coverage": covered,
                                      "uncovered": n_total - covered,
                                      "time_seconds": round(elapsed, 3)})
                for depth in sorted(self._depth_patterns):
                    logger.info(
                        "  depth=%d: %d patterns, %s",
                        depth,
                        self._depth_patterns[depth],
                        _format_duration(self._depth_time.get(depth, 0.0)),
                    )
                phase_coverage_after = _coverage_summary(coverage, min_coverage)
                logger.info(
                    "Phase %d complete: +%d unique patterns (%d total), "
                    "uncovered=%d, below_target=%d, at/above=%d, time=%s",
                    phase + 1,
                    len(patterns) - before,
                    len(patterns),
                    phase_coverage_after["zero"],
                    phase_coverage_after["below_target"],
                    phase_coverage_after["at_or_above_target"],
                    _format_duration(elapsed),
                )
                if len(patterns) == before:
                    logger.info(
                        "Phase %d added no new unique patterns; advancing support schedule",
                        phase + 1,
                    )
                base_sup -= config.get("sup_discount", .01)
            phase_patterns = len(patterns)
            post_phase_coverage = _coverage_summary(coverage, min_coverage)
            logger.info(
                "Phase mining complete: %d patterns, uncovered=%d, "
                "below_target=%d, time=%s",
                phase_patterns,
                post_phase_coverage["zero"],
                post_phase_coverage["below_target"],
                _format_duration(phase_time),
            )

            fallback_start = time.time()
            # Phase counters track cohort work. Fallback eligibility uses exact
            # matching of the vocabulary against every discovery graph.
            self._stage = "train_matching"
            self._matching_total = len(patterns)
            self._matching_completed = 0
            existing_coverage = self._match_training_coverage(
                patterns, hypergraphs, graph_ids
            )
            root_to_local = {int(graph_id): index for index, graph_id in enumerate(graph_ids)}
            coverage = [0] * n_total
            for hits in existing_coverage.values():
                for graph_id in hits:
                    coverage[root_to_local[graph_id]] += 1
            # Hypergraphs that contain no pattern get patterns of their own:
            # a breadth-first search over their own extensions, up to
            # max_depth, until they contain min_coverage patterns. Each new
            # pattern is credited to every uncovered hypergraph containing it.
            fallback_indices = [
                index for index, value in enumerate(coverage) if value == 0
            ]
            self._stage = "fallback"
            self._fallback_total = len(fallback_indices)
            self._fallback_processed = 0
            fallback_before = len(patterns)
            logger.info(
                "Fallback start: target=%d, eligible_graphs=%d",
                min_coverage,
                self._fallback_total,
            )
            for local_index in fallback_indices:
                if coverage[local_index] == 0:
                    graph = hypergraphs[local_index]
                    queue = [self._new_code()]
                    while queue and coverage[local_index] < min_coverage:
                        current = queue.pop(0)
                        if (config.get("max_depth") is not None and
                                len(current.get_extensions()) >= config["max_depth"]):
                            continue
                        try:
                            extensions = get_extensions(current, [graph])
                        except MemoryError as exc:
                            raise MemoryError(
                                "Fallback mining exhausted memory: "
                                f"graph={graph_ids[local_index]}"
                            ) from exc
                        for extension in extensions:
                            self._fallback_candidates += 1
                            candidate = copy.deepcopy(current)
                            if (candidate.extend(extension) is False or
                                    not is_minimum(candidate)):
                                continue
                            queue.append(candidate)
                            if candidate not in seen:
                                seen.add(candidate)
                                patterns.append(candidate)
                                hits = self._match_training_coverage(
                                    [candidate],
                                    [hypergraphs[index] for index in fallback_indices],
                                    [graph_ids[index] for index in fallback_indices],
                                )
                                for graph_id in hits.get(0, {}):
                                    coverage[root_to_local[graph_id]] += 1
                                if coverage[local_index] >= min_coverage:
                                    break
                self._fallback_processed += 1
                if (self._fallback_processed % 100 == 0 or
                        self._fallback_processed == self._fallback_total):
                    logger.info(
                        "Fallback progress: %d/%d graphs, +%d patterns, "
                        "candidates=%d, elapsed=%s",
                        self._fallback_processed,
                        self._fallback_total,
                        len(patterns) - fallback_before,
                        self._fallback_candidates,
                        _format_duration(time.time() - fallback_start),
                    )
            fallback_patterns = len(patterns) - fallback_before
            fallback_time = time.time() - fallback_start
            post_fallback_coverage = _coverage_summary(coverage, min_coverage)
            logger.info(
                "Fallback complete: +%d patterns, candidates=%d, uncovered=%d, "
                "below_target=%d, time=%s",
                fallback_patterns,
                self._fallback_candidates,
                post_fallback_coverage["zero"],
                post_fallback_coverage["below_target"],
                _format_duration(fallback_time),
            )
            # Recompute every pattern against the discovery population,
            # producing exact binary presence.
            coverage_start = time.time()
            self._stage = "train_matching"
            self._matching_total = len(patterns)
            self._matching_completed = 0
            logger.info(
                "Discovery matching start: %d patterns x %d graphs",
                len(patterns), n_total,
            )
            train_coverage = self._match_training_coverage(
                patterns, hypergraphs, graph_ids
            )
            coverage_time = time.time() - coverage_start
            logger.info(
                "Discovery matching complete: %d patterns x %d graphs, time=%s",
                len(patterns), n_total, _format_duration(coverage_time),
            )

        finally:
            self._stage = "cleanup"
            watchdog_stop.set()
            if watchdog is not None:
                watchdog.join(timeout=1.0)
            if handle:
                handle.close()
            self.out_file = None
        matched_coverage_summary = _matched_coverage_summary(
            train_coverage, graph_ids, min_coverage
        )
        covered_graphs = n_total - matched_coverage_summary["zero"]
        total_time = time.time() - self.start_time
        logger.info(
            "Mining complete: %d patterns, uncovered=%d, below_target=%d, "
            "covered=%d/%d, total_time=%s",
            len(patterns),
            matched_coverage_summary["zero"],
            matched_coverage_summary["below_target"],
            covered_graphs,
            n_total,
            _format_duration(total_time),
        )
        stats = {"patterns": len(patterns),
                 "phase_patterns": phase_patterns,
                 "fallback_patterns": fallback_patterns,
                 "phases": phase_details,
                 "graphs": n_total, "covered_graphs": covered_graphs,
                 "min_coverage": min_coverage,
                 "fallback_max_depth": config.get("max_depth"),
                 "fallback_pattern_depths": dict(Counter(
                     len(pattern.get_extensions())
                     for pattern in patterns[phase_patterns:]
                 )),
                 "overlap_labels": overlap_binning or "none",
                 "exhaustive_depth": exhaustive_depth,
                 "exhaustive_min_graphs": exhaustive_min_graphs,
                 "pattern_depths": {str(depth): count for depth, count in sorted(Counter(
                     len(pattern.get_extensions()) for pattern in patterns).items())},
                 "fallback_eligible_graphs": len(fallback_indices),
                 "post_phase_coverage": post_phase_coverage,
                 "post_fallback_working_coverage": post_fallback_coverage,
                 "matched_train_coverage": matched_coverage_summary,
                 "candidates": self.candidates,
                 "fallback_candidates": self._fallback_candidates,
                 "phase_seconds": phase_time,
                 "fallback_seconds": fallback_time,
                 "coverage_seconds": coverage_time,
                 "total_seconds": total_time}
        report_lines = [
            f"# Mining Report: {domain_name}", "",
            f"Discovery graphs: {n_total}", "",
            f"Coverage target: {min_coverage} pattern(s) per graph.", "",
            "| Patterns | Phase | Fallback | Discovery coverage | Total time |",
            "|---:|---:|---:|---:|---:|",
            f"| {len(patterns)} | {phase_patterns} | {fallback_patterns} | "
            f"{covered_graphs}/{n_total} | {_format_duration(total_time)} |",
            "", "| Phase candidates | Fallback candidates |",
            "|---:|---:|",
            f"| {self.candidates} | {self._fallback_candidates} |",
            "", "## Mining phases", "",
        ]
        if phase_details:
            report_lines.extend([
                "| Phase | min_sup | New patterns | Total patterns | Covered | Uncovered | Time | DFS |",
                "|---:|---:|---:|---:|---:|---:|---:|---|",
            ])
            for item in phase_details:
                report_lines.append(
                    f"| {item['phase']} | {item['min_sup']} | {item['patterns_found']} | "
                    f"{item['total_patterns']} | {item['coverage']} | {item['uncovered']} | "
                    f"{_format_duration(item['time_seconds'])} | "
                    f"{'skipped (unchanged search)' if item['dfs_skipped'] else 'run'} |"
                )
        else:
            report_lines.append(
                "No mining phase completed before fallback."
            )
        report_lines.extend([
            "", "## Timing", "",
            f"Phase mining: {_format_duration(phase_time)}; "
            f"fallback: {_format_duration(fallback_time)}; "
            f"discovery matching: {_format_duration(coverage_time)}.", "",
            "## Coverage", "",
            "| Point | Target | Zero | Below target | At/above target | Mean | Range |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        coverage_points = [
            ("After phases", min_coverage, post_phase_coverage),
            ("After fallback (working counter)", min_coverage, post_fallback_coverage),
            ("After final discovery matching", min_coverage,
             stats["matched_train_coverage"]),
        ]
        for label, target, summary in coverage_points:
            report_lines.append(
                f"| {label} | {target} | {summary['zero']} | {summary['below_target']} | "
                f"{summary['at_or_above_target']} | {summary['mean']:.2f} | "
                f"{summary['minimum']}--{summary['maximum']} |"
            )
        report_lines.append("")
        report = "\n".join(report_lines)
        return MiningResult(patterns, train_coverage, stats, report)

    def _fhgm(self, code, hypergraphs, min_sup, max_depth=None,
              root_graph_ids=None):
        """
        Core FHGM recursive algorithm.

        Extends a sub-hypergraph candidate in a DFS manner, checking for
        canonical ordering (to avoid duplicates) and minimum support threshold.

        Args:
            code (CandidateCode): Current sub-hypergraph candidate.
            hypergraphs (list): Database of hypergraphs to mine from.
            min_sup (int): Minimum support threshold.
            max_depth (int or None): Maximum recursion depth. None = unlimited.
            root_graph_ids (list or None): For each graph in ``hypergraphs``,
                its index in the phase-root graph list. ``None`` initializes
                the identity mapping for the root call.

        Returns:
            list: List of (CandidateCode, pattern_id, graph_indices) tuples
                  for frequent patterns. ``graph_indices`` always refers to the
                  phase-root ``hypergraphs`` list, even after recursive pruning.
        """
        if root_graph_ids is None:
            root_graph_ids = list(range(len(hypergraphs)))
        elif len(root_graph_ids) != len(hypergraphs):
            raise ValueError(
                "root_graph_ids must align one-to-one with hypergraphs"
            )

        self._depth += 1
        _enter_time = time.time()

        # Initialize tracking for this depth level
        if self._depth not in self._depth_patterns:
            self._depth_patterns[self._depth] = 0
            self._depth_time[self._depth] = 0.0

        # Depth limit check
        if max_depth is not None and self._depth > max_depth:
            logger.debug(f"Depth limit ({max_depth}) reached at depth={self._depth}")
            self._depth_time[self._depth] = self._depth_time.get(self._depth, 0.0) + (time.time() - _enter_time)
            self._depth -= 1
            return []

        # Find possible extensions for the sub-hypergraph
        t0 = time.time()
        self._extension_calls += 1
        self._active_extension_call = self._extension_calls
        self._active_extension_started = t0
        try:
            extensions = get_extensions(code, hypergraphs, track_graph_ids=True)
        finally:
            ext_time = time.time() - t0
            self._active_extension_call = 0
            self._active_extension_started = 0.0

        if (
            self._extension_calls == 1
            or self._extension_calls % _EXTENSION_LOG_INTERVAL == 0
            or ext_time >= 5.0
        ):
            logger.debug(
                "EXTEND call=%d depth=%d extensions=%d graphs=%d (%.1fs)",
                self._extension_calls,
                self._depth,
                len(extensions),
                len(hypergraphs),
                ext_time,
            )

        # Try and check each extension
        codes = extend_code(code, extensions, tracked=True)

        results = []
        for idx, c_tuple in enumerate(codes):
            self.candidates += 1
            c, support, local_graph_ids = c_tuple

            if is_minimum(c) is False:  # Canonical ordering check
                continue
            if support < self._required_support(len(c.get_extensions()), min_sup):
                continue
            if support != len(local_graph_ids):
                raise RuntimeError(
                    "Extension support count and graph IDs disagree"
                )

            graph_ids = [root_graph_ids[index] for index in local_graph_ids]

            self.out_file.write("t # " + str(self.found) + "\n")
            self.out_file.write(str(c.get_hypergraph()) + "\n")
            results.append((c, self.found, graph_ids))
            self.found += 1
            self._depth_patterns[self._depth] = self._depth_patterns.get(self._depth, 0) + 1

            child_start = time.time()
            child_hypergraphs = [
                hypergraphs[index] for index in local_graph_ids
            ]
            result = self._fhgm(
                c,
                child_hypergraphs,
                min_sup,
                max_depth,
                root_graph_ids=graph_ids,
            )
            # Subtract child recursion time so this depth only counts its own work
            _enter_time += (time.time() - child_start)
            for res in result:
                results.append(res)

        # Accumulate time spent at this depth level (child time already subtracted)
        self._depth_time[self._depth] = self._depth_time.get(self._depth, 0.0) + (time.time() - _enter_time)
        self._depth -= 1

        return results
