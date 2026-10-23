"""Frozen-vocabulary matching for validation and test hypergraphs."""

import logging
import time

logger = logging.getLogger(__name__)


def vocabulary_signature(patterns):
    return tuple(tuple(code.get_extensions()) for code in patterns)


def match_pattern_vocabulary(patterns, hypergraphs, graph_ids, *, partition_name):
    if len(hypergraphs) != len(graph_ids) or len(set(graph_ids)) != len(graph_ids):
        raise ValueError("hypergraphs and unique graph_ids must have equal lengths")
    before = vocabulary_signature(patterns)
    coverage = {}
    started = time.time()
    progress_interval = 500
    for pattern_id, code in enumerate(patterns):
        matched = {}
        pattern_graph = code.get_hypergraph()
        for local_index, hypergraph in enumerate(hypergraphs):
            try:
                present = pattern_graph.is_subhypergraph_of(hypergraph)
            except MemoryError as exc:
                raise MemoryError(f"Frozen matching exhausted memory: pattern={pattern_id}, graph={graph_ids[local_index]}, partition={partition_name}") from exc
            if present:
                matched[int(graph_ids[local_index])] = 1
        if matched:
            coverage[pattern_id] = matched
        completed = pattern_id + 1
        if completed < len(patterns) and completed % progress_interval == 0:
            logger.debug(
                "Frozen matching progress: %d/%d patterns against %s",
                completed,
                len(patterns),
                partition_name,
            )
    if vocabulary_signature(patterns) != before:
        raise RuntimeError("Frozen vocabulary was mutated during matching")
    presence_count = sum(len(entries) for entries in coverage.values())
    logger.info(
        "Frozen matching complete: %s (%d patterns x %d graphs, %d presences, %.1fs)",
        partition_name,
        len(patterns),
        len(hypergraphs),
        presence_count,
        time.time() - started,
    )
    return coverage


def merge_binary_coverage(patterns, partitions, *, known_graph_ids):
    merged = {pattern_id: {} for pattern_id in range(len(patterns))}
    known = set(map(int, known_graph_ids))
    for partition_name, graph_ids, coverage in partitions:
        allowed = set(map(int, graph_ids))
        for pattern_id, entries in coverage.items():
            if pattern_id not in merged:
                raise ValueError("Coverage references unknown pattern")
            if not set(entries).issubset(allowed):
                raise ValueError(f"{partition_name} coverage references another partition")
            if not set(entries).issubset(known):
                raise ValueError("Coverage references unknown graph")
            merged[pattern_id].update({int(graph_id): 1 for graph_id in entries})
    return merged
