"""
Supervised pattern selection, used only in the selection ablation.

Patterns are ranked on the training labels by the entropy-based score of Alam
et al. (2021) and kept in that order until every training hypergraph contains
``min_cov`` kept patterns.
"""

import math
import logging

logger = logging.getLogger(__name__)


def entropy(bucket):
    """
    Calculate Shannon entropy of a label distribution.

    Args:
        bucket (dict): Mapping from label to count.

    Returns:
        float: Entropy value in bits.
    """
    total = sum(bucket.values())
    if total == 0:
        return 0.0
    res = 0.0
    for count in bucket.values():
        if count > 0:
            p = count / total
            res -= p * math.log2(p)
    return res


def gain(subhypergraph, labels):
    """
    Calculate H(Y|X) / H(X), the entropy-based score of Alam et al. (2021).

    X is the pattern's presence and Y the class. Lower values rank first.

    Args:
        subhypergraph (list): List of hypergraph indices where the pattern appears.
        labels: Array of class labels for all hypergraphs in the training set.

    Returns:
        float: Conditional-entropy ratio (lower is better).
    """
    n = len(labels)
    k = len(subhypergraph)

    if k == 0 or k == n:
        return float("inf")

    subhg_set = set(subhypergraph)

    positive_bucket = dict()
    negative_bucket = dict()
    positive, negative = 0, 0
    for i in range(n):
        if i in subhg_set:
            positive += 1
            if labels[i] in positive_bucket:
                positive_bucket[labels[i]] += 1
            else:
                positive_bucket[labels[i]] = 1
        else:
            negative += 1
            if labels[i] in negative_bucket:
                negative_bucket[labels[i]] += 1
            else:
                negative_bucket[labels[i]] = 1

    split = -(k / n) * math.log2(k / n) - ((n - k) / n) * math.log2((n - k) / n)

    return (
        (positive / (positive + negative)) * entropy(positive_bucket)
        + (negative / (positive + negative)) * entropy(negative_bucket)
    ) / split


class SelectedCodes(dict):
    """Coverage columns with their original frozen-vocabulary IDs retained."""

    def __init__(self):
        super().__init__()
        self.pattern_ids = []


def filter_codes(codes, labels, train_index, total_labels_count, min_cov=15):
    """
    Select patterns by their score and coverage of the training hypergraphs.

    Patterns are visited in increasing order of :func:`gain` on the training
    labels. A pattern is kept when it occurs in a training hypergraph that
    contains fewer than ``min_cov`` kept patterns so far.

    Args:
        codes (dict): Mapping from pattern ID to {hypergraph_id: presence_value}.
        labels: Array of class labels (for the training subset).
        train_index: Array of training set indices.
        total_labels_count (int): Total number of hypergraphs.
        min_cov (int): Number of kept patterns each training hypergraph should contain.

    Returns:
        SelectedCodes: Mapping from new sequential IDs (0..M-1) to
            {hypergraph_id: presence_value}, with the original IDs in ``pattern_ids``.
    """
    if type(min_cov) is not int or min_cov <= 0:
        raise ValueError("min_cov must be a positive integer")
    train_index_set = set(train_index)
    index = {train_index[i]: i for i in range(len(train_index))}

    code_list = []
    for c in codes:
        subhypergraph = [index[item] for item in codes[c] if item in train_index_set]
        code_list.append((c, gain(subhypergraph, labels)))

    code_list.sort(key=lambda tup: tup[1])

    selected_pattern_ids = set()
    coverage = [0] * total_labels_count
    for c, _ in code_list:
        picked = False
        for g in codes[c]:
            if g not in train_index_set:
                continue
            if coverage[g] < min_cov:
                coverage[g] += 1
                picked = True
        if picked:
            selected_pattern_ids.add(c)

    # Preserve sorted rank order for sequential indexing
    ret = SelectedCodes()
    r = 0
    for c, _ in code_list:
        if c in selected_pattern_ids:
            ret[r] = codes[c]
            ret.pattern_ids.append(c)
            r += 1

    logger.info(f"Pattern selection (min_cov={min_cov}): "
                f"{len(codes)} patterns in -> {len(ret)} patterns out")
    return ret
