"""Portable, validated JSON serialization for frozen pattern vocabularies."""

import json
from .candidate import CandidateCode


SCHEMA_VERSION = 2
# Schema 3: overlap-labelled vocabularies (5-tuple extensions whose last entry
# is a list of [earlier edge, overlap label]); ``overlap_binning`` is stored
# at document level. Vocabularies without overlap labels keep schema 2.
OVERLAP_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3}


def _as_json_extension(extension):
    if len(extension) > 4:
        return [*extension[:4], [list(pair) for pair in extension[4]]]
    return list(extension)


def _from_json_extension(extension, overlap_binning):
    if overlap_binning is None:
        if not isinstance(extension, list) or len(extension) != 4:
            raise ValueError("Invalid pattern extension")
        return tuple(extension)
    if (not isinstance(extension, list) or len(extension) != 5 or
            not isinstance(extension[4], list) or
            not all(isinstance(pair, list) and len(pair) == 2 for pair in extension[4])):
        raise ValueError("Invalid pattern extension")
    return (*extension[:4], tuple(tuple(pair) for pair in extension[4]))


def _signature(code):
    return [_as_json_extension(extension) for extension in code.get_extensions()]


def save_pattern_vocabulary(path, patterns, metadata):
    ids = [pattern_id for pattern_id, _ in enumerate(patterns)]
    if ids != list(range(len(patterns))):
        raise ValueError("Pattern IDs must be dense")
    modes = {code.overlap_binning for code in patterns}
    if len(modes) > 1:
        raise ValueError("Patterns mix overlap modes")
    overlap_binning = modes.pop() if modes else None
    schema = SCHEMA_VERSION if overlap_binning is None else OVERLAP_SCHEMA_VERSION
    document = {
        "schema_version": schema,
        "metadata": {**metadata, "schema_version": schema, "pattern_count": len(patterns)},
        "patterns": [{"id": index, "extensions": _signature(code)}
                     for index, code in enumerate(patterns)],
    }
    if overlap_binning is not None:
        document["overlap_binning"] = overlap_binning
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def load_pattern_vocabulary(path):
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    schema = document.get("schema_version")
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unsupported pattern vocabulary schema")
    overlap_binning = document.get("overlap_binning")
    if (schema == OVERLAP_SCHEMA_VERSION) != (overlap_binning is not None):
        raise ValueError("Overlap binning must be stored exactly for schema 3")
    records = document.get("patterns")
    if not isinstance(records, list) or [record.get("id") for record in records] != list(range(len(records))):
        raise ValueError("Pattern vocabulary IDs must be dense and unique")
    patterns = []
    for record in records:
        extensions = record.get("extensions")
        if not isinstance(extensions, list):
            raise ValueError("Pattern extensions must be a list")
        code = CandidateCode(overlap_binning)
        for extension in extensions:
            result = code.extend(_from_json_extension(extension, overlap_binning))
            if result is False:
                raise ValueError("Pattern reconstruction failed")
        if _signature(code) != extensions:
            raise ValueError("Pattern reconstruction signature mismatch")
        patterns.append(code)
    metadata = document.get("metadata", {})
    if metadata.get("pattern_count") != len(patterns):
        raise ValueError("Pattern vocabulary count mismatch")
    return patterns, metadata
