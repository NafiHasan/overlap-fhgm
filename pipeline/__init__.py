"""Fully inductive and transductive-vocabulary runs of the method."""

__all__ = [
    "SplitIndices",
    "load_and_validate_split_manifest",
    "run_fully_inductive_pipeline",
    "run_transductive_pipeline",
]


def __getattr__(name):
    # Hashing, input validation, and artifact inspection must not initialize
    # TensorFlow merely by importing pipeline.
    if name in {"SplitIndices", "load_and_validate_split_manifest"}:
        from . import splits
        return getattr(splits, name)
    if name in __all__:
        from . import inductive
        return getattr(inductive, name)
    raise AttributeError(name)
