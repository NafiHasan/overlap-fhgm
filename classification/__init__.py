"""Pattern counts, pattern selection, the classifiers and their evaluation."""

from .feature_selection import entropy, gain, filter_codes


def __getattr__(name):
    """Keep pure utilities importable without initializing TensorFlow."""
    from importlib import import_module

    modules = {"NN": ".classifiers"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "entropy",
    "gain",
    "filter_codes",
    "NN",
]
