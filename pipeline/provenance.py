"""Source identities independent of generated results and documentation edits."""

import importlib.metadata
from pathlib import Path
import platform
import os
import subprocess

from .artifacts import canonical_json_hash, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("numpy", "tensorflow", "scikit-learn")


def implementation_identity(root=REPO_ROOT):
    root = Path(root)
    mining_names = {
        str(path.relative_to(root))
        for directory in ("mining", "hypergraph")
        for path in (root / directory).glob("*.py")
    }
    mining_names.update({
        "pipeline/artifacts.py", "pipeline/provenance.py",
        "pipeline/preprocessing.py", "pipeline/splits.py",
    })
    run_names = mining_names | {
        str(path.relative_to(root))
        for directory in ("classification", "pipeline", "utils", "scripts")
        for path in (root / directory).glob("*.py")
    }
    run_names.add("config.py")
    locks = {name: sha256_file(root / name)
             for name in ("uv.lock", "pyproject.toml", ".python-version")}
    files = {name: sha256_file(root / name) for name in sorted(run_names)}
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "machine": platform.machine(),
               "execution_settings": {name: os.environ.get(name) for name in (
                   "CUDA_VISIBLE_DEVICES", "TF_ENABLE_ONEDNN_OPTS", "OMP_NUM_THREADS",
                   "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
               )}}
    for name in DEPENDENCIES:
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = None
    common = {"locks": locks, "runtime": runtime}
    return {
        "schema_version": 1,
        "files": files,
        **common,
        "mining": canonical_json_hash({
            **common, "files": {name: files[name] for name in sorted(mining_names)},
        }),
        "run": canonical_json_hash({**common, "files": files}),
    }


def capture_provenance():
    """Call before writing artifacts; never infer source dirtiness from outputs."""
    implementation = implementation_identity()
    try:
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args], check=True,
                capture_output=True, text=True,
            ).stdout.strip()

        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        source_dirty = bool(git(
            "status", "--porcelain", "--untracked-files=all", "--",
            "mining", "hypergraph", "classification", "pipeline", "utils", "scripts",
            "config.py", ":(top,glob)run_*.py", *implementation["locks"],
        ))
    except (OSError, subprocess.CalledProcessError):
        commit = dirty = source_dirty = None
    return {
        "schema_version": 1,
        "git_commit": commit,
        "git_dirty": dirty,
        "git_source_dirty": source_dirty,
        "implementation": implementation,
    }


def require_unchanged_source(provenance):
    if implementation_identity()["run"] != provenance["implementation"]["run"]:
        raise RuntimeError("Implementation or dependencies changed during this invocation")
