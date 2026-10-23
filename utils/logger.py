"""
Dual-output logging: console (concise) + file (detailed).

Console shows INFO-level messages (progress, results, summaries).
Log file captures project DEBUG-level messages (sampled DFS and other details),
while suppressing third-party DEBUG/INFO chatter. Mining progress messages are
INFO so long-running progress remains visible in both outputs.

Usage:
    from utils.logger import setup_logging
    import logging
    logger = logging.getLogger(__name__)

    setup_logging("campaign", dataset="rhg_3", run_id="MAIN")
    # creates logs/rhg_3/campaign/rhg_3_campaign_MAIN_<timestamp>_<pid>.log

    logger.info("Phase 1 complete")      # -> console + file
    logger.debug("Extension call")       # -> file only
"""

import logging
import os
import re
import sys
from datetime import datetime


# TensorFlow can emit native Abseil/CUDA initialization messages directly
# to stderr before Python logging exists. Set native minimum levels as soon as
# an entrypoint imports this module; Python warnings/errors remain unaffected.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")


# Datetime format used across all log output
LOG_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

# These libraries can emit thousands of low-value DEBUG records when the root
# logger is configured at DEBUG. Their warnings and errors remain visible.
_NOISY_THIRD_PARTY_LOGGERS = (
    "absl",
    "h5py",
    "matplotlib",
    "PIL",
    "tensorflow",
)


def _is_noisy_third_party_record(record):
    """Return whether a below-WARNING record belongs to a noisy dependency."""
    if record.levelno >= logging.WARNING:
        return False
    return any(
        record.name == namespace or record.name.startswith(namespace + ".")
        for namespace in _NOISY_THIRD_PARTY_LOGGERS
    )


class _ThirdPartyNoiseFilter(logging.Filter):
    """Keep dependency warnings/errors but discard their routine chatter."""

    def filter(self, record):
        return not _is_noisy_third_party_record(record)


def _quiet_noisy_dependencies():
    """Avoid creating routine dependency records in the first place."""
    for namespace in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(namespace).setLevel(logging.WARNING)


class _ConsoleFormatter(logging.Formatter):
    """Concise format for console: [datetime] message"""

    def format(self, record):
        dt = datetime.fromtimestamp(record.created).strftime(LOG_DATETIME_FMT)
        return f"[{dt}] {record.getMessage()}"


class _FileFormatter(logging.Formatter):
    """Detailed format for log file: [datetime] [LEVEL] name -- message"""

    def format(self, record):
        dt = datetime.fromtimestamp(record.created).strftime(LOG_DATETIME_FMT)
        return f"[{dt}] [{record.levelname}] {record.name} -- {record.getMessage()}"


def _safe_path_component(value):
    """Return a stable filename-safe representation of a logging scope."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def setup_logging(stage_name, *, dataset=None, run_id=None):
    """
    Set up dual-output logging for a pipeline stage.

    Creates a timestamped log file and configures the root logger
    with two handlers:
      - Console (stdout): INFO and above, concise format
      - File: project DEBUG and above, detailed format

    Third-party namespaces known to emit verbose initialization and font
    discovery traces are limited to WARNING and above on both handlers.

    Args:
        stage_name (str): Name of the stage, e.g. "campaign" or "ablation".
        dataset (str | None): Dataset or other logging scope; "multi_dataset"
            when omitted.
        run_id (str | None): Optional run identifier included in the filename.

    Returns:
        str: Path to the created log file.
    """
    dataset_scope = _safe_path_component(dataset or "multi_dataset")
    stage_scope = _safe_path_component(stage_name)
    run_scope = _safe_path_component(run_id) if run_id is not None else None
    log_dir = os.path.join("logs", dataset_scope, stage_scope)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename_parts = [dataset_scope, stage_scope]
    if run_scope is not None:
        filename_parts.append(run_scope)
    filename_parts.extend((timestamp, str(os.getpid())))
    log_filename = os.path.join(log_dir, "_".join(filename_parts) + ".log")

    # Get root logger and clear any existing handlers
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Console handler -- INFO level, concise format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ConsoleFormatter())
    console_handler.addFilter(_ThirdPartyNoiseFilter())
    root.addHandler(console_handler)

    # File handler -- DEBUG level, detailed format
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FileFormatter())
    file_handler.addFilter(_ThirdPartyNoiseFilter())
    root.addHandler(file_handler)

    # Configure these before TensorFlow/Matplotlib are imported by pipeline
    # modules. The handler filters also cover child loggers that later set
    # their own DEBUG level.
    _quiet_noisy_dependencies()

    logging.info(f"Logging to: {log_filename}")
    return log_filename
