#!/usr/bin/env bash
# Hypergraph WL kernel baseline (scripts/baselines/run_hypergraph_wl.py).
#
#   bash scripts/baselines/run_hypergraph_wl.sh run    DATASET [extra options]
#   bash scripts/baselines/run_hypergraph_wl.sh launch [DATASET ...]
#   bash scripts/baselines/run_hypergraph_wl.sh status
#
# launch starts one detached tmux session per dataset (default all six;
# session hgwl_<dataset>, log in logs/baselines/). Each run writes every
# kernel, label and held-out-norm variant; existing records are skipped.
# Runs need committed, clean source.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Six datasets run side by side; keep BLAS from oversubscribing the cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

PY=.venv/bin/python
RUNNER=scripts/baselines/run_hypergraph_wl.py

DATASETS=(imdb_dir_form imdb_dir_genre steam_player twitter_friend rhg_10 rhg_3)

usage() { sed -n '2,11p' "$0" >&2; exit 2; }

case "${1:-}" in
  run)
    [[ $# -ge 2 ]] || usage
    "$PY" "$RUNNER" "$2" "${@:3}"
    ;;
  launch)
    command -v tmux >/dev/null || { echo "ERROR: tmux is not installed" >&2; exit 1; }
    selected=("${@:2}")
    [[ ${#selected[@]} -gt 0 ]] || selected=("${DATASETS[@]}")
    for dataset in "${selected[@]}"; do
      session="hgwl_${dataset}"
      if tmux has-session -t "=$session" 2>/dev/null; then
        echo "ERROR: tmux session $session already exists; close it first" >&2; exit 1
      fi
    done
    mkdir -p logs/baselines
    for dataset in "${selected[@]}"; do
      session="hgwl_${dataset}"
      tmux new-session -d -s "$session" -c "$PWD" \
        "bash $0 run $dataset 2>&1 | tee -a logs/baselines/hgwl_${dataset}.log; \
         echo; echo \"[$session] finished; see logs/baselines/hgwl_${dataset}.log\"; exec bash"
      echo "started $session"
    done
    ;;
  status)
    "$PY" - <<'EOF'
import json
from pathlib import Path

for path in sorted(Path("results").glob("*/composite/baselines/HGWL_*/experiment.json")):
    record = json.loads(path.read_text())
    a = record["aggregate"]
    print(f"{record['dataset']:15s} {record['run_id']:30s} test F1 {a['macro_f1_mean']:6.2f} "
          f"± {a['macro_f1_std']:4.2f}  acc {a['accuracy_mean']:6.2f} ± {a['accuracy_std']:4.2f}  "
          f"(val F1 {a['val_macro_f1_mean']:6.2f})")
EOF
    ;;
  *) usage ;;
esac
