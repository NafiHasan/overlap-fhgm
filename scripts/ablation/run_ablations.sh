#!/usr/bin/env bash
# Ablation suites (spec: scripts/ablation/suites.json).
#
#   bash scripts/ablation/run_ablations.sh plan   SUITE
#   bash scripts/ablation/run_ablations.sh init   SUITE
#   bash scripts/ablation/run_ablations.sh run    SUITE DATASET [extra options]
#   bash scripts/ablation/run_ablations.sh launch SUITE [DATASET ...]
#   bash scripts/ablation/run_ablations.sh status SUITE
#   bash scripts/ablation/run_ablations.sh report SUITE
#
# Run init once, then one dataset per terminal, or launch: one detached tmux
# session per dataset (default all six; session abl_<SUITE>_<dataset>, log in
# logs/ablation/). Extra run options are passed through, e.g. --tier
# classification, --study FACT DEPTH, --repeat 0.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Match the recorded campaign execution settings. They are part of the
# implementation identity, so the base run's mining caches are reused only
# when these are identical for init, run and resume.
unset CUDA_VISIBLE_DEVICES OMP_NUM_THREADS TF_ENABLE_ONEDNN_OPTS
unset TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/overlap-fhgm-mpl}"

PY=.venv/bin/python
RUNNER=scripts/ablation/run_ablations.py
REPORT=scripts/ablation/report_ablations.py

DATASETS=(imdb_dir_form imdb_dir_genre steam_player twitter_friend rhg_10 rhg_3)

usage() { sed -n '2,14p' "$0" >&2; exit 2; }

case "${1:-}" in
  plan|init|status)
    [[ $# -ge 2 ]] || usage
    "$PY" "$RUNNER" "$1" --suite "$2" "${@:3}"
    ;;
  run)
    [[ $# -ge 3 ]] || usage
    "$PY" "$RUNNER" run --suite "$2" --dataset "$3" "${@:4}"
    ;;
  launch)
    [[ $# -ge 2 ]] || usage
    command -v tmux >/dev/null || { echo "ERROR: tmux is not installed" >&2; exit 1; }
    suite="$2"
    selected=("${@:3}")
    [[ ${#selected[@]} -gt 0 ]] || selected=("${DATASETS[@]}")
    for dataset in "${selected[@]}"; do
      session="abl_${suite}_${dataset}"
      if tmux has-session -t "=$session" 2>/dev/null; then
        echo "ERROR: tmux session $session already exists; close it first" >&2; exit 1
      fi
    done
    mkdir -p logs/ablation
    for dataset in "${selected[@]}"; do
      session="abl_${suite}_${dataset}"
      tmux new-session -d -s "$session" -c "$PWD" \
        "bash $0 run $suite $dataset 2>&1 | tee -a logs/ablation/${suite}_${dataset}.log; \
         echo; echo \"[$session] finished; see logs/ablation/${suite}_${dataset}.log\"; exec bash"
      echo "started $session"
    done
    ;;
  report)
    [[ $# -ge 2 ]] || usage
    "$PY" "$REPORT" --suite "$2" "${@:3}"
    ;;
  *) usage ;;
esac
