#!/usr/bin/env bash
# Grid searches (spec: scripts/grid_search/grids.json).
#
#   bash scripts/grid_search/run_grid.sh plan   GRID
#   bash scripts/grid_search/run_grid.sh init   GRID
#   bash scripts/grid_search/run_grid.sh run    GRID DATASET [extra options]
#   bash scripts/grid_search/run_grid.sh status GRID
#   bash scripts/grid_search/run_grid.sh report GRID
#   bash scripts/grid_search/run_grid.sh launch GRID
#
# launch initializes the grid if needed and starts one detached tmux session
# per dataset (form, genre, steam, twitter, rhg_10, rhg_3); attach with
# `tmux attach -t steam`. Missing mining caches are mined by the first run
# that needs them. Extra run options are passed through, e.g. --mining O_D2,
# --limit 4, --repeat 0.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Match the recorded campaign execution settings. They are part of the
# implementation identity, so the base run's mining caches are reused only
# when these are identical for init, run and resume.
unset CUDA_VISIBLE_DEVICES OMP_NUM_THREADS TF_ENABLE_ONEDNN_OPTS
unset TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/overlap-fhgm-mpl}"

PY=.venv/bin/python
RUNNER=scripts/grid_search/run_grid.py

usage() { sed -n '2,16p' "$0" >&2; exit 2; }

# tmux session name -> dataset
SESSIONS=(form:imdb_dir_form genre:imdb_dir_genre steam:steam_player
          twitter:twitter_friend rhg_10:rhg_10 rhg_3:rhg_3)

launch() {
  local grid="$1" pair session dataset
  command -v tmux >/dev/null || { echo "ERROR: tmux is not installed" >&2; exit 1; }
  for pair in "${SESSIONS[@]}"; do
    if tmux has-session -t "=${pair%%:*}" 2>/dev/null; then
      echo "ERROR: tmux session ${pair%%:*} already exists; close it first" >&2; exit 1
    fi
  done
  if [[ ! -f "results/grid_search/$grid/grid.json" ]]; then
    "$PY" "$RUNNER" init --grid "$grid"
  fi
  for pair in "${SESSIONS[@]}"; do
    session="${pair%%:*}"; dataset="${pair#*:}"
    tmux new-session -d -s "$session" -c "$PWD" \
      "bash scripts/grid_search/run_grid.sh run $grid $dataset; status=\$?; \
       echo; echo \"[$session] finished with exit code \$status\"; exec bash"
    echo "started tmux session $session ($dataset)"
  done
}

case "${1:-}" in
  plan|init|status|report)
    [[ $# -ge 2 ]] || usage
    "$PY" "$RUNNER" "$1" --grid "$2" "${@:3}"
    ;;
  run)
    [[ $# -ge 3 ]] || usage
    "$PY" "$RUNNER" run --grid "$2" --dataset "$3" "${@:4}"
    ;;
  launch)
    [[ $# -eq 2 ]] || usage
    launch "$2"
    ;;
  *) usage ;;
esac
