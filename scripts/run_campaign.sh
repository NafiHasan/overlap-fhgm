#!/usr/bin/env bash
# Campaigns of the method's settings (pipeline/profiles.py).
#
#   bash scripts/run_campaign.sh init    CAMPAIGN [inductive|transductive] [FLAGS...]
#   bash scripts/run_campaign.sh run     CAMPAIGN DATASET [--repeat R]
#   bash scripts/run_campaign.sh analyze CAMPAIGN
#
# init once per campaign, before any run; extra FLAGS (such as
# --require-clean-source) go to the runner. The run ID is the campaign ID.
# Then run one dataset per terminal. Campaigns that share mining settings
# share caches and must not run on the same dataset at once.
set -euo pipefail
cd "$(dirname "$0")/.."

# Match the recorded execution settings; they are part of the run identity
# and must be identical for init, run and resume.
unset CUDA_VISIBLE_DEVICES OMP_NUM_THREADS TF_ENABLE_ONEDNN_OPTS
unset TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/overlap-fhgm-mpl}"

PY=.venv/bin/python
RUNNER=scripts/run_campaign.py
ROOT=results/campaigns

usage() { sed -n '2,11p' "$0" >&2; exit 2; }

case "${1:-}" in
  init)
    [[ $# -ge 2 ]] || usage
    "$PY" "$RUNNER" --campaign-id "$2" --protocol "${3:-inductive}" \
      --cache-policy build-missing --stage init "${@:4}"
    ;;
  run)
    [[ $# -ge 3 ]] || usage
    if [[ ! -f "$ROOT/$2/campaign.json" ]]; then
      echo "Campaign $2 is not initialized; run: bash $0 init $2" >&2
      exit 1
    fi
    "$PY" "$RUNNER" --campaign-id "$2" --datasets "$3" --stage run "${@:4}"
    ;;
  analyze)
    [[ $# -eq 2 ]] || usage
    "$PY" scripts/analyze_campaign.py --manifest "$ROOT/$2/campaign.json" \
      --output-dir "$ROOT/$2/analysis" >/dev/null
    cat "$ROOT/$2/analysis/analysis.md"
    ;;
  *) usage ;;
esac
