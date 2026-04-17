#!/bin/sh
# Run `hypoevolve run` sequentially many times (default 500), each run after the previous exits.
# First invocation detaches with nohup and writes one log file for the whole batch.
# POSIX sh — safe to run as: sh scripts/run_hypoevolve.sh
#
# Usage:
#   ./scripts/run_hypoevolve.sh [args for hypoevolve run]
#   RUNS=100 ./scripts/run_hypoevolve.sh --config hypoevolve.yaml
# Run from the directory where config-relative paths (e.g. dataset.yaml) resolve.

set -eu

RUNS="${RUNS:-500}"

if [ -z "${HYPOEVOLVE_BATCH_INNER:-}" ]; then
  LOG_FILE="hypoevolve_batch_$(date +%Y%m%d_%H%M%S).log"
  echo "Starting ${RUNS} sequential hypoevolve runs in background (nohup)..."
  echo "Log file: ${LOG_FILE}"
  echo "Monitor:  tail -f ${LOG_FILE}"
  HYPOEVOLVE_BATCH_INNER=1 nohup "$0" "$@" >"${LOG_FILE}" 2>&1 &
  echo "PID:      $!"
  exit 0
fi

i=1
while [ "$i" -le "$RUNS" ]; do
  echo ""
  echo "========== hypoevolve run ${i}/${RUNS} ($(date -Is)) =========="
  hypoevolve run "$@"
  i=$((i + 1))
done

echo ""
echo "========== batch finished: ${RUNS} runs completed ($(date -Is)) =========="
