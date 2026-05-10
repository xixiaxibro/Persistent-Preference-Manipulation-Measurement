#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
RUNS_BASE=${RUNS_BASE:-${ARTIFACT_ROOT:-$SCRIPT_DIR/runs}}
RUN_ROOT=${RUN_ROOT:-$RUNS_BASE/${RUN_ID:-collect_ccmain2026_08}}
INPUT_FILE=${INPUT_FILE:-$RUN_ROOT/00_collect/prompt_links.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/01_filter_by_platform}
OUTPUT_FILE=${OUTPUT_FILE:-$OUTPUT_DIR/prompt_links.jsonl}
LOG_FILE=${LOG_FILE:-$OUTPUT_DIR/filter_by_platform.log}
PID_FILE=${PID_FILE:-$OUTPUT_DIR/filter_by_platform.pid}
WORKERS=${WORKERS:-0}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-30}

if [[ ! -d "$SCRIPT_DIR" ]]; then
  echo "ERROR: script directory does not exist: $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: collect output does not exist: $INPUT_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -e "$OUTPUT_FILE" || -e "$LOG_FILE" || -e "$PID_FILE" ]]; then
  echo "ERROR: filter outputs already exist under: $OUTPUT_DIR" >&2
  echo "Remove the old output/log/pid files before rerunning this stage." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

nohup python3 -u filter_by_platform.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --workers "$WORKERS" \
  --progress-interval "$PROGRESS_INTERVAL" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "LOG: $LOG_FILE"
echo "INPUT_FILE: $INPUT_FILE"
echo "OUTPUT_FILE: $OUTPUT_FILE"
