#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
RUNS_BASE=${RUNS_BASE:-${ARTIFACT_ROOT:-$SCRIPT_DIR/runs}}
RUN_ID=${RUN_ID:-collect_ccmain2026_12}
RUN_ROOT=${RUN_ROOT:-$RUNS_BASE/$RUN_ID}
CRAWL=${CRAWL:-CC-MAIN-2026-12}
OUTPUT_DIR="$RUN_ROOT/00_collect"
OUTPUT_FILE="$OUTPUT_DIR/prompt_links.jsonl"
PROGRESS_FILE="$OUTPUT_FILE.progress"
LOG_FILE="$OUTPUT_DIR/collect_candidate_pages_from_wat.log"
PID_FILE="$OUTPUT_DIR/collect_candidate_pages_from_wat.pid"
WORKERS=20

if [[ ! -d "$SCRIPT_DIR" ]]; then
  echo "ERROR: script directory does not exist: $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -d "$RUNS_BASE" ]]; then
  echo "ERROR: runs base directory does not exist: $RUNS_BASE" >&2
  exit 1
fi

if [[ -e "$OUTPUT_FILE" || -e "$PROGRESS_FILE" || -e "$LOG_FILE" || -e "$PID_FILE" ]]; then
  echo "ERROR: target run already has output/log/progress files under: $OUTPUT_DIR" >&2
  echo "Choose a new RUN_ID or remove the old files first." >&2
  exit 1
fi

if [[ ! -d "$RUN_ROOT" ]]; then
  mkdir -p "$RUN_ROOT"
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

cd "$SCRIPT_DIR"

nohup python3 -u collect_candidate_pages_from_wat.py \
  --crawl "$CRAWL" \
  --output "$OUTPUT_FILE" \
  --workers "$WORKERS" \
  --progress-file "$PROGRESS_FILE" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "LOG: $LOG_FILE"
echo "PROGRESS: $PROGRESS_FILE"
