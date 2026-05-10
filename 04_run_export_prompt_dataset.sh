#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
RUNS_BASE=${RUNS_BASE:-${ARTIFACT_ROOT:-$SCRIPT_DIR/runs}}
RUN_ROOT=${RUN_ROOT:-$RUNS_BASE/${RUN_ID:-collect_ccmain2026_08}}
INPUT_FILE=${INPUT_FILE:-$RUN_ROOT/01_filter_by_platform/prompt_links.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/04_export_prompt_dataset}
OUTPUT_FILE=${OUTPUT_FILE:-$OUTPUT_DIR/prompt_dataset.jsonl}
MANIFEST_FILE=${MANIFEST_FILE:-$OUTPUT_DIR/export_manifest.json}
LOG_FILE=${LOG_FILE:-$OUTPUT_DIR/export_prompt_dataset.log}
PID_FILE=${PID_FILE:-$OUTPUT_DIR/export_prompt_dataset.pid}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-20000}
SEED=${SEED:-42}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-10}

if [[ ! -d "$SCRIPT_DIR" ]]; then
  echo "ERROR: script directory does not exist: $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: Stage 01 filtered input does not exist: $INPUT_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -e "$OUTPUT_FILE" || -e "$MANIFEST_FILE" || -e "$LOG_FILE" || -e "$PID_FILE" ]]; then
  echo "ERROR: export outputs already exist under: $OUTPUT_DIR" >&2
  echo "Remove the old output/manifest/log/pid files before rerunning this stage." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

nohup python3 -u export_prompt_dataset.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --manifest "$MANIFEST_FILE" \
  --total-samples "$TOTAL_SAMPLES" \
  --seed "$SEED" \
  --progress-interval "$PROGRESS_INTERVAL" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "LOG: $LOG_FILE"
echo "INPUT_FILE: $INPUT_FILE"
echo "OUTPUT_FILE: $OUTPUT_FILE"
echo "MANIFEST_FILE: $MANIFEST_FILE"
