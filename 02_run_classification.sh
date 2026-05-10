#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
RUNS_BASE=${RUNS_BASE:-${ARTIFACT_ROOT:-$SCRIPT_DIR/runs}}
RUN_ROOT=${RUN_ROOT:-$RUNS_BASE/${RUN_ID:-collect_ccmain2026_08}}
INPUT_FILE=${INPUT_FILE:-$RUN_ROOT/01_filter_by_platform/prompt_links.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/02_classify}
OUTPUT_FILE=${OUTPUT_FILE:-$OUTPUT_DIR/classified_prompt_links.jsonl}
LOG_FILE=${LOG_FILE:-$OUTPUT_DIR/classify_prompt_links.log}
PID_FILE=${PID_FILE:-$OUTPUT_DIR/classify_prompt_links.pid}
CLASSIFY_MODEL_DIR=${CLASSIFY_MODEL_DIR:-${MODEL_DIR:-$SCRIPT_DIR/models/tier2_gt_hq10000_plus_recboost_20260407_v2}}
CLASSIFY_DEVICE=${CLASSIFY_DEVICE:-cpu}
CLASSIFY_BATCH_SIZE=${CLASSIFY_BATCH_SIZE:-8}
CLASSIFY_PROGRESS_INTERVAL=${CLASSIFY_PROGRESS_INTERVAL:-10}

if [[ ! -d "$SCRIPT_DIR" ]]; then
  echo "ERROR: script directory does not exist: $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: filter output does not exist: $INPUT_FILE" >&2
  exit 1
fi

if [[ ! -d "$CLASSIFY_MODEL_DIR/model" || ! -f "$CLASSIFY_MODEL_DIR/thresholds.json" ]]; then
  echo "ERROR: active classification model is incomplete: $CLASSIFY_MODEL_DIR" >&2
  echo "Expected: $CLASSIFY_MODEL_DIR/model and $CLASSIFY_MODEL_DIR/thresholds.json" >&2
  exit 1
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util

required = ("numpy", "torch", "transformers")
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
then
  echo "ERROR: missing semantic inference dependencies." >&2
  echo "Install them with: python3 -m pip install -r $SCRIPT_DIR/requirements.txt -r $SCRIPT_DIR/requirements-inference.txt" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -e "$OUTPUT_FILE" || -e "$LOG_FILE" || -e "$PID_FILE" ]]; then
  echo "ERROR: classification outputs already exist under: $OUTPUT_DIR" >&2
  echo "Remove the old output/log/pid files before rerunning this stage." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

nohup python3 -u classify_prompt_links.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --model-dir "$CLASSIFY_MODEL_DIR" \
  --device "$CLASSIFY_DEVICE" \
  --batch-size "$CLASSIFY_BATCH_SIZE" \
  --include-benign \
  --progress-interval "$CLASSIFY_PROGRESS_INTERVAL" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "LOG: $LOG_FILE"
echo "INPUT_FILE: $INPUT_FILE"
echo "MODEL_DIR: $CLASSIFY_MODEL_DIR"
echo "DEVICE: $CLASSIFY_DEVICE"
echo "BATCH_SIZE: $CLASSIFY_BATCH_SIZE"
echo "OUTPUT_FILE: $OUTPUT_FILE"
