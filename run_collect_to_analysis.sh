#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
RUNS_BASE=${RUNS_BASE:-${ARTIFACT_ROOT:-$SCRIPT_DIR/runs}}
RUN_ROOT=${RUN_ROOT:-}
RUN_ID=${RUN_ID:-}
CRAWL=${CRAWL:-}

START_STAGE=${START_STAGE:-collect}
END_STAGE=${END_STAGE:-analysis}

COLLECT_WORKERS=${COLLECT_WORKERS:-20}
FILTER_WORKERS=${FILTER_WORKERS:-0}
FILTER_PROGRESS_INTERVAL=${FILTER_PROGRESS_INTERVAL:-30}

CLASSIFY_MODEL_DIR=${CLASSIFY_MODEL_DIR:-${MODEL_DIR:-$SCRIPT_DIR/models/tier2_gt_hq10000_plus_recboost_20260407_v2}}
CLASSIFY_DEVICE=${CLASSIFY_DEVICE:-cpu}
CLASSIFY_BATCH_SIZE=${CLASSIFY_BATCH_SIZE:-8}
CLASSIFY_PROGRESS_INTERVAL=${CLASSIFY_PROGRESS_INTERVAL:-10}
CLASSIFY_OUTPUT_FILE=${CLASSIFY_OUTPUT_FILE:-}

SOURCE_TOP_N=${SOURCE_TOP_N:-200}
TRANCO_CACHE=${TRANCO_CACHE:-$SCRIPT_DIR/tranco_top1m.csv}
TRANCO_MODE=${TRANCO_MODE:-fixed}

stage_index() {
  case "$1" in
    collect) echo 0 ;;
    filter) echo 1 ;;
    classify) echo 2 ;;
    analysis) echo 3 ;;
    *)
      echo "ERROR: unknown stage: $1" >&2
      exit 1
      ;;
  esac
}

stage_enabled() {
  local stage="$1"
  local index
  index=$(stage_index "$stage")
  (( index >= START_STAGE_INDEX && index <= END_STAGE_INDEX ))
}

require_absent_path() {
  local path="$1"
  local label="$2"
  if [[ -e "$path" ]]; then
    echo "ERROR: $label already exists: $path" >&2
    echo "Use START_STAGE/END_STAGE to resume from a later stage, or remove the existing output first." >&2
    exit 1
  fi
}

require_directory() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: $label does not exist: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $label does not exist: $path" >&2
    exit 1
  fi
}

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  "$@" 2>&1 | tee "$log_file"
}

if [[ ! -d "$SCRIPT_DIR" ]]; then
  echo "ERROR: script directory does not exist: $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -d "$RUNS_BASE" ]]; then
  echo "ERROR: runs base directory does not exist: $RUNS_BASE" >&2
  exit 1
fi

if [[ -z "$RUN_ROOT" ]]; then
  if [[ -z "$RUN_ID" ]]; then
    echo "ERROR: set RUN_ID or RUN_ROOT before running the wrapper." >&2
    exit 1
  fi
  RUN_ROOT="$RUNS_BASE/$RUN_ID"
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(basename "$RUN_ROOT")"
fi

if [[ -z "$CRAWL" ]]; then
  echo "ERROR: set CRAWL to the Common Crawl snapshot name, for example CC-MAIN-2026-12." >&2
  exit 1
fi

if [[ ! -d "$CLASSIFY_MODEL_DIR/model" || ! -f "$CLASSIFY_MODEL_DIR/thresholds.json" ]]; then
  echo "ERROR: active classification model is incomplete: $CLASSIFY_MODEL_DIR" >&2
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

START_STAGE_INDEX=$(stage_index "$START_STAGE")
END_STAGE_INDEX=$(stage_index "$END_STAGE")
if (( START_STAGE_INDEX > END_STAGE_INDEX )); then
  echo "ERROR: START_STAGE must not come after END_STAGE." >&2
  exit 1
fi

MODEL_TAG="$(basename "$CLASSIFY_MODEL_DIR")"
if [[ -z "$CLASSIFY_OUTPUT_FILE" ]]; then
  CLASSIFY_OUTPUT_FILE="$RUN_ROOT/02_classify/classified_prompt_links.$MODEL_TAG.jsonl"
fi

COLLECT_OUTPUT_DIR="$RUN_ROOT/00_collect"
COLLECT_OUTPUT_FILE="$COLLECT_OUTPUT_DIR/prompt_links.jsonl"
COLLECT_PROGRESS_FILE="$COLLECT_OUTPUT_FILE.progress"
COLLECT_LOG_FILE="$COLLECT_OUTPUT_DIR/collect_candidate_pages_from_wat.log"

FILTER_OUTPUT_DIR="$RUN_ROOT/01_filter_by_platform"
FILTER_OUTPUT_FILE="$FILTER_OUTPUT_DIR/prompt_links.jsonl"
FILTER_LOG_FILE="$FILTER_OUTPUT_DIR/filter_by_platform.log"

CLASSIFY_OUTPUT_DIR="$RUN_ROOT/02_classify"
CLASSIFY_LOG_FILE="$CLASSIFY_OUTPUT_DIR/classify_prompt_links.log"

SOURCE_OUTPUT_DIR="$RUN_ROOT/03_source_url_analysis"
SOURCE_LOG_FILE="$SOURCE_OUTPUT_DIR/analyze_source_risk.log"

TARGET_OUTPUT_DIR="$RUN_ROOT/03b_target_analysis"
TARGET_LOG_FILE="$TARGET_OUTPUT_DIR/analyze_target_risk.log"

mkdir -p "$RUN_ROOT"
cd "$SCRIPT_DIR"

echo "RUN_ID: $RUN_ID"
echo "RUN_ROOT: $RUN_ROOT"
echo "CRAWL: $CRAWL"
echo "START_STAGE: $START_STAGE"
echo "END_STAGE: $END_STAGE"
echo "CLASSIFY_MODEL_DIR: $CLASSIFY_MODEL_DIR"

if stage_enabled collect; then
  require_absent_path "$COLLECT_OUTPUT_FILE" "collect output"
  require_absent_path "$COLLECT_PROGRESS_FILE" "collect progress file"
  require_absent_path "$COLLECT_LOG_FILE" "collect log file"
  mkdir -p "$COLLECT_OUTPUT_DIR"
  run_with_log \
    "$COLLECT_LOG_FILE" \
    python3 -u collect_candidate_pages_from_wat.py \
      --crawl "$CRAWL" \
      --output "$COLLECT_OUTPUT_FILE" \
      --workers "$COLLECT_WORKERS" \
      --progress-file "$COLLECT_PROGRESS_FILE"
elif stage_enabled filter; then
  require_file "$COLLECT_OUTPUT_FILE" "collect output"
fi

if stage_enabled filter; then
  require_absent_path "$FILTER_OUTPUT_FILE" "filter output"
  require_absent_path "$FILTER_LOG_FILE" "filter log file"
  mkdir -p "$FILTER_OUTPUT_DIR"
  run_with_log \
    "$FILTER_LOG_FILE" \
    python3 -u filter_by_platform.py \
      --input "$COLLECT_OUTPUT_FILE" \
      --output "$FILTER_OUTPUT_FILE" \
      --workers "$FILTER_WORKERS" \
      --progress-interval "$FILTER_PROGRESS_INTERVAL"
elif stage_enabled classify; then
  require_file "$FILTER_OUTPUT_FILE" "filter output"
fi

if stage_enabled classify; then
  require_absent_path "$CLASSIFY_OUTPUT_FILE" "classification output"
  require_absent_path "$CLASSIFY_LOG_FILE" "classification log file"
  mkdir -p "$CLASSIFY_OUTPUT_DIR"
  run_with_log \
    "$CLASSIFY_LOG_FILE" \
    python3 -u classify_prompt_links.py \
      --input "$FILTER_OUTPUT_FILE" \
      --output "$CLASSIFY_OUTPUT_FILE" \
      --model-dir "$CLASSIFY_MODEL_DIR" \
      --device "$CLASSIFY_DEVICE" \
      --batch-size "$CLASSIFY_BATCH_SIZE" \
      --include-benign \
      --progress-interval "$CLASSIFY_PROGRESS_INTERVAL"
elif stage_enabled analysis; then
  require_file "$CLASSIFY_OUTPUT_FILE" "classification output"
fi

if stage_enabled analysis; then
  require_absent_path "$SOURCE_LOG_FILE" "source analysis log file"
  require_absent_path "$TARGET_LOG_FILE" "target analysis log file"
  mkdir -p "$SOURCE_OUTPUT_DIR" "$TARGET_OUTPUT_DIR"
  run_with_log \
    "$SOURCE_LOG_FILE" \
    python3 -u analyze_source_risk.py \
      --input "$CLASSIFY_OUTPUT_FILE" \
      --output-dir "$SOURCE_OUTPUT_DIR" \
      --crawl-name "$CRAWL" \
      --top-n "$SOURCE_TOP_N" \
      --tranco-cache "$TRANCO_CACHE" \
      --tranco-mode "$TRANCO_MODE"
  run_with_log \
    "$TARGET_LOG_FILE" \
    python3 -u analyze_target_risk.py \
      --input "$CLASSIFY_OUTPUT_FILE" \
      --output-dir "$TARGET_OUTPUT_DIR" \
      --crawl-name "$CRAWL"
fi

echo "DONE"
echo "COLLECT_OUTPUT_FILE: $COLLECT_OUTPUT_FILE"
echo "FILTER_OUTPUT_FILE: $FILTER_OUTPUT_FILE"
echo "CLASSIFY_OUTPUT_FILE: $CLASSIFY_OUTPUT_FILE"
echo "SOURCE_OUTPUT_DIR: $SOURCE_OUTPUT_DIR"
echo "TARGET_OUTPUT_DIR: $TARGET_OUTPUT_DIR"
