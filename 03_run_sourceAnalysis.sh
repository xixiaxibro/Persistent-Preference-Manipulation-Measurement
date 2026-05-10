#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
RUNS_BASE=${RUNS_BASE:-${ARTIFACT_ROOT:-$SCRIPT_DIR/runs}}
RUN_ROOT=${RUN_ROOT:-$RUNS_BASE/${RUN_ID:-collect_ccmain2026_08}}
INPUT_FILE=${INPUT_FILE:-$RUN_ROOT/02_classify/classified_prompt_links.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/03_source_url_analysis}
LOG_FILE=${LOG_FILE:-$OUTPUT_DIR/analyze_source_urls.log}
PID_FILE=${PID_FILE:-$OUTPUT_DIR/analyze_source_urls.pid}
TRANCO_CACHE=${TRANCO_CACHE:-$SCRIPT_DIR/tranco_top1m.csv}
TRANCO_MODE=${TRANCO_MODE:-download-if-missing}
TOP_N=${TOP_N:-200}
EXAMPLES_PER_GROUP=${EXAMPLES_PER_GROUP:-2}
CRAWL_NAME=${CRAWL_NAME:-CC-MAIN-2026-08}
ONLY_NONEMPTY_PROMPT=${ONLY_NONEMPTY_PROMPT:-0}
SUSPICIOUS_ONLY=${SUSPICIOUS_ONLY:-0}
SHARDS=${SHARDS:-32}

resolve_input_file() {
  local classify_dir="$RUN_ROOT/02_classify"
  local candidates=()
  local path

  for path in "$classify_dir"/classified_prompt_links*.jsonl "$classify_dir"/classified_prompt_links*.jsonl.gz; do
    if [[ -f "$path" ]]; then
      candidates+=("$path")
    fi
  done

  if [[ ${#candidates[@]} -eq 1 ]]; then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  if [[ ${#candidates[@]} -gt 1 ]]; then
    echo "ERROR: multiple classified input files found under: $classify_dir" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 1
  fi

  return 1
}

if [[ ! -d "$SCRIPT_DIR" ]]; then
  echo "ERROR: script directory does not exist: $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  if RESOLVED_INPUT_FILE=$(resolve_input_file); then
    INPUT_FILE="$RESOLVED_INPUT_FILE"
  fi
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: input file does not exist: $INPUT_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -e "$LOG_FILE" || -e "$PID_FILE" ]]; then
  echo "ERROR: source analysis log or pid file already exists under: $OUTPUT_DIR" >&2
  echo "Remove the old files before rerunning this stage." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

CMD=(
  python3 -u run_sharded_source_analysis.py
  --input "$INPUT_FILE"
  --output-dir "$OUTPUT_DIR"
  --top-n "$TOP_N"
  --examples-per-group "$EXAMPLES_PER_GROUP"
  --tranco-cache "$TRANCO_CACHE"
  --tranco-mode "$TRANCO_MODE"
  --shards "$SHARDS"
)

if [[ -n "$CRAWL_NAME" ]]; then
  CMD+=(--crawl-name "$CRAWL_NAME")
fi

if [[ "$ONLY_NONEMPTY_PROMPT" == "1" ]]; then
  CMD+=(--only-nonempty-prompt)
fi

if [[ "$SUSPICIOUS_ONLY" == "1" ]]; then
  CMD+=(--suspicious-only)
fi

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "LOG: $LOG_FILE"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "INPUT_FILE: $INPUT_FILE"
echo "TRANCO_CACHE: $TRANCO_CACHE"
