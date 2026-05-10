# Reproduction Notes

## Requirements

Full reproduction requires:

- access to Common Crawl WAT path listings
- external storage for per-crawl JSONL outputs
- a compatible semantic classifier checkpoint
- a Tranco cache or network access for downloading one
- enough CPU/GPU capacity for classifier inference

## Configuration

Recommended environment variables:

```bash
export RUNS_BASE=/path/to/external/runs
export MODEL_DIR=/path/to/classifier_checkpoint
export REPORTS_DIR=/path/to/external/reports
export TRANCO_CACHE=/path/to/tranco_top1m.csv
```

The checked-in examples are only for smoke tests and schema orientation.

## License

This repository uses the MIT License. The license permits reuse, modification, redistribution, and commercial use with attribution and warranty disclaimer.
