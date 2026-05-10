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

## License Choices

No license has been selected yet.

Common options:

- MIT: simple permissive license; allows reuse, modification, redistribution, and commercial use with attribution and warranty disclaimer.
- Apache-2.0: permissive license like MIT, with an explicit patent grant and more detailed compliance terms.
- BSD-3-Clause: permissive license similar to MIT, with an added non-endorsement clause.
- GPL-3.0: copyleft license; derivative works distributed to others generally need to remain GPL-compatible.
- No license: default copyright applies; others have no clear permission to reuse or redistribute.

For an academic reproducibility repository, MIT or Apache-2.0 are usually the lowest-friction choices. Apache-2.0 is preferable if patent language matters; MIT is shorter and simpler.
