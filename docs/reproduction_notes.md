# Reproduction Notes

## Demo Reproduction

The demo uses synthetic fixture data:

```bash
python scripts/run_pipeline.py \
  --config configs/runs.example.yaml \
  --run-id demo \
  --crawl DEMO \
  --paths-file examples/fixtures/demo_wat.paths \
  --classifier rule \
  --overwrite
```

This validates stage wiring, schemas, and output directories.

## Paper-Scale Reproduction

Paper-scale reproduction additionally requires:

- Common Crawl WAT access
- external storage for run outputs
- the semantic classifier checkpoint, if reproducing model-based labels
- optional Tranco cache for source-domain ranking

The default `rule` classifier is for public smoke tests, not for reproducing paper result numbers.

## License

This repository uses the MIT License.
