# Artifact Policy

The GitHub repository should contain only:

- source code
- documentation
- configuration examples
- tiny synthetic examples
- paper-facing notes that do not include large raw outputs

The following must remain outside git:

- Common Crawl-derived JSONL files
- full filtered prompt-link corpora
- full classified corpora
- full source/target risk tables
- full template, language, source-distribution, and coverage outputs
- replay result dumps
- logs, caches, temporary files, and local benchmark outputs
- model weights, tokenizer files, checkpoints, and generated model packages
- `.env`, API keys, tokens, credentials, and private absolute paths

Use external artifact storage and point scripts to it with `RUNS_BASE` or `ARTIFACT_ROOT`.

Small examples in `examples/` are synthetic and are not paper evidence.
