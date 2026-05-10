# Artifact Policy

This repository contains only the minimal runnable measurement pipeline, documentation, configuration examples, and tiny synthetic fixtures.

Do not commit:

- Common Crawl-derived JSONL corpora
- full filtered or classified prompt-link outputs
- full source/target risk tables
- paper-table exports and internal downstream reports
- model checkpoints, tokenizer files, and generated model packages
- replay traces, screenshots, browser profiles, or platform session data
- `.env`, API keys, tokens, credentials, logs, caches, and private paths

Internal analyses such as template reuse, language distribution, source distribution, reverse template coverage, and paper-table export should be documented in the paper or supplementary notes, but they are intentionally not part of this minimal public artifact.
