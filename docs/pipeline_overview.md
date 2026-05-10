# Pipeline Overview

The public artifact implements the minimum reproducible measurement path:

1. Collect prompt-link candidates from Common Crawl WAT metadata or a local fixture.
2. Filter candidates to known AI assistant platforms.
3. Enrich and classify prompt text.
4. Aggregate medium/high-risk rows by source domain.
5. Aggregate medium/high-risk rows by target platform.
6. Build a cross-crawl summary from one or more run roots.

Per-run outputs use this shape:

```text
<run-root>/
├── 00_collect/prompt_links.jsonl
├── 01_filter_by_platform/prompt_links.jsonl
├── 02_classify/classified_prompt_links.jsonl
├── 03_source_risk/
├── 03_target_risk/
└── 04_cross_crawl/
```

The demo fixture is synthetic and validates pipeline mechanics, not paper claims.
