"""
Stage 2: Filter prompt-link JSONL by AI platform and report distribution.

Reads the Stage 1 JSONL output (one JSON object per line), keeps only rows
whose ``target_url`` matches a known AI-platform signature, writes them
verbatim to the output file, and prints a platform-distribution summary.

Design for 250 GB inputs on a 112-thread / 128 GiB machine
------------------------------------------------------------
* **mmap-based byte sharding** – the uncompressed input file is memory-mapped
  and divided into N equal byte ranges.  Each worker process owns one shard,
  seeks to the first complete line after its start offset, and streams
  forward until the shard boundary.  No line is processed twice; no line is
  skipped.  This allows all 112 threads to read the same file in parallel
  with zero inter-process coordination and zero copying.
* **Three-tier per-line filtering** – (1) raw-byte domain scan, (2)
  byte-level ``target_url`` extraction without full JSON parse, (3) full
  ``match_platform_with_exclusion`` only for survivors.  Tier 1 rejects
  the ~90 %+ of lines that target non-AI domains with no parsing at all.
* **Per-shard output files** – each worker writes its own output shard,
  eliminating all write contention.  A final merge pass concatenates the
  shards into one file (or they can be consumed directly).
* **Zero mutation** – matched lines are written as the original raw bytes,
  so there is no JSON round-trip cost and the output is bit-identical to
  the corresponding input lines.
* **Gzip fallback** – if the input is ``.gz`` compressed, mmap/sharding
  is not possible, so a single-process streaming path is used instead.
"""
from __future__ import annotations

import argparse
import gzip
import json
import mmap
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import BinaryIO

from env_config import load_project_env
from platform_signatures import (
    PLATFORM_SIGNATURES,
    PlatformSignature,
    match_platform_with_exclusion,
)

load_project_env()

# ---------------------------------------------------------------------------
# Pre-compiled domain lookup for the fast-path filter
# ---------------------------------------------------------------------------

def _build_domain_index() -> dict[str, list[PlatformSignature]]:
    """
    Build a mapping from every known host suffix to its signature(s).

    Used by the fast-path to decide whether a line *might* contain a
    matching URL before paying for ``urlparse`` + full signature evaluation.
    """
    index: dict[str, list[PlatformSignature]] = {}
    for sig in PLATFORM_SIGNATURES:
        for suffix in sig.host_suffixes:
            index.setdefault(suffix, []).append(sig)
    return index


_DOMAIN_INDEX: dict[str, list[PlatformSignature]] = _build_domain_index()

# Encoded domain fragments for raw-byte scanning.
_DOMAIN_BYTES: tuple[bytes, ...] = tuple(
    suffix.encode("ascii") for suffix in _DOMAIN_INDEX
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_IO_BUFFER_SIZE = 8 * 1024 * 1024  # 8 MB read/write buffers


def _open_input_stream(path: Path) -> BinaryIO:
    """Open a gzip-compressed file for binary streaming (non-mmap path)."""
    return gzip.open(path, "rb")


# ---------------------------------------------------------------------------
# Fast-path: extract target_url without full JSON parse
# ---------------------------------------------------------------------------

_TARGET_URL_KEY = b'"target_url"'


def _fast_extract_target_url(line: bytes) -> str | None:
    """
    Try to extract the ``target_url`` string value directly from raw bytes.

    Returns the URL string on success, or ``None`` if the fast-path cannot
    safely extract it (caller should fall back to ``json.loads``).
    """
    pos = line.find(_TARGET_URL_KEY)
    if pos == -1:
        return None

    pos += len(_TARGET_URL_KEY)
    colon = line.find(b":", pos)
    if colon == -1 or colon > pos + 3:
        return None
    pos = colon + 1

    # Skip whitespace.
    while pos < len(line) and line[pos:pos + 1] in (b" ", b"\t"):
        pos += 1

    if pos >= len(line) or line[pos:pos + 1] != b'"':
        return None
    pos += 1  # skip opening quote

    end = pos
    has_escape = False
    while end < len(line):
        byte = line[end:end + 1]
        if byte == b"\\":
            has_escape = True
            end += 2
            continue
        if byte == b'"':
            break
        end += 1
    else:
        return None

    raw_value = line[pos:end]

    if has_escape:
        try:
            return json.loads(b'"' + raw_value + b'"')
        except (json.JSONDecodeError, ValueError):
            return None

    try:
        return raw_value.decode("utf-8")
    except UnicodeDecodeError:
        return None


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _line_might_match(line: bytes) -> bool:
    """
    Ultra-cheap byte scan: does the line contain any known domain suffix?
    """
    for domain_bytes in _DOMAIN_BYTES:
        if domain_bytes in line:
            return True
    return False


def _match_url(target_url: str) -> str | None:
    """
    Run full platform matching.  Returns the platform name or ``None``.
    """
    sig, excluded = match_platform_with_exclusion(target_url)
    if excluded or sig is None:
        return None
    return sig.name


def _extract_and_match(line: bytes) -> str | None:
    """
    Given a raw JSONL line, return the matched platform name or ``None``.

    Applies the full three-tier filter chain.
    """
    if not _line_might_match(line):
        return None

    target_url = _fast_extract_target_url(line)
    if target_url is None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            row = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
        target_url = row.get("target_url", "")
        if not isinstance(target_url, str) or not target_url:
            return None

    return _match_url(target_url)


# ---------------------------------------------------------------------------
# mmap-based shard processing (one worker)
# ---------------------------------------------------------------------------

def _process_shard(
    input_path: str,
    shard_start: int,
    shard_end: int,
    output_path: str,
    shard_id: int,
) -> dict[str, object]:
    """
    Process one byte-range shard of the input file.

    The worker memory-maps the file, seeks to *shard_start*, aligns to
    the next complete line (unless shard_start == 0), and processes every
    line whose start offset is < *shard_end*.
    """
    platform_counts: Counter[str] = Counter()
    lines_seen = 0
    lines_kept = 0
    bytes_scanned = 0

    fd = os.open(input_path, os.O_RDONLY)
    try:
        file_size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    except Exception:
        os.close(fd)
        raise

    try:
        pos = shard_start

        # Align to the first complete line.
        if pos != 0:
            # Skip the partial line that straddles the boundary.
            nl = mm.find(b"\n", pos)
            if nl == -1:
                # No newline found — this shard has no complete lines.
                return {
                    "shard_id": shard_id,
                    "lines_seen": 0,
                    "lines_kept": 0,
                    "bytes_scanned": 0,
                    "platform_counts": {},
                    "output_path": output_path,
                }
            pos = nl + 1

        with open(output_path, "wb", buffering=_IO_BUFFER_SIZE) as fout:
            while pos < shard_end:
                nl = mm.find(b"\n", pos)
                if nl == -1:
                    # Last line without trailing newline.
                    line = mm[pos:]
                    next_pos = file_size
                else:
                    line = mm[pos:nl + 1]
                    next_pos = nl + 1

                bytes_scanned += next_pos - pos

                stripped = line.strip()
                if stripped:
                    lines_seen += 1
                    platform = _extract_and_match(stripped)
                    if platform is not None:
                        lines_kept += 1
                        platform_counts[platform] += 1
                        # Write the original bytes (with newline).
                        if line.endswith(b"\n"):
                            fout.write(line)
                        else:
                            fout.write(line + b"\n")

                pos = next_pos

    finally:
        mm.close()
        os.close(fd)

    return {
        "shard_id": shard_id,
        "lines_seen": lines_seen,
        "lines_kept": lines_kept,
        "bytes_scanned": bytes_scanned,
        "platform_counts": dict(platform_counts),
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Streaming fallback for gzip inputs
# ---------------------------------------------------------------------------

def _process_gzip_stream(
    input_path: Path,
    output_path: Path,
    progress_interval: float,
) -> dict[str, object]:
    """Single-process streaming path for .gz inputs (cannot be sharded)."""
    platform_counts: Counter[str] = Counter()
    lines_seen = 0
    lines_kept = 0
    bytes_read = 0
    start_time = time.monotonic()
    last_report = start_time

    with gzip.open(input_path, "rb") as fin, \
         open(output_path, "wb", buffering=_IO_BUFFER_SIZE) as fout:
        for raw_line in fin:
            lines_seen += 1
            bytes_read += len(raw_line)

            stripped = raw_line.strip()
            if not stripped:
                continue

            platform = _extract_and_match(stripped)
            if platform is not None:
                lines_kept += 1
                platform_counts[platform] += 1
                if raw_line.endswith(b"\n"):
                    fout.write(raw_line)
                else:
                    fout.write(raw_line + b"\n")

            if lines_seen % 500_000 == 0:
                now = time.monotonic()
                if now - last_report >= progress_interval:
                    elapsed = now - start_time
                    speed = bytes_read / elapsed if elapsed > 0 else 0
                    _print_progress(
                        f"[{elapsed:>6.0f}s] lines: {lines_seen:>12,}  "
                        f"kept: {lines_kept:>10,}  "
                        f"read: {_format_size(bytes_read)}  "
                        f"speed: {_format_speed(speed)}  (gzip stream)"
                    )
                    last_report = now

    elapsed = time.monotonic() - start_time
    speed = bytes_read / elapsed if elapsed > 0 else 0

    return {
        "lines_seen": lines_seen,
        "lines_kept": lines_kept,
        "bytes_read": bytes_read,
        "elapsed_seconds": round(elapsed, 2),
        "avg_speed": _format_speed(speed),
        "platform_distribution": dict(platform_counts.most_common()),
        "platforms_matched": len(platform_counts),
        "mode": "gzip_stream",
    }


# ---------------------------------------------------------------------------
# Merge shard outputs
# ---------------------------------------------------------------------------

def _merge_shards(shard_paths: list[str], output_path: Path) -> None:
    """Concatenate per-shard output files into the final output."""
    with open(output_path, "wb", buffering=_IO_BUFFER_SIZE) as fout:
        for shard_path in shard_paths:
            with open(shard_path, "rb", buffering=_IO_BUFFER_SIZE) as fin:
                while True:
                    chunk = fin.read(_IO_BUFFER_SIZE)
                    if not chunk:
                        break
                    fout.write(chunk)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_size(n: int | float) -> str:
    if n < 1024:
        return f"{n:.0f}B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f}MB"
    return f"{n / 1024 ** 3:.2f}GB"


def _format_speed(bps: float) -> str:
    if bps < 1024:
        return f"{bps:.0f}B/s"
    if bps < 1024 ** 2:
        return f"{bps / 1024:.1f}KB/s"
    return f"{bps / 1024 ** 2:.1f}MB/s"


def _print_progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_filter(
    input_path: Path,
    output_path: Path,
    *,
    workers: int,
    progress_interval: float = 30.0,
) -> dict[str, object]:
    """
    Filter *input_path*, keep lines matching a known AI platform,
    write them verbatim to *output_path*.
    """
    is_gzip = input_path.suffix == ".gz"

    if is_gzip:
        _print_progress(
            "Input is gzip-compressed — using single-process streaming mode.\n"
            "For full parallelism, decompress first:  gzip -dk input.jsonl.gz"
        )
        result = _process_gzip_stream(input_path, output_path, progress_interval)
        result["input"] = str(input_path)
        result["output"] = str(output_path)
        result["workers_used"] = 1
        return result

    # --- mmap + shard path ---
    file_size = input_path.stat().st_size
    _print_progress(f"Input size: {_format_size(file_size)}  ({file_size:,} bytes)")

    # Determine shard count.  With 112 threads available the OS page-cache
    # can feed many concurrent readers on the same file.  However, too many
    # tiny shards add overhead (process start, mmap setup, merge I/O).
    # Cap the minimum shard size at 64 MB.
    MIN_SHARD_BYTES = 64 * 1024 * 1024
    max_shards_by_size = max(1, file_size // MIN_SHARD_BYTES)
    num_shards = min(workers, max_shards_by_size)

    shard_size = file_size // num_shards
    shard_ranges: list[tuple[int, int]] = []
    for i in range(num_shards):
        start = i * shard_size
        end = file_size if i == num_shards - 1 else (i + 1) * shard_size
        shard_ranges.append((start, end))

    # Create a temporary directory for per-shard output files.
    tmp_dir = tempfile.mkdtemp(
        prefix="filter_by_platform_",
        dir=output_path.parent,
    )

    _print_progress(
        f"Sharding: {num_shards} shards across {workers} workers "
        f"(~{_format_size(shard_size)}/shard)"
    )
    _print_progress(f"Temp dir: {tmp_dir}\n")

    start_time = time.monotonic()

    shard_output_paths: list[str] = []
    shard_args: list[tuple] = []
    for i, (s, e) in enumerate(shard_ranges):
        shard_out = os.path.join(tmp_dir, f"shard_{i:04d}.jsonl")
        shard_output_paths.append(shard_out)
        shard_args.append((str(input_path), s, e, shard_out, i))

    # Run shards in a process pool.
    with multiprocessing.Pool(processes=num_shards) as pool:
        shard_results = pool.starmap(_process_shard, shard_args)

    shard_elapsed = time.monotonic() - start_time
    _print_progress(f"\nAll shards complete in {shard_elapsed:.1f}s.  Merging...")

    # Merge shard outputs.
    merge_start = time.monotonic()
    _merge_shards(shard_output_paths, output_path)
    merge_elapsed = time.monotonic() - merge_start
    _print_progress(f"Merge complete in {merge_elapsed:.1f}s.")

    # Clean up temp dir.
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Aggregate stats.
    total_lines_seen = 0
    total_lines_kept = 0
    total_bytes_scanned = 0
    platform_counts: Counter[str] = Counter()

    for result in shard_results:
        total_lines_seen += result["lines_seen"]
        total_lines_kept += result["lines_kept"]
        total_bytes_scanned += result["bytes_scanned"]
        platform_counts.update(result["platform_counts"])

    total_elapsed = time.monotonic() - start_time
    avg_speed = total_bytes_scanned / shard_elapsed if shard_elapsed > 0 else 0
    distribution = dict(platform_counts.most_common())

    return {
        "input": str(input_path),
        "output": str(output_path),
        "mode": "mmap_sharded",
        "workers_used": num_shards,
        "lines_seen": total_lines_seen,
        "lines_kept": total_lines_kept,
        "bytes_scanned": total_bytes_scanned,
        "shard_seconds": round(shard_elapsed, 2),
        "merge_seconds": round(merge_elapsed, 2),
        "elapsed_seconds": round(total_elapsed, 2),
        "avg_parallel_throughput": _format_speed(avg_speed),
        "platform_distribution": distribution,
        "platforms_matched": len(distribution),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Stage 1 prompt-link JSONL to rows matching known AI "
            "platforms and report the platform distribution."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL (or .jsonl.gz) from Stage 1.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for matched rows.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel shard workers (default: 0 = auto, which uses "
            "min(cpu_count, file_size / 64MB)).  Ignored for gzip input."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between progress reports to stderr (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers > 0 else os.cpu_count() or 4

    _print_progress(f"Input:   {input_path}")
    _print_progress(f"Output:  {output_path}")
    _print_progress(f"CPUs:    {os.cpu_count()} logical cores available")
    _print_progress(f"Workers: {workers} requested")
    _print_progress("")

    summary = run_filter(
        input_path,
        output_path,
        workers=workers,
        progress_interval=args.progress_interval,
    )

    # Final progress.
    _print_progress("")
    _print_progress(
        f"Done.  {summary['lines_seen']:,} lines read, "
        f"{summary['lines_kept']:,} kept "
        f"in {summary['elapsed_seconds']}s "
        f"({summary.get('avg_parallel_throughput', summary.get('avg_speed', '?'))})"
    )

    # Platform distribution table to stderr.
    dist = summary["platform_distribution"]
    if dist:
        _print_progress("")
        _print_progress("Platform distribution:")
        max_name_len = max(len(name) for name in dist)
        total_kept = summary["lines_kept"]
        for name, count in sorted(dist.items(), key=lambda kv: -kv[1]):
            pct = count / total_kept * 100 if total_kept else 0
            _print_progress(f"  {name:<{max_name_len}}  {count:>12,}  ({pct:5.1f}%)")
        _print_progress(f"  {'TOTAL':<{max_name_len}}  {total_kept:>12,}")

    # Machine-readable summary to stdout.
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())