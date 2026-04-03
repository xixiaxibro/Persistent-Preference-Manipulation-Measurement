"""
Collect prompt-carrying links from Common Crawl WAT files.

This pipeline keeps the download and WAT parsing architecture from the original
collector, but removes per-link platform attribution from the hot path.
It only keeps links whose query string contains prompt-like parameters.
"""
from __future__ import annotations

import argparse
import gzip
import html
import io
import json
import multiprocessing
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator
from urllib import request
from urllib.parse import parse_qs, urlparse

from env_config import load_project_env

load_project_env()

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from warcio.archiveiterator import ArchiveIterator
except ImportError:
    ArchiveIterator = None


COMMON_CRAWL_DATA_ROOT = "https://data.commoncrawl.org"
_DOWNLOAD_CHUNK_SIZE = 256 * 1024
_PROMPT_QUERY_KEYS = frozenset({"q", "prompt"})
_GROK_TEXT_QUERY_KEY = "text"
_GROK_TEXT_HOSTS = frozenset({"x.com", "www.x.com"})
_MAX_IN_FLIGHT_MULTIPLIER = 4


def _require_warcio() -> None:
    if ArchiveIterator is not None:
        return
    raise RuntimeError(
        "Missing dependency: warcio. Install it with `python3 -m pip install warcio`."
    )


def _download_bytes(url: str) -> bytes:
    with request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _iter_lines_from_gzip_bytes(blob: bytes) -> Iterator[str]:
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as handle:
        for raw_line in handle:
            yield raw_line.decode("utf-8", errors="replace").strip()


def load_wat_paths(crawl: str, *, paths_url: str | None = None) -> list[str]:
    resolved = paths_url or f"{COMMON_CRAWL_DATA_ROOT}/crawl-data/{crawl}/wat.paths.gz"
    blob = _download_bytes(resolved)
    return [line for line in _iter_lines_from_gzip_bytes(blob) if line]


def iter_local_wat_paths(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def _safe_get_link_target(link_record: dict[str, Any]) -> str:
    for key in ("url", "path", "href"):
        value = link_record.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate.startswith("http://") or candidate.startswith("https://"):
                return candidate
    return ""


def _safe_get_anchor_text(link_record: dict[str, Any]) -> str:
    for key in ("text", "anchor", "alt"):
        value = link_record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _format_size(n: int | float) -> str:
    if n < 1024:
        return f"{n:.0f}B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    if n < 1024 ** 3:
        return f"{n / 1024**2:.1f}MB"
    return f"{n / 1024**3:.2f}GB"


def _format_speed(bps: float) -> str:
    if bps < 1024:
        return f"{bps:.0f}B/s"
    if bps < 1024 ** 2:
        return f"{bps / 1024:.1f}KB/s"
    return f"{bps / 1024**2:.1f}MB/s"


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _extract_links_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    envelope = payload.get("Envelope", {})
    if not isinstance(envelope, dict):
        return []
    payload_metadata = envelope.get("Payload-Metadata", {})
    if not isinstance(payload_metadata, dict):
        return []
    http_response_metadata = payload_metadata.get("HTTP-Response-Metadata", {})
    if not isinstance(http_response_metadata, dict):
        return []
    html_metadata = http_response_metadata.get("HTML-Metadata", {})
    if not isinstance(html_metadata, dict):
        return []
    links = html_metadata.get("Links", [])
    if not isinstance(links, list):
        return []
    return [item for item in links if isinstance(item, dict)]


def _normalize_prompt_value(value: str) -> str:
    text = html.unescape(value)
    text = text.replace("\u0000", " ")
    return re.sub(r"\s+", " ", text).strip()


def _might_have_prompt_parameter(url: str) -> bool:
    lowered = url.lower()
    return (
        "?q=" in lowered
        or "&q=" in lowered
        or "?prompt=" in lowered
        or "&prompt=" in lowered
        or (
            ("?text=" in lowered or "&text=" in lowered)
            and ("//x.com/i/grok" in lowered or "//www.x.com/i/grok" in lowered)
        )
    )


def extract_prompt_parameters(url: str) -> dict[str, list[str]]:
    if not _might_have_prompt_parameter(url):
        return {}

    try:
        parsed = urlparse(url)
    except ValueError:
        return {}

    if not parsed.query:
        return {}

    try:
        raw_query = parse_qs(parsed.query, keep_blank_values=False)
    except ValueError:
        return {}

    allow_grok_text = (
        parsed.netloc.lower() in _GROK_TEXT_HOSTS
        and parsed.path.lower().startswith("/i/grok")
    )

    results: dict[str, list[str]] = {}
    for key, values in raw_query.items():
        normalized_key = key.lower()
        if normalized_key in _PROMPT_QUERY_KEYS:
            pass
        elif normalized_key == _GROK_TEXT_QUERY_KEY and allow_grok_text:
            pass
        else:
            continue
        cleaned = []
        for value in values:
            normalized_value = _normalize_prompt_value(value)
            if normalized_value:
                cleaned.append(normalized_value)
        if cleaned:
            results[normalized_key] = cleaned
    return results


class QueueStream:
    def __init__(self, q: queue.Queue):
        self._q = q
        self._buf = b""
        self._eof = False

    def read(self, size: int = -1) -> bytes:
        if self._eof and not self._buf:
            return b""
        while not self._eof and (size < 0 or len(self._buf) < size):
            try:
                chunk = self._q.get(timeout=600)
            except queue.Empty:
                self._eof = True
                break
            if chunk is None:
                self._eof = True
                break
            self._buf += chunk
        if size < 0:
            result = self._buf
            self._buf = b""
        else:
            result = self._buf[:size]
            self._buf = self._buf[size:]
        return result

    def readline(self, size: int = -1) -> bytes:
        while not self._eof and b"\n" not in self._buf:
            try:
                chunk = self._q.get(timeout=600)
            except queue.Empty:
                self._eof = True
                break
            if chunk is None:
                self._eof = True
                break
            self._buf += chunk
        idx = self._buf.find(b"\n")
        if idx >= 0:
            result = self._buf[: idx + 1]
            self._buf = self._buf[idx + 1 :]
        else:
            result = self._buf
            self._buf = b""
        return result

    def close(self) -> None:
        pass


def _download_thread(
    url: str,
    data_q: queue.Queue,
    stats_q: multiprocessing.Queue,
    worker_id: int,
) -> None:
    try:
        resp = request.urlopen(url, timeout=300)
        content_length = resp.headers.get("Content-Length")
        total_bytes = int(content_length) if content_length else None
        stats_q.put(("file_size", worker_id, total_bytes))

        while True:
            chunk = resp.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            data_q.put(chunk)
            stats_q.put(("dl_bytes", worker_id, len(chunk)))
        resp.close()
    except Exception as exc:
        stats_q.put(("dl_error", worker_id, str(exc)))
    finally:
        data_q.put(None)


def process_one_wat(
    wat_url: str,
    crawl: str,
    worker_id: int,
    stats_q: multiprocessing.Queue,
) -> list[dict[str, Any]]:
    _require_warcio()
    stats_q.put(("start", worker_id, wat_url))

    data_q: queue.Queue = queue.Queue(maxsize=64)
    dl = threading.Thread(target=_download_thread, args=(wat_url, data_q, stats_q, worker_id), daemon=True)
    dl.start()

    candidates: list[dict[str, Any]] = []
    stream = QueueStream(data_q)
    pages = 0
    links = 0

    try:
        for record in ArchiveIterator(stream):
            if record.rec_type != "metadata":
                continue
            target_uri = record.rec_headers.get_header("WARC-Target-URI") or ""
            if not target_uri:
                continue
            try:
                payload = json.loads(record.content_stream().read())
            except (json.JSONDecodeError, Exception):
                continue

            page_links = _extract_links_from_payload(payload)
            pages += 1
            links += len(page_links)

            if pages % 500 == 0:
                stats_q.put(("parse_progress", worker_id, pages, links))

            for link_record in page_links:
                try:
                    target_url = _safe_get_link_target(link_record)
                    if not target_url:
                        continue

                    prompt_parameters = extract_prompt_parameters(target_url)
                    if not prompt_parameters:
                        continue

                    candidates.append(
                        {
                            "crawl": crawl,
                            "wat_file": wat_url,
                            "source_url": target_uri,
                            "target_url": target_url,
                            "anchor_text": _safe_get_anchor_text(link_record),
                            "link_path": link_record.get("path", ""),
                            "prompt_parameters": prompt_parameters,
                        }
                    )
                except Exception:
                    continue
    except Exception:
        pass

    dl.join(timeout=30)
    stats_q.put(("done", worker_id, len(candidates), pages, links))
    return candidates


class ProgressTracker:
    def __init__(self, total_files: int):
        self.total_files = total_files
        self._lock = threading.Lock()
        self._workers: dict[int, dict[str, Any]] = {}
        self.files_done = 0
        self.files_failed = 0
        self.total_candidates = 0
        self.total_bytes = 0
        self.total_pages = 0
        self.total_links = 0
        self._start = time.time()

    def handle_message(self, msg: tuple) -> None:
        kind = msg[0]
        with self._lock:
            if kind == "start":
                _, wid, url = msg
                self._workers[wid] = {
                    "url": url,
                    "dl_bytes": 0,
                    "total": None,
                    "pages": 0,
                    "links": 0,
                    "start": time.time(),
                    "done": False,
                }
            elif kind == "file_size":
                _, wid, total = msg
                if wid in self._workers:
                    self._workers[wid]["total"] = total
            elif kind == "dl_bytes":
                _, wid, n = msg
                if wid in self._workers:
                    self._workers[wid]["dl_bytes"] += n
                self.total_bytes += n
            elif kind == "parse_progress":
                _, wid, pages, links = msg
                if wid in self._workers:
                    self._workers[wid]["pages"] = pages
                    self._workers[wid]["links"] = links
            elif kind == "done":
                _, wid, candidates, pages, links = msg
                if wid in self._workers:
                    self._workers[wid]["pages"] = pages
                    self._workers[wid]["links"] = links
                    self._workers[wid]["done"] = True
                if candidates >= 0:
                    self.files_done += 1
                    self.total_candidates += candidates
                else:
                    self.files_failed += 1
                self.total_pages += pages
                self.total_links += links

    def mark_future_failed(self, wid: int) -> None:
        with self._lock:
            worker = self._workers.get(wid)
            if worker is not None and worker["done"]:
                return
            if worker is not None:
                self.total_pages += worker["pages"]
                self.total_links += worker["links"]
                worker["done"] = True
            self.files_failed += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._start
            avg_speed = self.total_bytes / elapsed if elapsed > 0 else 0
            active = []
            for wid, info in sorted(self._workers.items()):
                if info["done"]:
                    continue
                dt = time.time() - info["start"]
                speed = info["dl_bytes"] / dt if dt > 0 else 0
                pct = (info["dl_bytes"] / info["total"] * 100) if info["total"] else None
                active.append(
                    {
                        "wid": wid,
                        "dl_bytes": info["dl_bytes"],
                        "total": info["total"],
                        "pct": pct,
                        "speed": speed,
                        "pages": info["pages"],
                        "links": info["links"],
                    }
                )
            return {
                "elapsed": elapsed,
                "files_done": self.files_done,
                "files_failed": self.files_failed,
                "total_files": self.total_files,
                "total_candidates": self.total_candidates,
                "total_bytes": self.total_bytes,
                "total_pages": self.total_pages,
                "total_links": self.total_links,
                "avg_speed": avg_speed,
                "active": active,
            }


def _stats_collector(stats_q: multiprocessing.Queue, tracker: ProgressTracker, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            msg = stats_q.get(timeout=0.5)
            tracker.handle_message(msg)
        except queue.Empty:
            continue
        except Exception:
            continue


def _progress_printer(tracker: ProgressTracker, stop_event: threading.Event, interval: float = 30.0) -> None:
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break

        snap = tracker.snapshot()
        done = snap["files_done"] + snap["files_failed"]
        active_count = len(snap["active"])
        _print_progress(
            f"[{snap['elapsed']:>5.0f}s] "
            f"Files: {done}/{snap['total_files']} done "
            f"({snap['files_failed']} err) | "
            f"Active: {active_count} | "
            f"Prompt-links: {snap['total_candidates']} | "
            f"Pages: {snap['total_pages']} | "
            f"Links: {snap['total_links']} | "
            f"DL: {_format_size(snap['total_bytes'])} "
            f"({_format_speed(snap['avg_speed'])})"
        )


def _load_completed(progress_path: Path) -> set[str]:
    completed: set[str] = set()
    if not progress_path.exists():
        return completed
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                completed.add(line)
    return completed


def _mark_completed(progress_path: Path, url: str) -> None:
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(url + "\n")


def _run_pipeline(
    wat_urls: list[str],
    crawl: str,
    output_path: Path,
    workers: int,
    progress_path: Path,
) -> dict[str, Any]:
    completed = _load_completed(progress_path)
    remaining = [url for url in wat_urls if url not in completed]
    skipped = len(wat_urls) - len(remaining)

    total_all = len(wat_urls)
    total = len(remaining)
    tracker = ProgressTracker(total_all)
    tracker.files_done = skipped

    if skipped > 0:
        _print_progress(f"Resume: {skipped} files already done, {total} remaining.")

    if total == 0:
        _print_progress("All files already completed. Nothing to do.")
        return {
            "crawl": crawl,
            "output": str(output_path),
            "wat_files_ok": skipped,
            "wat_files_failed": 0,
            "prompt_link_rows_written": 0,
            "pages_scanned": 0,
            "links_scanned": 0,
            "total_bytes_downloaded": 0,
            "elapsed_seconds": 0,
            "avg_download_speed": "0B/s",
            "resumed_from": skipped,
            "filter_keys": sorted(_PROMPT_QUERY_KEYS),
            "special_cases": ["x.com/i/grok:text"],
        }

    manager = multiprocessing.Manager()
    stats_q = manager.Queue()

    stop_event = threading.Event()
    collector = threading.Thread(target=_stats_collector, args=(stats_q, tracker, stop_event), daemon=True)
    printer = threading.Thread(target=_progress_printer, args=(tracker, stop_event, 30.0), daemon=True)
    collector.start()
    printer.start()

    _print_progress(f"Pipeline2: {total} WAT files to process ({total_all} total), {workers} worker processes")
    _print_progress("Filter: keep q=/prompt= globally, plus x.com/i/grok?text= as a special case")
    _print_progress(f"Server CPUs: {os.cpu_count()} logical cores available\n")

    total_new_candidates = 0
    max_in_flight = max(workers, workers * _MAX_IN_FLIGHT_MULTIPLIER)

    with output_path.open("a", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            remaining_iter = iter(enumerate(remaining))
            futures = {}

            def submit_until_window_full() -> None:
                while len(futures) < max_in_flight:
                    try:
                        worker_id, url = next(remaining_iter)
                    except StopIteration:
                        break
                    future = pool.submit(process_one_wat, url, crawl, worker_id, stats_q)
                    futures[future] = (worker_id, url)

            submit_until_window_full()

            while futures:
                done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    wid, url = futures.pop(future)
                    name = url.rsplit("/", 1)[-1]
                    try:
                        candidates = future.result()
                        for row in candidates:
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        _mark_completed(progress_path, url)
                        total_new_candidates += len(candidates)
                        time.sleep(0.3)
                        snap = tracker.snapshot()
                        done = snap["files_done"] + snap["files_failed"]
                        _print_progress(
                            f"  ✓ [{done}/{total_all}] W{wid}: +{len(candidates)} prompt-links from {name}  "
                            f"(cumul: {total_new_candidates} prompt-links, "
                            f"{snap['total_pages']} pages)"
                        )
                    except Exception as exc:
                        tracker.mark_future_failed(wid)
                        _print_progress(f"  ✗ W{wid}: FAILED {name}: {exc}")

                submit_until_window_full()

    time.sleep(1)
    stop_event.set()
    collector.join(timeout=3)
    printer.join(timeout=3)

    snap = tracker.snapshot()
    return {
        "crawl": crawl,
        "output": str(output_path),
        "wat_files_ok": snap["files_done"],
        "wat_files_failed": snap["files_failed"],
        "prompt_link_rows_written": total_new_candidates,
        "pages_scanned": snap["total_pages"],
        "links_scanned": snap["total_links"],
        "total_bytes_downloaded": snap["total_bytes"],
        "elapsed_seconds": round(snap["elapsed"], 2),
        "avg_download_speed": _format_speed(snap["avg_speed"]),
        "resumed_from": skipped,
        "filter_keys": sorted(_PROMPT_QUERY_KEYS),
        "special_cases": ["x.com/i/grok:text"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect prompt-carrying links from Common Crawl WAT files."
    )
    parser.add_argument("--crawl", help="Common Crawl identifier, e.g. CC-MAIN-2026-04.")
    parser.add_argument("--paths-url", help="Optional custom URL for wat.paths.gz.")
    parser.add_argument("--paths-file", help="Optional local wat.paths or wat.paths.gz file.")
    parser.add_argument("--max-wat-files", type=int, default=0, help="Maximum number of WAT files to inspect (0 = all).")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent worker processes (default: 20).")
    parser.add_argument(
        "--progress-file",
        help="Path to progress file for resume support. Default: {output}.progress",
    )
    return parser.parse_args()


def _resolve_wat_urls(args: argparse.Namespace) -> list[str]:
    if args.paths_file:
        raw_paths = list(iter_local_wat_paths(Path(args.paths_file)))
    else:
        if not args.crawl:
            raise ValueError("Either --crawl or --paths-file is required.")
        raw_paths = load_wat_paths(args.crawl, paths_url=args.paths_url)

    wat_urls = []
    for item in raw_paths:
        if item.startswith("http://") or item.startswith("https://"):
            wat_urls.append(item)
        else:
            wat_urls.append(f"{COMMON_CRAWL_DATA_ROOT}/{item.lstrip('/')}")
    return wat_urls


def main() -> int:
    args = parse_args()
    crawl = args.crawl or "unknown-crawl"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path(args.progress_file) if args.progress_file else Path(args.output + ".progress")

    try:
        all_wat_urls = _resolve_wat_urls(args)
    except Exception as exc:
        print(f"Failed to resolve WAT paths: {exc}", file=sys.stderr)
        return 1

    wat_urls = all_wat_urls[: args.max_wat_files] if args.max_wat_files > 0 else all_wat_urls

    _print_progress(f"Resolved {len(all_wat_urls)} WAT files, will process {len(wat_urls)}.")
    _print_progress(f"Progress file: {progress_path}")

    summary = _run_pipeline(
        wat_urls=wat_urls,
        crawl=crawl,
        output_path=output_path,
        workers=args.workers,
        progress_path=progress_path,
    )

    _print_progress("")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())