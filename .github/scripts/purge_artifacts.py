#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delete GitHub Actions artifacts older than the repository retention window.

`list` crawls the artifact API and writes the candidates to a file for
inspection, grouped by workflow run with the largest runs first. `delete`
removes the artifacts in that file and records progress next to it, so an
interrupted run resumes where it stopped. `auto` repeats list and delete until
nothing older than the cutoff is left. Uses GITHUB_TOKEN if set, otherwise the
token of the logged-in gh CLI. Progress is logged to stderr; results go to
stdout. Rate limits never fail a run: the script waits for the reset and goes on.
"""

import argparse
import itertools
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError

from github_utils import gh_fetch_url_and_headers, GITHUB_API_URL

PER_PAGE = 100
WORKERS = 4
# The repository setting; anything younger is GitHub's to keep.
RETENTION_DAYS = 14
# GitHub's longest retention; nothing older can still be live.
MAX_RETENTION_DAYS = 90
MAX_ATTEMPTS = 5
# Requests per hour left untouched so that gh keeps working for whoever runs this.
RATE_LIMIT_RESERVE = 500
DEFAULT_PER_MINUTE = 75
MAX_CYCLE_FAILURES = 5
RETRY_MINUTES = 5
SOCKET_TIMEOUT_SECONDS = 60
LOG_EVERY_PAGES = 100
LOG_EVERY_DELETES = 250


def seconds_until_reset(headers: Any) -> float:
    return max(0.0, float(headers.get("X-RateLimit-Reset", 0)) - time.time()) + 1


def is_rate_limited(err: HTTPError, body: str) -> bool:
    if err.code == 429 or "Retry-After" in err.headers:
        return True
    return err.code == 403 and "rate limit" in body.lower()


def request(url: str, method: str = "GET") -> bytes:
    path = url[len(GITHUB_API_URL) :]
    failures = 0
    while True:
        started = time.monotonic()
        try:
            headers, body = gh_fetch_url_and_headers(
                url,
                headers={"Accept": "application/vnd.github+json"},
                method=method,
                reader=lambda conn: conn.read(),
            )
        except HTTPError as err:
            text = err.read().decode(errors="replace").strip()[:200]
            if is_rate_limited(err, text):
                if "Retry-After" in err.headers:
                    wait = float(err.headers["Retry-After"])
                elif err.headers.get("X-RateLimit-Remaining") == "0":
                    wait = seconds_until_reset(err.headers)
                else:
                    wait = 60.0
                logging.warning(
                    "%s %s -> %d %s, rate limited, resuming in %.0fs",
                    method,
                    path,
                    err.code,
                    text,
                    wait,
                )
                time.sleep(wait)
                continue
            failures += 1
            if err.code < 500 or failures == MAX_ATTEMPTS:
                logging.info("%s %s -> %d %s", method, path, err.code, text)
                raise
            wait = float(2**failures)
            logging.warning(
                "%s %s -> %d %s, retrying in %.0fs", method, path, err.code, text, wait
            )
            time.sleep(wait)
        except (OSError, HTTPException) as err:
            failures += 1
            if failures == MAX_ATTEMPTS:
                raise
            wait = float(2**failures)
            logging.warning(
                "%s %s failed (%r), retrying in %.0fs", method, path, err, wait
            )
            time.sleep(wait)
        else:
            remaining = int(headers.get("X-RateLimit-Remaining", RATE_LIMIT_RESERVE))
            logging.info(
                "%s %s in %.1fs, %d of %s hourly requests left",
                method,
                path,
                time.monotonic() - started,
                remaining,
                headers.get("X-RateLimit-Limit", "?"),
            )
            if remaining < RATE_LIMIT_RESERVE:
                wait = seconds_until_reset(headers)
                logging.info("Hourly rate limit nearly used up, sleeping %.0fs", wait)
                time.sleep(wait)
            return body


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def tb(size_in_bytes: int) -> str:
    return f"{size_in_bytes / 1e12:.1f}"


def fetch_page(repo: str, page: int) -> List[Dict[str, Any]]:
    url = f"{GITHUB_API_URL}/repos/{repo}/actions/artifacts?per_page={PER_PAGE}&page={page}"
    return json.loads(request(url))["artifacts"]


def first_page_older_than(repo: str, edge: datetime) -> int:
    """Bisect page numbers (the list is newest-first) for the first page ending past edge."""
    total = json.loads(
        request(f"{GITHUB_API_URL}/repos/{repo}/actions/artifacts?per_page=1")
    )
    lo, hi = 1, -(-total["total_count"] // PER_PAGE) + 1
    while lo < hi:
        mid = (lo + hi) // 2
        artifacts = fetch_page(repo, mid)
        if not artifacts or parse_time(artifacts[-1]["created_at"]) < edge:
            hi = mid
        else:
            lo = mid + 1
    return lo


def list_candidates(repo: str, older_than_days: int) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=older_than_days)
    oldest_live = now - timedelta(days=MAX_RETENTION_DAYS)
    logging.info(
        "Listing artifacts of %s older than %d days (created before %s)",
        repo,
        older_than_days,
        cutoff.strftime("%Y-%m-%d %H:%M UTC"),
    )
    start = time.monotonic()
    page = first_page_older_than(repo, cutoff)
    logging.info("Artifacts older than %d days start on page %d", older_than_days, page)
    # Keyed by id: uploads arriving during the crawl shift the list, so pages can overlap.
    candidates: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(WORKERS) as pool:
        while True:
            pages = pool.map(lambda p: fetch_page(repo, p), range(page, page + WORKERS))
            artifacts = [artifact for artifacts in pages for artifact in artifacts]
            for artifact in artifacts:
                if (
                    not artifact["expired"]
                    and parse_time(artifact["created_at"]) < cutoff
                ):
                    run = artifact["workflow_run"]
                    candidates[artifact["id"]] = {
                        "id": artifact["id"],
                        "name": artifact["name"],
                        "size_in_bytes": artifact["size_in_bytes"],
                        "created_at": artifact["created_at"],
                        "run_url": (
                            f"https://github.com/{repo}/actions/runs/{run['id']}"
                            if run
                            else None
                        ),
                    }
            if not artifacts or parse_time(artifacts[-1]["created_at"]) < oldest_live:
                break
            page += WORKERS
            if page % LOG_EVERY_PAGES < WORKERS:
                logging.info(
                    "page %d: %s candidates (%s TB) so far, oldest seen %s",
                    page,
                    f"{len(candidates):,}",
                    tb(sum(a["size_in_bytes"] for a in candidates.values())),
                    artifacts[-1]["created_at"],
                )
    logging.info(
        "Crawled up to page %d in %.1f minutes",
        page + WORKERS - 1,
        (time.monotonic() - start) / 60,
    )
    # Largest runs first, artifacts of one run together, largest artifact first within it.
    run_bytes: Counter = Counter()
    for artifact in candidates.values():
        run_bytes[artifact["run_url"]] += artifact["size_in_bytes"]
    return sorted(
        candidates.values(),
        key=lambda a: (
            -run_bytes[a["run_url"]],
            a["run_url"] or "",
            -a["size_in_bytes"],
        ),
    )


def write_candidates(
    candidates: List[Dict[str, Any]], out: Path, older_than_days: int
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for artifact in candidates:
            f.write(json.dumps(artifact) + "\n")
    total = sum(artifact["size_in_bytes"] for artifact in candidates)
    print(
        f"✅ {len(candidates):,} artifacts older than {older_than_days} days hold "
        f"{tb(total)} TB. Written to {out}"
    )


def delete_candidates(
    repo: str, candidates_path: Path, per_minute: float, limit: Optional[int]
) -> Tuple[int, int]:
    done_path = candidates_path.with_name(candidates_path.name + ".done")
    done = set()
    if done_path.exists():
        done = {int(line) for line in done_path.read_text().split()}
    todo = [
        json.loads(line) for line in candidates_path.read_text().splitlines() if line
    ]
    todo = [artifact for artifact in todo if artifact["id"] not in done][:limit]
    logging.info(
        "%s already deleted, %s to go (%s TB) at %g per minute; progress is recorded in %s",
        f"{len(done):,}",
        f"{len(todo):,}",
        tb(sum(artifact["size_in_bytes"] for artifact in todo)),
        per_minute,
        done_path,
    )
    start = time.monotonic()
    count = 0
    freed = 0

    def delete_one(artifact: Dict[str, Any]) -> Dict[str, Any]:
        try:
            request(
                f"{GITHUB_API_URL}/repos/{repo}/actions/artifacts/{artifact['id']}",
                "DELETE",
            )
        except HTTPError as err:
            if err.code != 404:
                raise
        return artifact

    # Small chunks, each given its share of the minute, keep the rate steady and
    # leave little queued when an error or Ctrl-C stops the run.
    chunk = 5 * WORKERS
    next_chunk_at = time.monotonic()
    with ThreadPoolExecutor(WORKERS) as pool, done_path.open("a") as done_file:
        for chunk_start in range(0, len(todo), chunk):
            wait = next_chunk_at - time.monotonic()
            if wait > 0:
                logging.info(
                    "Waiting %.1fs to stay at %g deletes per minute, %s left...",
                    wait,
                    per_minute,
                    f"{len(todo) - chunk_start:,}",
                )
                time.sleep(wait)
            next_chunk_at = time.monotonic() + chunk * 60 / per_minute
            for artifact in pool.map(
                delete_one, todo[chunk_start : chunk_start + chunk]
            ):
                done_file.write(f"{artifact['id']}\n")
                done_file.flush()
                count += 1
                freed += artifact["size_in_bytes"]
                if count % LOG_EVERY_DELETES == 0:
                    elapsed = time.monotonic() - start
                    eta = elapsed / count * (len(todo) - count)
                    logging.info(
                        "%s/%s deleted, %s TB freed, %.1f h elapsed, about %.1f h left",
                        f"{count:,}",
                        f"{len(todo):,}",
                        tb(freed),
                        elapsed / 3600,
                        eta / 3600,
                    )
    print(
        f"Deleted {count:,} artifacts ({tb(freed)} TB) in "
        f"{(time.monotonic() - start) / 3600:.1f} h; {len(done) + count:,} recorded in {done_path}"
    )
    return count, freed


def auto(repo: str, older_than_days: int, per_minute: float) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="purge-artifacts-"))
    logging.info("Candidates files and progress are kept in %s", workdir)
    cycles = deleted = freed = failures = 0
    for attempt in itertools.count(1):
        try:
            candidates = list_candidates(repo, older_than_days)
            if not candidates:
                break
            out = workdir / f"candidates-{attempt}.jsonl"
            write_candidates(candidates, out, older_than_days)
            count, size = delete_candidates(repo, out, per_minute, limit=None)
        except Exception:
            failures += 1
            if failures == MAX_CYCLE_FAILURES:
                raise
            logging.exception(
                "Cycle %d failed, starting over in %d minutes", attempt, RETRY_MINUTES
            )
            time.sleep(RETRY_MINUTES * 60)
            continue
        cycles += 1
        deleted += count
        freed += size
        failures = 0
    print(
        f"Nothing older than {older_than_days} days remains; deleted {deleted:,} "
        f"artifacts ({tb(freed)} TB) in {cycles} cycles"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lister = sub.add_parser(
        "list",
        help="write the artifacts older than the cutoff to a file, largest runs first",
    )
    lister.add_argument("--repo", default="pytorch/executorch")
    lister.add_argument(
        "--older-than-days",
        type=int,
        default=RETENTION_DAYS,
        help=f"at least {RETENTION_DAYS}, the repository retention (default)",
    )
    lister.add_argument(
        "--out",
        type=Path,
        help="candidates file; defaults to a new temporary directory",
    )
    deleter = sub.add_parser(
        "delete",
        help="delete the artifacts in a file written by list, resuming if interrupted",
    )
    deleter.add_argument("--repo", default="pytorch/executorch")
    deleter.add_argument("--candidates", type=Path, required=True)
    deleter.add_argument("--per-minute", type=float, default=DEFAULT_PER_MINUTE)
    deleter.add_argument("--limit", type=int, help="stop after this many deletions")
    runner = sub.add_parser(
        "auto",
        help="list and delete, again and again, until nothing older than the cutoff is left",
    )
    runner.add_argument("--repo", default="pytorch/executorch")
    runner.add_argument(
        "--older-than-days",
        type=int,
        default=RETENTION_DAYS,
        help=f"at least {RETENTION_DAYS}, the repository retention (default)",
    )
    runner.add_argument("--per-minute", type=float, default=DEFAULT_PER_MINUTE)
    args = parser.parse_args()
    if args.command != "delete" and args.older_than_days < RETENTION_DAYS:
        parser.error(f"--older-than-days must be at least {RETENTION_DAYS}")
    return args


def main() -> None:
    args = parse_args()
    log_format = "%(asctime)s %(message)s"
    if sys.stderr.isatty():
        log_format = f"\033[2m{log_format}\033[0m"
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format=log_format,
        datefmt="%H:%M:%S",
    )
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
    if not os.environ.get("GITHUB_TOKEN"):
        try:
            os.environ["GITHUB_TOKEN"] = subprocess.check_output(
                ["gh", "auth", "token"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            sys.exit("GITHUB_TOKEN is not set and `gh auth token` failed")
    if args.command == "list":
        out = (
            args.out
            or Path(tempfile.mkdtemp(prefix="purge-artifacts-")) / "candidates.jsonl"
        )
        write_candidates(
            list_candidates(args.repo, args.older_than_days), out, args.older_than_days
        )
    elif args.command == "delete":
        delete_candidates(args.repo, args.candidates, args.per_minute, args.limit)
    else:
        auto(args.repo, args.older_than_days, args.per_minute)


if __name__ == "__main__":
    main()
