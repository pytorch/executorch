#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delete old GitHub Actions runs, including their artifacts, logs, and history.

`list` crawls the artifact API and writes one candidate per workflow run,
largest first. `delete` verifies that each run is completed, has not been
updated since the cutoff, and contains no newer artifacts before deleting it.
Progress is recorded in a .runs.done file, so interrupted deletions resume.
`auto` repeats list and delete until no candidates remain or no runs can be
deleted. Candidate files from the artifact-only script must be regenerated.
Uses GITHUB_TOKEN if set, otherwise the token of the logged-in gh CLI. Progress
is logged to stderr; results go to stdout. Rate limits never fail a run: the
script waits for the reset and goes on.
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
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


class RunNotEligible(Exception):
    pass


def seconds_until_reset(headers: Any) -> float:
    return max(0.0, float(headers.get("X-RateLimit-Reset", 0)) - time.time()) + 1


def is_rate_limited(err: HTTPError, body: str) -> bool:
    if err.code == 429 or "Retry-After" in err.headers:
        return True
    return err.code == 403 and "rate limit" in body.lower()


def request(
    url: str,
    method: str = "GET",
    before_attempt: Optional[Callable[[], None]] = None,
) -> bytes:
    path = url[len(GITHUB_API_URL) :]
    failures = 0
    while True:
        if before_attempt is not None:
            before_attempt()
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
                if method == "GET":
                    # Eligibility checks need metadata fetched after the wait.
                    continue
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
                    if not run:
                        continue
                    candidates[artifact["id"]] = {
                        "run_id": run["id"],
                        "size_in_bytes": artifact["size_in_bytes"],
                    }
            if not artifacts or parse_time(artifacts[-1]["created_at"]) < oldest_live:
                break
            page += WORKERS
            if page % LOG_EVERY_PAGES < WORKERS:
                logging.info(
                    "page %d: %s old artifacts (%s TB) so far, oldest seen %s",
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
    runs: Dict[int, Dict[str, Any]] = {}
    for artifact in candidates.values():
        run_id = artifact["run_id"]
        run = runs.setdefault(
            run_id,
            {
                "repo": repo,
                "run_id": run_id,
                "run_url": f"https://github.com/{repo}/actions/runs/{run_id}",
                "cutoff": cutoff.isoformat(),
                "artifact_count": 0,
                "size_in_bytes": 0,
            },
        )
        run["artifact_count"] += 1
        run["size_in_bytes"] += artifact["size_in_bytes"]
    return sorted(runs.values(), key=lambda run: (-run["size_in_bytes"], run["run_id"]))


def write_candidates(
    candidates: List[Dict[str, Any]], out: Path, older_than_days: int
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for run in candidates:
            f.write(json.dumps(run) + "\n")
    total = sum(run["size_in_bytes"] for run in candidates)
    artifacts = sum(run["artifact_count"] for run in candidates)
    print(
        f"✅ {len(candidates):,} candidate runs hold {artifacts:,} artifacts older than "
        f"{older_than_days} days ({tb(total)} TB). Written to {out}. "
        "Deletion also removes the runs' logs and history; eligibility is checked before deletion."
    )


def delete_candidates(
    repo: str, candidates_path: Path, per_minute: float, limit: Optional[int]
) -> Tuple[int, int]:
    done_path = candidates_path.with_name(candidates_path.name + ".runs.done")
    done = set()
    if done_path.exists():
        done = {int(line) for line in done_path.read_text().split()}
    todo = [
        json.loads(line) for line in candidates_path.read_text().splitlines() if line
    ]
    latest_cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    seen = set()
    for run in todo:
        if "run_id" not in run:
            raise ValueError(
                "Artifact candidate files are unsupported; regenerate with list"
            )
        if run["repo"] != repo:
            raise ValueError(
                f"Candidate repository {run['repo']} does not match {repo}"
            )
        if type(run["run_id"]) is not int or run["run_id"] <= 0:
            raise ValueError("Candidate run IDs must be positive integers")
        if run["run_id"] in seen:
            raise ValueError(f"Duplicate candidate run {run['run_id']}")
        seen.add(run["run_id"])
        cutoff = parse_time(run["cutoff"])
        if cutoff.tzinfo is None or cutoff > latest_cutoff:
            raise ValueError(
                f"Candidate cutoff must be at least {RETENTION_DAYS} days old"
            )
    todo = [run for run in todo if run["run_id"] not in done][:limit]
    logging.info(
        "%s runs already recorded, %s candidates (%s TB) at %g runs per minute; progress is recorded in %s",
        f"{len(done):,}",
        f"{len(todo):,}",
        tb(sum(run["size_in_bytes"] for run in todo)),
        per_minute,
        done_path,
    )
    start = time.monotonic()
    count = freed = processed = skipped = missing = artifact_count = 0

    def delete_one(run: Dict[str, Any]) -> Tuple[str, int, int]:
        url = f"{GITHUB_API_URL}/repos/{repo}/actions/runs/{run['run_id']}"
        cutoff = parse_time(run["cutoff"])
        artifacts = size = 0

        def check_run() -> None:
            nonlocal artifacts, size
            artifacts = size = 0
            for page in itertools.count(1):
                batch = json.loads(
                    request(f"{url}/artifacts?per_page={PER_PAGE}&page={page}")
                )["artifacts"]
                for artifact in batch:
                    if parse_time(artifact["created_at"]) >= cutoff:
                        raise RunNotEligible("contains artifacts newer than the cutoff")
                    if not artifact["expired"]:
                        artifacts += 1
                        size += artifact["size_in_bytes"]
                if len(batch) < PER_PAGE:
                    break
            current = json.loads(request(url))
            if current["status"] != "completed":
                raise RunNotEligible("is not completed")
            if parse_time(current["updated_at"]) >= cutoff:
                raise RunNotEligible("was updated since the cutoff")

        try:
            request(url, "DELETE", before_attempt=check_run)
        except RunNotEligible as err:
            logging.info("Skipping run %s: %s", run["run_id"], err)
            return "skipped", 0, 0
        except HTTPError as err:
            if err.code != 404:
                raise
            logging.info("Run %s is already absent", run["run_id"])
            return "missing", 0, 0
        return "deleted", artifacts, size

    # Small chunks, each given its share of the minute, keep the rate steady and
    # leave little queued when an error or Ctrl-C stops the run.
    chunk = 5 * WORKERS
    next_chunk_at = time.monotonic()
    with ThreadPoolExecutor(WORKERS) as pool, done_path.open("a") as done_file:
        for chunk_start in range(0, len(todo), chunk):
            wait = next_chunk_at - time.monotonic()
            if wait > 0:
                left = len(todo) - processed
                pace = (next_chunk_at - start) / processed
                logging.info(
                    "Waiting %.1fs to stay at %g runs per minute, %s left (about %.1f h)...",
                    wait,
                    per_minute,
                    f"{left:,}",
                    pace * left / 3600,
                )
                time.sleep(wait)
            next_chunk_at = time.monotonic() + chunk * 60 / per_minute
            chunk_runs = todo[chunk_start : chunk_start + chunk]
            for run, (status, artifacts, size) in zip(
                chunk_runs, pool.map(delete_one, chunk_runs)
            ):
                processed += 1
                if status == "skipped":
                    skipped += 1
                else:
                    done_file.write(f"{run['run_id']}\n")
                    done_file.flush()
                    if status == "missing":
                        missing += 1
                    else:
                        count += 1
                        artifact_count += artifacts
                        freed += size
                if processed % LOG_EVERY_DELETES == 0:
                    elapsed = time.monotonic() - start
                    eta = elapsed / processed * (len(todo) - processed)
                    logging.info(
                        "%s/%s runs processed, %s deleted, %s skipped, %s TB removed, %.1f h elapsed, about %.1f h left",
                        f"{processed:,}",
                        f"{len(todo):,}",
                        f"{count:,}",
                        f"{skipped:,}",
                        tb(freed),
                        elapsed / 3600,
                        eta / 3600,
                    )
    print(
        f"Deleted {count:,} runs with {artifact_count:,} artifacts ({tb(freed)} TB) in "
        f"{(time.monotonic() - start) / 3600:.1f} h; {skipped:,} skipped, {missing:,} already absent; "
        f"{len(done) + count + missing:,} recorded in {done_path}"
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
                logging.info("No runs with old, unexpired artifacts remain")
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
        if count == 0:
            logging.info(
                "No runs deleted this cycle; stopping with remaining candidates skipped or already absent"
            )
            break
    print(f"Deleted {deleted:,} runs ({tb(freed)} TB) in {cycles} cycles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lister = sub.add_parser(
        "list",
        help="write runs holding artifacts older than the cutoff to a file, largest first",
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
        help="delete eligible runs, their artifacts and logs, resuming if interrupted",
    )
    deleter.add_argument("--repo", default="pytorch/executorch")
    deleter.add_argument("--candidates", type=Path, required=True)
    deleter.add_argument(
        "--per-minute",
        type=float,
        default=DEFAULT_PER_MINUTE,
        help="maximum candidate runs processed per minute",
    )
    deleter.add_argument(
        "--limit", type=int, help="process at most this many candidate runs"
    )
    runner = sub.add_parser(
        "auto",
        help="list and delete runs until no candidates remain or no runs can be deleted",
    )
    runner.add_argument("--repo", default="pytorch/executorch")
    runner.add_argument(
        "--older-than-days",
        type=int,
        default=RETENTION_DAYS,
        help=f"at least {RETENTION_DAYS}, the repository retention (default)",
    )
    runner.add_argument(
        "--per-minute",
        type=float,
        default=DEFAULT_PER_MINUTE,
        help="maximum candidate runs processed per minute",
    )
    args = parser.parse_args()
    if args.command != "delete" and args.older_than_days < RETENTION_DAYS:
        parser.error(f"--older-than-days must be at least {RETENTION_DAYS}")
    if args.command != "list" and not 0 < args.per_minute < float("inf"):
        parser.error("--per-minute must be positive and finite")
    if args.command == "delete" and args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
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
