#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fail CI when CUDA benchmark throughput regresses vs recent main history.

Compares the current run's decode throughput (``throughput_mean``) and
prefill throughput (``prefill_throughput_mean``) against the median of the
last few successful ``cuda.yml`` runs on ``main`` for the same
(model, quantization, GPU), using artifacts stored under
``s3://<bucket>/executorch-cuda-perf/<run_id>/<attempt>/``.

Any metric dropping more than ``--threshold-pct`` fails the job
(``exit 1`` with ``::error::`` annotations). Keys with fewer than
``--min-history`` baseline points are skipped (cold-start protection).

A PR carrying ``--bypass-label`` passes unconditionally with a ``::notice::``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from typing import Any, Iterable

logger: logging.Logger = logging.getLogger(__name__)

KNOWN_QUANTS = (
    "non-quantized",
    "quantized-int4-tile-packed",
    "quantized-int4-weight-only",
)

# Artifact dir prefixes, current and legacy (retired cuda-perf.yml) layout.
DIR_PREFIXES = ("cuda-bench-", "results-")


def split_artifact_dir(dirname: str) -> tuple[str, str] | None:
    """Split '<prefix><model_safe>-<quant>' into (model_safe, quant).

    Also accepts a nested path: walks up until a segment carries the prefix
    (e2e uploads results under a ``bench/`` subdir), so both
    ``cuda-bench-<model>-<quant>`` and ``cuda-bench-<model>-<quant>/bench``
    resolve.
    """
    parts = dirname.split(os.sep)
    for part in reversed(parts):
        for prefix in DIR_PREFIXES:
            if part.startswith(prefix):
                rest = part[len(prefix):]
                for quant in KNOWN_QUANTS:
                    if rest.endswith("-" + quant):
                        return rest[: -(len(quant) + 1)], quant
    return None


def load_current_results(results_dir: str) -> dict[tuple[str, str, str], dict[str, float]]:
    """Walk downloaded artifacts, return {(model, quant, gpu): means}."""
    current: dict[tuple[str, str, str], dict[str, float]] = {}
    for root, _dirs, files in os.walk(results_dir):
        if "benchmark_results.json" not in files:
            continue
        with open(os.path.join(root, "benchmark_results.json")) as f:
            data = json.load(f)
        means = {
            "decode": data.get("throughput_mean"),
            "prefill": data.get("prefill_throughput_mean"),
        }
        if means["decode"] is None or means["prefill"] is None:
            logger.warning("Skipping %s: missing throughput means", root)
            continue
        model, quant, gpu = identify_result(root, files, data)
        if model is None or quant is None or gpu is None:
            logger.warning("Skipping %s: cannot identify (model, quant, gpu)", root)
            continue
        current[(model, quant, gpu)] = means
    return current


def identify_result(
    root: str, files: list[str], data: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    """Resolve (model_safe, quant, gpu) preferring metadata.json, then v3, then dirname.

    Each field falls back independently: legacy metadata.json files (from
    the retired cuda-perf.yml) carry model/quantization but no gpu_name,
    so the v3 runners GPU must still apply even when model/quant are
    already known. Slash-normalization matches the metadata path so both
    produce the same key.
    """
    model: str | None = None
    quant: str | None = None
    gpu: str | None = None
    if "metadata.json" in files:
        with open(os.path.join(root, "metadata.json")) as f:
            meta = json.load(f)
        raw_model = meta.get("model")
        if raw_model:
            model = str(raw_model).replace("/", "_")
        quant = meta.get("quantization")
        gpu = meta.get("gpu_name")
    if "benchmark_results_v3.json" in files and (
        model is None or quant is None or gpu is None
    ):
        with open(os.path.join(root, "benchmark_results_v3.json")) as f:
            v3 = json.load(f)
        if v3:
            rec = v3[0]
            if model is None or quant is None:
                full = rec.get("model", {}).get("name", "")
                for known in KNOWN_QUANTS:
                    if full.endswith("_" + known):
                        model = (model or full[: -(len(known) + 1)]).replace("/", "_")
                        quant = quant or known
            if gpu is None:
                runners = rec.get("runners", [])
                gpu = runners[0].get("name") if runners else None
    if quant is None:
        # v1 model_name ("whisper-small_non-quantized") is org-less, so it
        # can only safely contribute the quant, never the model key.
        short = data.get("model_name", "")
        for known in KNOWN_QUANTS:
            if short.endswith("_" + known):
                quant = known
    if model is None or quant is None:
        split = split_artifact_dir(root)
        if split is not None:
            model = model or split[0]
            quant = quant or split[1]
    return model, quant, gpu


def github_api(url: str, token: str) -> Any:
    """GET a GitHub REST URL, return parsed JSON."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def list_successful_runs(repo: str, workflow: str, branch: str, token: str) -> list[int]:
    """Newest-first run ids of successful workflow runs on a branch."""
    runs: list[int] = []
    page = 1
    while len(runs) < 30 and page <= 3:
        url = (
            f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}"
            f"/runs?status=success&branch={branch}&per_page=30&page={page}"
        )
        payload = github_api(url, token)
        batch = payload.get("workflow_runs", [])
        if not batch:
            break
        runs.extend(r["id"] for r in batch)
        page += 1
    return runs


def has_bypass_label(repo: str, pr_number: str, label: str, token: str) -> bool:
    """Whether a PR carries the bypass label."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/labels?per_page=100"
    labels = github_api(url, token)
    return any(entry.get("name") == label for entry in labels)


def run_aws(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an aws CLI command, never through a shell."""
    return subprocess.run(
        ["aws", *args], capture_output=True, text=True, timeout=300
    )


def list_attempts(bucket: str, prefix: str, run_id: int) -> list[str]:
    """Attempt prefixes under s3://bucket/prefix/<run_id>/."""
    result = run_aws(["s3", "ls", f"s3://{bucket}/{prefix}/{run_id}/"])
    if result.returncode != 0:
        return []
    attempts = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "PRE":
            attempts.append(parts[1].rstrip("/"))
    return attempts


def fetch_history_points(
    bucket: str,
    prefix: str,
    run_id: int,
    keys: Iterable[tuple[str, str, str]],
    tmp_root: str,
) -> dict[tuple[str, str, str], dict[str, list[float]]]:
    """Pull one run's benchmark means from S3 into per-key point lists."""
    points: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for attempt in list_attempts(bucket, prefix, run_id):
        dest = os.path.join(tmp_root, str(run_id), attempt)
        result = run_aws(
            [
                "s3",
                "sync",
                f"s3://{bucket}/{prefix}/{run_id}/{attempt}/",
                dest,
                "--exclude",
                "*",
                "--include",
                "benchmark_results.json",
                "--include",
                "benchmark_results_v3.json",
                "--include",
                "metadata.json",
            ]
        )
        if result.returncode != 0:
            continue
        # First attempt with data wins for this run id.
        has_data = any(
            "benchmark_results.json" in files
            for _root, _dirs, files in os.walk(dest)
        )
        if has_data:
            for key, means in load_current_results(dest).items():
                if key in keys:
                    entry = points.setdefault(
                        key, {"decode": [], "prefill": []}
                    )
                    entry["decode"].append(means["decode"])
                    entry["prefill"].append(means["prefill"])
            break
    return points


def collect_baselines(args: argparse.Namespace, keys: set[tuple[str, str, str]]) -> dict:
    """Median of the last N successful main runs per key from S3."""
    run_ids = list_successful_runs(args.repo, args.workflow, args.branch, args.token)
    # Don't compare against ourselves when re-running the current run id.
    run_ids = [r for r in run_ids if str(r) != str(args.exclude_run_id)]
    per_key: dict[tuple[str, str, str], dict[str, list[float]]] = {
        key: {"decode": [], "prefill": []} for key in keys
    }
    with tempfile.TemporaryDirectory(prefix="cuda-regression-") as tmp_root:
        for run_id in run_ids:
            if all(len(v["decode"]) >= args.baseline_window for v in per_key.values()):
                break
            for key, vals in fetch_history_points(
                args.s3_bucket, args.s3_prefix, run_id, keys, tmp_root
            ).items():
                slot = per_key[key]
                if len(slot["decode"]) < args.baseline_window:
                    slot["decode"].extend(vals["decode"])
                    slot["prefill"].extend(vals["prefill"])
    baselines = {}
    for key, vals in per_key.items():
        if len(vals["decode"]) >= args.min_history:
            baselines[key] = {
                "decode": statistics.median(vals["decode"]),
                "prefill": statistics.median(vals["prefill"]),
                "n": len(vals["decode"]),
            }
    return baselines


def check_regressions(
    current: dict, baselines: dict, threshold_pct: float
) -> list[dict[str, Any]]:
    """Compare current means vs baselines, return per-key verdict rows."""
    rows = []
    for key, means in sorted(current.items()):
        base = baselines.get(key)
        if base is None:
            rows.append({"key": key, "status": "SKIP", "reason": "no baseline history"})
            continue
        deltas = {
            name: (means[name] - base[name]) / base[name] * 100.0
            for name in ("decode", "prefill")
        }
        failed = any(delta < -threshold_pct for delta in deltas.values())
        rows.append(
            {
                "key": key,
                "status": "FAIL" if failed else "PASS",
                "new_decode": means["decode"],
                "base_decode": base["decode"],
                "delta_decode": deltas["decode"],
                "new_prefill": means["prefill"],
                "base_prefill": base["prefill"],
                "delta_prefill": deltas["prefill"],
                "n": base["n"],
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]], threshold_pct: float) -> str:
    """Render verdict rows as a markdown table."""
    lines = [
        "## CUDA perf regression check",
        "",
        f"Fail threshold: either metric drops more than {threshold_pct:.1f}% "
        "vs the median of recent successful main runs (same model, quant, GPU).",
        "",
        "| model | quant | gpu | decode new / base (Δ%) | "
        "prefill new / base (Δ%) | history | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        model, quant, gpu = row["key"]
        if row["status"] == "SKIP":
            lines.append(
                f"| {model} | {quant} | {gpu} | — | — | 0 | SKIP ({row['reason']}) |"
            )
            continue
        lines.append(
            f"| {model} | {quant} | {gpu} | "
            f"{row['new_decode']:.1f} / {row['base_decode']:.1f} "
            f"({row['delta_decode']:+.1f}%) | "
            f"{row['new_prefill']:.1f} / {row['base_prefill']:.1f} "
            f"({row['delta_prefill']:+.1f}%) | "
            f"n={row['n']} | {row['status']} |"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--workflow", default="cuda.yml")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--s3-bucket", default="gha-artifacts")
    parser.add_argument("--s3-prefix", default="executorch-cuda-perf")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--pr-number", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--bypass-label", default="bypass-perf-regression")
    parser.add_argument("--exclude-run-id", default=os.environ.get("RUN_ID", ""))
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--baseline-window", type=int, default=5)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--summary-file", default=os.environ.get("GITHUB_STEP_SUMMARY", ""))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: bypass check, baseline compare, markdown report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    if args.pr_number and args.token:
        try:
            if has_bypass_label(args.repo, args.pr_number, args.bypass_label, args.token):
                print(
                    f"::notice::Perf regression check bypassed by "
                    f"'{args.bypass_label}' label on PR #{args.pr_number}"
                )
                return 0
        except Exception as e:
            logger.warning("Bypass label lookup failed, continuing with check: %s", e)

    current = load_current_results(args.results_dir)
    if not current:
        print("::warning::No current benchmark results found; skipping regression check")
        return 0

    baselines = collect_baselines(args, set(current))
    rows = check_regressions(current, baselines, args.threshold_pct)
    report = render_markdown(rows, args.threshold_pct)
    print(report)
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(report + "\n")

    failures = [r for r in rows if r["status"] == "FAIL"]
    for row in failures:
        model, quant, gpu = row["key"]
        print(
            f"::error::Perf regression in {model} [{quant}] on {gpu}: "
            f"decode {row['delta_decode']:+.1f}%, "
            f"prefill {row['delta_prefill']:+.1f}% "
            f"(threshold -{args.threshold_pct:.1f}%). "
            f"Add the '{args.bypass_label}' label to bypass with justification."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
