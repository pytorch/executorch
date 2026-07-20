#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bloaty binary-size reports for CI."""

import argparse
import csv
import io
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BLOATY_CONFIG = REPO_ROOT / "test" / "bloaty" / "executorch.bloaty"
BLOATY_CMD = shlex.split(os.environ.get("BLOATY", "bloaty"))

# Buckets considered "ExecuTorch source code" for the summary table. Everything
# else (stdlib, libc, startup, metadata, other) is shown separately.
EXECUTORCH_SOURCE_BUCKETS = [
    "runtime",
    "extension",
    "backends",
    "kernels",
    "cmsis_nn",
    "tokenizers",
    "flatbuffer",
]


def _run(cmd: List[str]) -> str:
    """Run a subprocess; on failure include stderr in the exception."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"command failed (exit {e.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{e.stderr}"
        ) from e


def run_bloaty(elf: Path, data_sources: str) -> List[Dict[str, object]]:
    # -n 0 defeats bloaty's default 20-row truncation. -s vm sorts by VM size
    # (bytes claimed in flash + RAM after load), which is what matters for
    # embedded targets — .bss claims RAM at runtime but has filesize 0.
    cmd = [
        *BLOATY_CMD,
        "-c",
        str(BLOATY_CONFIG),
        "-d",
        data_sources,
        "-n",
        "0",
        "--csv",
        "-s",
        "vm",
        str(elf),
    ]
    out = _run(cmd)
    reader = csv.DictReader(io.StringIO(out))
    rows: List[Dict[str, object]] = []
    for row in reader:
        parsed: Dict[str, object] = {}
        for k in reader.fieldnames or []:
            if k in ("vmsize", "filesize"):
                parsed[k] = int(row[k])
            else:
                parsed[k] = row[k]
        rows.append(parsed)
    return rows


def bloaty_text(
    elf: Path,
    data_sources: str,
    top_n: int,
    source_filter: Optional[str] = None,
) -> str:
    cmd = [
        *BLOATY_CMD,
        "-c",
        str(BLOATY_CONFIG),
        "-d",
        data_sources,
        "-n",
        str(top_n),
        "-s",
        "vm",
    ]
    if source_filter is not None:
        cmd += ["--source-filter", source_filter]
    cmd.append(str(elf))
    return _run(cmd)


def strip_copy(elf: Path, strip_tool: str) -> Path:
    stripped = elf.with_suffix(elf.suffix + ".stripped")
    _run([strip_tool, "-o", str(stripped), str(elf)])
    return stripped


@dataclass
class BinaryReport:
    job: str
    binary_name: str
    head_sha: str
    stripped_head: int
    segments_head: List[Dict[str, object]] = field(default_factory=list)
    sections_head: List[Dict[str, object]] = field(default_factory=list)
    groups_head: List[Dict[str, object]] = field(default_factory=list)
    groups_head_stripped: List[Dict[str, object]] = field(default_factory=list)
    symbols_head: List[Dict[str, object]] = field(default_factory=list)


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def render_table(rows: List[Dict[str, object]], key: str) -> str:
    if not rows:
        return "_(no data)_"
    out = ["| {} | vmsize | filesize |".format(key), "|---|---:|---:|"]
    for r in sorted(rows, key=lambda x: -int(x["vmsize"])):
        if r[key] == "TOTAL":
            continue
        out.append(f"| `{r[key]}` | {r['vmsize']:,} | {r['filesize']:,} |")
    return "\n".join(out)


def render_step_summary(
    report: BinaryReport, full_text: str, head_only_text: str
) -> str:
    et_rows = [
        r
        for r in report.groups_head
        if r.get("executorch") in EXECUTORCH_SOURCE_BUCKETS
    ]
    et_total = sum(int(r["vmsize"]) for r in et_rows)
    lines = [
        f"## Bloaty: `{report.job}` / `{report.binary_name}`",
        "",
        f"- head sha: `{report.head_sha}`",
        f"- stripped head vm size: **{report.stripped_head:,} bytes**",
        f"- ExecuTorch source total (unstripped, bucketed, vm): **{et_total:,} bytes**",
        "",
        "### Per-bucket sizes (unstripped, all buckets)",
        "",
        render_table(report.groups_head, "executorch"),
        "",
        "<details><summary>Full bloaty output</summary>",
        "",
        "```",
        full_text.rstrip(),
        "```",
        "",
        "</details>",
        "",
        "<details><summary>Top ExecuTorch source symbols</summary>",
        "",
        "```",
        head_only_text.rstrip(),
        "```",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def cmd_measure(args: argparse.Namespace) -> int:
    head = Path(args.head).resolve()
    if not head.exists():
        print(f"head ELF does not exist: {head}", file=sys.stderr)
        return 1

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stripped = strip_copy(head, args.strip_tool)
    try:
        groups_head_stripped = run_bloaty(stripped, "executorch")
    finally:
        stripped.unlink(missing_ok=True)
    # VM size of the stripped binary — flash + RAM bytes the loader claims.
    # .bss adds to vm but not file, so this differs from `ls -la` for any
    # binary with statically-allocated buffers.
    stripped_size = sum(
        int(r["vmsize"]) for r in groups_head_stripped if r.get("executorch") != "TOTAL"
    )

    segments_head = run_bloaty(head, "segments")
    sections_head = run_bloaty(head, "sections")
    groups_head = run_bloaty(head, "executorch")
    symbols_head = run_bloaty(head, "shortsymbols")

    report = BinaryReport(
        job=args.job,
        binary_name=args.binary_name,
        head_sha=args.head_sha,
        stripped_head=stripped_size,
        segments_head=segments_head,
        sections_head=sections_head,
        groups_head=groups_head,
        groups_head_stripped=groups_head_stripped,
        symbols_head=symbols_head,
    )

    atomic_write(out_dir / "metadata.json", json.dumps(asdict(report), indent=2))

    # executorch first → groups all symbols by bucket; sections then symbols
    # show what's inside each. Skipping `segments` (uninformative at this level).
    full_text = bloaty_text(head, "executorch,sections,shortsymbols", top_n=30)
    # Filter the head-only top-symbols dump to ExecuTorch source buckets only,
    # so stdlib / libc / startup / metadata / other don't crowd it out.
    head_only_text = bloaty_text(
        head,
        "executorch,shortsymbols",
        top_n=30,
        source_filter="|".join(EXECUTORCH_SOURCE_BUCKETS),
    )
    atomic_write(out_dir / "full.txt", full_text)
    atomic_write(out_dir / "head_only.txt", head_only_text)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(render_step_summary(report, full_text, head_only_text))

    print(f"wrote {out_dir / 'metadata.json'}")
    print(f"stripped head vm size: {stripped_size:,} bytes")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_measure = sub.add_parser("measure", help="Measure an ELF with bloaty")
    p_measure.add_argument(
        "--head", required=True, help="Path to head (unstripped) ELF"
    )
    p_measure.add_argument("--job", required=True, help="CI job identifier")
    p_measure.add_argument(
        "--binary-name", required=True, help="Binary name (e.g. size_test)"
    )
    p_measure.add_argument(
        "--head-sha", required=True, help="Git SHA of the head commit"
    )
    p_measure.add_argument(
        "--strip-tool", default="strip", help="Strip tool (e.g. arm-none-eabi-strip)"
    )
    p_measure.add_argument("--out", required=True, help="Output directory")
    p_measure.set_defaults(func=cmd_measure)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
