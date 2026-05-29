#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bloaty binary-size reports for CI.

  measure   Run bloaty against head (and optionally base) ELFs; write
            full.txt and metadata.json into --out.
  post      Read per-job artifacts in --in-dir; post (or update) a sticky
            PR comment.

Coupled to: .github/workflows/bloaty-size-comment.yml (caller of `post`)
            and the test-arm-cortex-m-size-test job in pull.yml (caller
            of `measure`).
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_utils import (  # noqa: E402
    gh_fetch_json_list,
    gh_fetch_url,
    gh_post_pr_comment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BLOATY_CONFIG = REPO_ROOT / "test" / "bloaty" / "executorch.bloaty"
# BLOATY env var can hold a space-separated command (e.g. "conda run -p /env bloaty").
BLOATY_CMD = os.environ.get("BLOATY", "bloaty").split()
COMMENT_MARKER = "<!-- executorch-ci-comment kind=bloaty-binary-size -->"
DELTA_NOISE_BYTES = 16
COMMENT_BODY_CAP = 60_000  # GitHub hard limit is 65536; leave headroom.
# Sticky-comment lookup trusts comments whose author login is in this set.
# Prevents marker collision if a user quotes COMMENT_MARKER in their own comment.
BOT_LOGINS = {"github-actions[bot]", "pytorch-bot[bot]", "facebook-github-bot"}


def run_bloaty(elf: Path, data_sources: str) -> List[Dict[str, int]]:
    """Parse `bloaty --csv -d <data_sources> -s file <elf>` into rows."""
    cmd = [
        *BLOATY_CMD,
        "-c",
        str(BLOATY_CONFIG),
        "-d",
        data_sources,
        "--csv",
        "-s",
        "file",
        str(elf),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    rows = []
    reader = csv.DictReader(io.StringIO(out))
    for row in reader:
        rows.append(
            {
                **{
                    k: row[k]
                    for k in reader.fieldnames
                    if k not in ("vmsize", "filesize")
                },
                "vmsize": int(row["vmsize"]),
                "filesize": int(row["filesize"]),
            }
        )
    return rows


def bloaty_text(
    elf: Path, base_elf: Optional[Path], data_sources: str, top_n: int
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
        "file",
        str(elf),
    ]
    if base_elf is not None:
        cmd += ["--", str(base_elf)]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def stripped_size(elf: Path, strip_tool: str) -> int:
    stripped = elf.with_suffix(elf.suffix + ".stripped")
    subprocess.run([strip_tool, "-o", str(stripped), str(elf)], check=True)
    try:
        return stripped.stat().st_size
    finally:
        stripped.unlink(missing_ok=True)


@dataclass
class BinaryReport:
    job: str
    binary_name: str
    head_sha: str
    base_sha: Optional[str]
    stripped_head: int
    stripped_base: Optional[int]
    sections_head: List[Dict[str, int]] = field(default_factory=list)
    sections_base: List[Dict[str, int]] = field(default_factory=list)
    groups_head: List[Dict[str, int]] = field(default_factory=list)
    groups_base: List[Dict[str, int]] = field(default_factory=list)
    segments_head: List[Dict[str, int]] = field(default_factory=list)
    segments_base: List[Dict[str, int]] = field(default_factory=list)
    symbols_head: List[Dict[str, int]] = field(default_factory=list)
    symbols_base: List[Dict[str, int]] = field(default_factory=list)
    base_build_failed: bool = False

    @property
    def stripped_delta(self) -> Optional[int]:
        if self.stripped_base is None:
            return None
        return self.stripped_head - self.stripped_base


def diff_rows(
    head: List[Dict[str, int]],
    base: List[Dict[str, int]],
    key: str,
) -> List[Tuple[str, int, int, int]]:
    """Return [(key, base_filesize, head_filesize, delta)] sorted by |delta| desc."""
    by_key = {r[key]: r["filesize"] for r in base}
    deltas = []
    for r in head:
        k = r[key]
        h = r["filesize"]
        b = by_key.pop(k, 0)
        deltas.append((k, b, h, h - b))
    for k, b in by_key.items():
        deltas.append((k, b, 0, -b))
    deltas.sort(key=lambda t: abs(t[3]), reverse=True)
    return deltas


def fmt_delta(d: int) -> str:
    return "0" if d == 0 else f"{d:+d}"


def render_table(header: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def render_report_section(r: BinaryReport) -> str:
    parts = []
    title = f"{r.job}: {r.binary_name}"
    if r.stripped_delta is not None:
        title += f" {fmt_delta(r.stripped_delta)} bytes"
    if r.base_build_failed:
        title += " (base build failed, head-only)"
    parts.append(f"<details><summary>{title}</summary>\n")

    def section(
        label: str, key: str, head: List[Dict[str, int]], base: List[Dict[str, int]]
    ) -> str:
        if not head:
            return ""
        if base:
            rows = []
            for k, b, h, d in diff_rows(head, base, key):
                if abs(d) < DELTA_NOISE_BYTES:
                    continue
                rows.append([k, str(b), str(h), fmt_delta(d)])
            if not rows:
                return f"\n{label} (Δ ≥ {DELTA_NOISE_BYTES}): _no rows above noise threshold_\n"
            return (
                f"\n{label} (Δ ≥ {DELTA_NOISE_BYTES} bytes):\n"
                + render_table([key, "base", "pr", "Δ"], rows)
                + "\n"
            )
        rows = [[r[key], str(r["filesize"])] for r in head]
        return f"\n{label}:\n" + render_table([key, "size"], rows) + "\n"

    if r.symbols_head and r.symbols_base:
        sym_deltas = diff_rows(r.symbols_head, r.symbols_base, "shortsymbols")
        top = [
            (k, b, h, d) for k, b, h, d in sym_deltas if abs(d) >= DELTA_NOISE_BYTES
        ][:5]
        if top:
            sym_rows = [[k, str(b), str(h), fmt_delta(d)] for k, b, h, d in top]
            parts.append(
                f"\nTop symbols by Δ (≥ {DELTA_NOISE_BYTES} bytes):\n"
                + render_table(["symbol", "base", "pr", "Δ"], sym_rows)
                + "\n"
            )

    parts.append(section("By segment", "segments", r.segments_head, r.segments_base))
    parts.append(section("By section", "sections", r.sections_head, r.sections_base))
    parts.append(section("By group", "executorch", r.groups_head, r.groups_base))
    parts.append(
        "\nFull per-symbol report: see `full.txt` in this run's workflow artifacts.\n"
    )
    parts.append("</details>\n")
    return "\n".join(p for p in parts if p)


def render_comment(
    reports: List[BinaryReport], run_url: Optional[str]
) -> Optional[str]:
    """Render the sticky comment body. Returns None when every binary is below the noise floor."""
    significant = any(
        r.stripped_delta is None or abs(r.stripped_delta) >= DELTA_NOISE_BYTES
        for r in reports
    )
    if not significant:
        return None

    lines = [COMMENT_MARKER, "## Binary size report", ""]
    head_sha = reports[0].head_sha if reports else ""
    base_sha = reports[0].base_sha if reports else ""
    header = f"Base: `{base_sha[:7] if base_sha else 'n/a'}` · Head: `{head_sha[:7]}`"
    if run_url:
        header += f" · [Full reports (workflow artifacts)]({run_url})"
    lines += [header, ""]

    summary_rows = []
    for r in reports:
        d = "n/a" if r.stripped_delta is None else fmt_delta(r.stripped_delta)
        base_s = "n/a" if r.stripped_base is None else str(r.stripped_base)
        summary_rows.append([r.job, r.binary_name, base_s, str(r.stripped_head), d])
    lines.append(
        render_table(
            ["Job", "Binary", "Base (stripped)", "PR (stripped)", "Δ"],
            summary_rows,
        )
    )
    lines.append("")

    for r in reports:
        if r.stripped_delta is not None and abs(r.stripped_delta) < DELTA_NOISE_BYTES:
            continue
        lines.append(render_report_section(r))

    body = "\n".join(lines)
    if len(body) > COMMENT_BODY_CAP:
        cut = body.rfind("\n\n", 0, COMMENT_BODY_CAP - 200)
        if cut <= 0:
            cut = COMMENT_BODY_CAP - 200
        body = (
            body[:cut] + "\n\n_…truncated. See workflow artifacts for full report._\n"
        )
    return body


def render_resolved_body(head_sha: str) -> str:
    """Body for editing the sticky comment when the PR's delta returns under the noise floor."""
    return (
        f"{COMMENT_MARKER}\n"
        "## Binary size report\n\n"
        f"Head: `{head_sha[:7]}` — binary size is within ±{DELTA_NOISE_BYTES} bytes of the merge base."
    )


def cmd_measure(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = Path(args.head)
    base = Path(args.base) if args.base else None
    base_build_failed = bool(args.base_build_failed)

    if not head.exists():
        print(f"head ELF not found: {head}", file=sys.stderr)
        return 1
    if base is not None and not base.exists():
        base = None
        base_build_failed = True

    strip_tool = args.strip_tool or "strip"
    report = BinaryReport(
        job=args.job,
        binary_name=args.binary_name,
        head_sha=args.head_sha,
        base_sha=args.base_sha,
        stripped_head=stripped_size(head, strip_tool),
        stripped_base=stripped_size(base, strip_tool) if base is not None else None,
        base_build_failed=base_build_failed,
    )
    report.segments_head = run_bloaty(head, "segments")
    report.sections_head = run_bloaty(head, "sections")
    report.groups_head = run_bloaty(head, "executorch")
    report.symbols_head = run_bloaty(head, "shortsymbols")
    if base is not None:
        report.segments_base = run_bloaty(base, "segments")
        report.sections_base = run_bloaty(base, "sections")
        report.groups_base = run_bloaty(base, "executorch")
        report.symbols_base = run_bloaty(base, "shortsymbols")

    (out_dir / "full.txt").write_text(
        bloaty_text(head, base, "sections,executorch,shortsymbols", top_n=100)
    )

    # Atomic write so a partial failure can't ship a half-written metadata.json.
    metadata = {
        "job": report.job,
        "binary_name": report.binary_name,
        "head_sha": report.head_sha,
        "base_sha": report.base_sha,
        "stripped_head": report.stripped_head,
        "stripped_base": report.stripped_base,
        "base_build_failed": report.base_build_failed,
        "segments_head": report.segments_head,
        "segments_base": report.segments_base,
        "sections_head": report.sections_head,
        "sections_base": report.sections_base,
        "groups_head": report.groups_head,
        "groups_base": report.groups_base,
        "symbols_head": report.symbols_head,
        "symbols_base": report.symbols_base,
    }
    tmp = out_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2))
    tmp.replace(out_dir / "metadata.json")
    delta_str = (
        fmt_delta(report.stripped_delta) if report.stripped_delta is not None else "n/a"
    )
    print(f"wrote {out_dir}/metadata.json (Δ stripped: {delta_str})")
    return 0


def load_reports(in_dir: Path) -> List[BinaryReport]:
    reports = []
    for meta_path in sorted(in_dir.glob("*/metadata.json")):
        m = json.loads(meta_path.read_text())
        reports.append(
            BinaryReport(
                job=m["job"],
                binary_name=m["binary_name"],
                head_sha=m["head_sha"],
                base_sha=m.get("base_sha"),
                stripped_head=m["stripped_head"],
                stripped_base=m.get("stripped_base"),
                base_build_failed=m.get("base_build_failed", False),
                segments_head=m.get("segments_head", []),
                segments_base=m.get("segments_base", []),
                sections_head=m.get("sections_head", []),
                sections_base=m.get("sections_base", []),
                groups_head=m.get("groups_head", []),
                groups_base=m.get("groups_base", []),
                symbols_head=m.get("symbols_head", []),
                symbols_base=m.get("symbols_base", []),
            )
        )
    return reports


def find_sticky_comment_id(org: str, repo: str, pr_num: int) -> Optional[int]:
    url = f"https://api.github.com/repos/{org}/{repo}/issues/{pr_num}/comments"
    # Hard cap pagination: PRs with >5000 comments are not a realistic case for size reports.
    for page in range(1, 51):
        comments = gh_fetch_json_list(url, params={"per_page": 100, "page": page})
        if not comments:
            return None
        for c in comments:
            author = c.get("user", {}).get("login", "")
            if author in BOT_LOGINS and c.get("body", "").startswith(COMMENT_MARKER):
                return c["id"]
        if len(comments) < 100:
            return None
    return None


def patch_comment(org: str, repo: str, comment_id: int, body: str) -> None:
    url = f"https://api.github.com/repos/{org}/{repo}/issues/comments/{comment_id}"
    gh_fetch_url(url, data={"body": body}, method="PATCH", reader=lambda x: x.read())


def cmd_post(args: argparse.Namespace) -> int:
    in_dir = Path(args.in_dir)
    reports = load_reports(in_dir)
    if not reports:
        print(
            f"no metadata.json files under {in_dir}; nothing to post", file=sys.stderr
        )
        return 0

    pr_num = args.pr_num
    if pr_num is None:
        for p in sorted(in_dir.glob("*/pr_number.txt")):
            pr_num = int(p.read_text().strip())
            break
    if pr_num is None:
        print(
            "no PR number (use --pr-num or include pr_number.txt in an artifact)",
            file=sys.stderr,
        )
        return 1

    body = render_comment(reports, args.run_url)

    if args.dry_run:
        print(body if body is not None else "(no comment — within noise floor)")
        return 0

    existing_id = find_sticky_comment_id(args.org, args.repo, pr_num)

    if body is None:
        # Within noise floor. Edit any stale sticky to say so — leaving a +200 comment
        # in place after the PR was fixed would mislead reviewers.
        if existing_id is None:
            print(
                f"no binary exceeds Δ={DELTA_NOISE_BYTES} bytes; no existing sticky to clear"
            )
            return 0
        head_sha = reports[0].head_sha if reports else ""
        patch_comment(args.org, args.repo, existing_id, render_resolved_body(head_sha))
        print(f"cleared stale sticky {existing_id} on PR {pr_num}")
        return 0

    if existing_id:
        patch_comment(args.org, args.repo, existing_id, body)
        print(f"updated comment {existing_id} on PR {pr_num}")
    else:
        gh_post_pr_comment(args.org, args.repo, pr_num, body)
        print(f"posted new comment on PR {pr_num}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("measure")
    m.add_argument("--head", required=True)
    m.add_argument("--base", help="base (merge-base) ELF; omit for head-only")
    m.add_argument("--job", required=True)
    m.add_argument("--binary-name", required=True)
    m.add_argument("--head-sha", default=os.environ.get("GITHUB_SHA", ""))
    m.add_argument("--base-sha", default="")
    m.add_argument("--out", required=True)
    m.add_argument("--strip-tool", help="e.g. arm-none-eabi-strip")
    m.add_argument("--base-build-failed", action="store_true")
    m.set_defaults(func=cmd_measure)

    po = sub.add_parser("post")
    po.add_argument("--in-dir", required=True)
    po.add_argument("--org", default="pytorch")
    po.add_argument("--repo", default="executorch")
    po.add_argument("--pr-num", type=int)
    po.add_argument("--run-url")
    po.add_argument("--dry-run", action="store_true")
    po.set_defaults(func=cmd_post)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
