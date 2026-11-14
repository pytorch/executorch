#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write the owner of each ExecuTorch test file to the HUD ClickHouse ``owners``.

HUD's test and cost pages join test results with ``ci.owners`` on (repo, file)
to attribute them. A test file's owner is the module label in its
``# Owner(s): ["module: ..."]`` header, else the area of the most specific
CODEOWNERS entry that covers it, else its directory (two levels under
backends/, examples/ and extension/, one level elsewhere).

Without ClickHouse credentials in the environment this is a dry run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

GROUPED_DIRECTORIES = {"backends", "examples", "extension"}
EXCLUDED_DIRECTORIES = {"fb", "third-party", "third_party"}
OWNERS_HEADER = "# Owner(s): "


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py")
    )


def codeowners_areas(text: str) -> list[str]:
    """CODEOWNERS paths in file order; a later match overrides an earlier one."""
    areas = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            areas.append(line.split()[0])
    return areas


def matches(area: str, path: str) -> bool:
    if area.startswith("/"):
        prefix = area.strip("/")
        return path == prefix or path.startswith(prefix + "/")
    # An unanchored single-component pattern matches at any depth.
    return area.strip("/") in path.split("/")


def directory_owner(path: str) -> str:
    parts = path.split("/")
    if parts[0] in GROUPED_DIRECTORIES and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


def header_owners(source: str) -> list[str]:
    for line in source.splitlines():
        if line.startswith(OWNERS_HEADER):
            labels = json.loads(line.removeprefix(OWNERS_HEADER))
            return sorted(
                label.removeprefix("module: ")
                for label in labels
                if label.startswith("module: ")
            )
    return []


def owner_of(path: str, source: str, areas: list[str]) -> list[str]:
    if owners := header_owners(source):
        return owners
    matched = [area for area in areas if matches(area, path)]
    if matched:
        return [matched[-1].strip("/")]
    return [directory_owner(path)]


def collect_test_owners(repo: Path, paths: list[str]) -> dict[str, list[str]]:
    areas = codeowners_areas((repo / "CODEOWNERS").read_text(encoding="utf-8"))
    owners = {}
    for path in paths:
        if not is_test_file(path) or EXCLUDED_DIRECTORIES.intersection(
            path.split("/")
        ):
            continue
        source = (repo / path).read_text(encoding="utf-8", errors="replace")
        owners[path] = owner_of(path, source, areas)
    if not owners:
        raise ValueError(f"No test files found in {repo}")
    return owners


def get_owner_updates(
    repo: str,
    current_owners: dict[str, list[str]],
    stored_owners: dict[str, list[str]],
) -> list[tuple[str, str, list[str]]]:
    # Write only new or changed files, and clear deleted files with no owners.
    updates = []
    for path in sorted(current_owners.keys() | stored_owners.keys()):
        owners = current_owners.get(path, [])
        if path not in stored_owners or set(owners) != set(stored_owners[path]):
            updates.append((repo, path, owners))
    return updates


def main(argv: list[str] | None = None) -> None:
    endpoint = os.environ.get("CLICKHOUSE_ENDPOINT")
    username = os.environ.get("CLICKHOUSE_USERNAME")
    password = os.environ.get("CLICKHOUSE_PASSWORD")
    parser = argparse.ArgumentParser(description="Populate ClickHouse test owners")
    parser.add_argument("--repo", default="pytorch/executorch")
    parser.add_argument(
        "--database",
        required=bool(endpoint and username and password),
        help="Target ClickHouse database",
    )
    args = parser.parse_args(argv)

    root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
        .strip()
    )
    paths = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).split()
    current_owners = collect_test_owners(root, paths)

    if not (endpoint and username and password):
        print(
            f"Dry run: {len(current_owners)} test files found; "
            "database not updated (missing credentials).",
            file=sys.stderr,
        )
        return

    clickhouse_connect = importlib.import_module("clickhouse_connect")
    with clickhouse_connect.get_client(
        host=endpoint.removeprefix("https://").removesuffix(":8443"),
        username=username,
        password=password,
        database=args.database,
        secure=True,
        port=8443,
    ) as client:
        stored = client.query(
            "SELECT file, owners FROM owners FINAL WHERE repo = {repo:String}",
            parameters={"repo": args.repo},
        )
        updates = get_owner_updates(
            args.repo, current_owners, dict(stored.result_rows)
        )
        if updates:
            client.insert("owners", updates, column_names=["repo", "file", "owners"])
        print(
            f"Found {len(current_owners)} test files; "
            f"updated {len(updates)} files in {args.database}.owners",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
