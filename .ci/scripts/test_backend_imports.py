#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""Validate that backend Python modules can be imported.

The workflow passes backend-specific paths and package prefixes so the same
checker can be reused for different backends.
"""

import argparse
import importlib
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        required=True,
        help="Display name for log messages, for example `QNN`.",
    )
    parser.add_argument(
        "--package-root",
        required=True,
        help="Path to the backend package root, relative to ExecuTorch root.",
    )
    parser.add_argument(
        "--package-prefix",
        required=True,
        help="Python package prefix, for example `executorch.backends.qualcomm`.",
    )
    parser.add_argument(
        "--skip-segment",
        action="append",
        default=["fb", "test", "tests"],
        help="Package path segment to skip while walking modules.",
    )
    return parser.parse_args()


def resolve_executorch_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "backends").is_dir() and (parent / "examples").is_dir():
            return parent
    raise RuntimeError(
        f"Could not locate ExecuTorch root from {Path(__file__).resolve()}"
    )


def resolve_directory(executorch_root: Path, relative_path: str) -> Path:
    directory = executorch_root / relative_path
    if not directory.is_dir():
        raise RuntimeError(
            f"Directory `{relative_path}` was not found under {executorch_root}"
        )
    return directory


def normalize_package_prefix(package_prefix: str) -> str:
    return package_prefix[:-1] if package_prefix.endswith(".") else package_prefix


def should_skip_path(path: Path, skip_segments: list[str]) -> bool:
    if any(segment in path.parts for segment in skip_segments):
        return True

    stem = path.stem
    return any(
        stem == segment or stem.startswith(f"{segment}_") for segment in skip_segments
    )


def discover_modules(
    package_root: Path,
    package_prefix: str,
    skip_segments: list[str],
) -> list[str]:
    modules = []
    for path in sorted(package_root.rglob("*.py")):
        relative_path = path.relative_to(package_root)
        if should_skip_path(relative_path, skip_segments):
            continue

        if relative_path.name == "__init__.py":
            module_suffix = ".".join(relative_path.parent.parts)
            if module_suffix:
                modules.append(f"{package_prefix}.{module_suffix}")
            else:
                modules.append(package_prefix)
            continue

        modules.append(
            f"{package_prefix}.{'.'.join(relative_path.with_suffix('').parts)}"
        )
    return modules


def main() -> None:
    args = parse_args()
    executorch_root = resolve_executorch_root()
    package_root = resolve_directory(executorch_root, args.package_root)
    package_prefix = normalize_package_prefix(args.package_prefix)

    failures: list[tuple[str, str, str]] = []
    modules = discover_modules(package_root, package_prefix, args.skip_segment)
    total_modules = len(modules)
    if total_modules == 0:
        print(f"No {args.name} Python modules found under {package_root}")
        sys.exit(1)

    for index, name in enumerate(modules, 1):
        print(f"[{index}/{total_modules}] importing {name}", flush=True)
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append((name, type(error).__name__, str(error)))

    if failures:
        print(f"{len(failures)}/{total_modules} {args.name} import failure(s):")
        for name, error_type, message in failures:
            print(f"  FAIL: {name} -- {error_type}: {message}")
        sys.exit(1)

    print(f"All {total_modules} {args.name} modules imported successfully")


if __name__ == "__main__":
    main()
