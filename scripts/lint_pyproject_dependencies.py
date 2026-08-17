#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DIRECT_REFERENCE = re.compile(r"^[A-Za-z0-9_.-]+(?:\s*\[[^\]]+\])?\s*@\s*\S+")


@dataclass(frozen=True)
class DirectReferenceDependency:
    section: str
    dependency: str
    line: int


def _line_number_for_dependency(pyproject_text: str, dependency: str) -> int:
    quoted_dependency = f'"{dependency}"'
    for line_number, line in enumerate(pyproject_text.splitlines(), start=1):
        if quoted_dependency in line or dependency in line:
            return line_number
    return 1


def _record_if_direct_reference(
    *,
    pyproject_text: str,
    section: str,
    dependency: Any,
    violations: list[DirectReferenceDependency],
) -> None:
    if not isinstance(dependency, str):
        return
    normalized_dependency = dependency.strip()
    if _DIRECT_REFERENCE.match(normalized_dependency):
        violations.append(
            DirectReferenceDependency(
                section=section,
                dependency=dependency,
                line=_line_number_for_dependency(pyproject_text, dependency),
            )
        )


def find_direct_reference_dependencies(
    pyproject_path: Path,
) -> list[DirectReferenceDependency]:
    pyproject_text = pyproject_path.read_text()
    pyproject = tomllib.loads(pyproject_text)
    project = pyproject.get("project", {})

    violations: list[DirectReferenceDependency] = []
    for dependency in project.get("dependencies", []):
        _record_if_direct_reference(
            pyproject_text=pyproject_text,
            section="project.dependencies",
            dependency=dependency,
            violations=violations,
        )

    optional_dependencies = project.get("optional-dependencies", {})
    for extra, dependencies in optional_dependencies.items():
        for dependency in dependencies:
            _record_if_direct_reference(
                pyproject_text=pyproject_text,
                section=f"project.optional-dependencies.{extra}",
                dependency=dependency,
                violations=violations,
            )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reject direct URL dependencies in project metadata. PyPI rejects "
            "packages that publish dependencies like 'pkg @ git+https://...'."
        )
    )
    parser.add_argument(
        "pyproject",
        type=Path,
        nargs="?",
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml",
    )
    args = parser.parse_args()

    violations = find_direct_reference_dependencies(args.pyproject)
    for violation in violations:
        print(
            "::error "
            f"file={args.pyproject},line={violation.line},"
            "title=Direct URL dependency in pyproject.toml::"
            f"{violation.section} contains '{violation.dependency}'. "
            "PyPI rejects direct URL dependencies in published package "
            "metadata; move it to a requirements file or install script.",
            file=sys.stderr,
        )

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
