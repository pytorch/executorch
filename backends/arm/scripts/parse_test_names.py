# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import ast
import os

import pathlib
import re

import sys

from executorch.backends.arm.scripts.collect_testname_resources import (
    MODEL_LIST,
    OP_LIST,
    PASS_LIST,
    TARGETS,
)


ALLOWED_DIRNAMES = ("misc", "passes", "models", "quantizer", "ops")

SUFFIX_GROUP = r"(?:" + r"|".join(TARGETS) + r")"
DEFAULT_PATTERN = re.compile(
    rf"^test_.*_(?P<target>{SUFFIX_GROUP})(?:_.*)?$",
)
TARGET_EXTRACT_PATTERN = re.compile(rf"(?P<target>{SUFFIX_GROUP})(?:_.*)?$")
TARGET_CHOICES_DISPLAY = ", ".join(TARGETS)


class TestCollector(ast.NodeVisitor):
    def __init__(self, path: pathlib.Path, collected: list[str]):
        self.path = path
        self._collected = collected
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._record_test(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._record_test(node)
        self.generic_visit(node)

    def _record_test(self, node: ast.AST):
        name = getattr(node, "name", "")
        if name.startswith("test_"):
            self._collected.append(str(self.path) + "::" + name)


def collect_test_files(test_root: pathlib.Path):
    search_dirs = []
    for dirname in ALLOWED_DIRNAMES:
        dir_path = test_root / dirname
        if dir_path.is_dir():
            search_dirs.append(dir_path)
        else:
            print(f"warning: skipped missing directory {dir_path}", file=sys.stderr)

    file_paths: list[pathlib.Path] = []
    for dir_path in search_dirs:
        file_paths.extend(dir_path.rglob("test_*.py"))
    return sorted(file_paths)


def collect_tests(file_paths: list[pathlib.Path]) -> list[str]:
    tests: list[str] = []
    for file_path in file_paths:
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as error:
            print(f"warning: failed to read {file_path}: {error}", file=sys.stderr)
            continue

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as error:
            print(f"warning: failed to parse {file_path}: {error}", file=sys.stderr)
            continue

        TestCollector(file_path, tests).visit(tree)

    return tests


def _match_allowed_op_prefix(test_name: str) -> tuple[str, str, bool]:
    """
    Parses a test name on the form
        test_OP_TARGET_<not_delegated>_<any_other_info>
    where OP must match a key in op_name_map and TARGET one string in TARGETS. The
    "not_delegated" suffix indicates that the test tests that the op is not delegated.

    Examples of valid names: "test_mm_u55_INT_not_delegated" and
    "test_add_scalar_tosa_FP_two_inputs".

    Returns a tuple (OP, TARGET, IS_DELEGATED) if valid.
    """
    test_name = test_name.removeprefix("test_")
    is_delegated = "not_delegated" not in test_name

    op = "None"
    target = "None"
    for potential_target in TARGETS:
        index = test_name.find(potential_target)
        if index != -1:
            op = test_name[: index - 1]
            target = potential_target
            break
    # Special case for convolution
    op = op.removesuffix("_1d")
    op = op.removesuffix("_2d")
    op = op.removesuffix("_3d")

    # Remove suffix for 16 bit activation and 8 bit weight test cases
    op = op.removesuffix("_16a8w")

    return op, target, is_delegated


def _match_allowed_model_prefix(token: str, allowed_models: list[str]) -> str | None:
    for allowed_model in allowed_models:
        if token == allowed_model:
            return allowed_model
    return None


def _match_allowed_pass_prefix(token: str, allowed_passes: list[str]) -> str | None:
    for allowed_pass in allowed_passes:
        if token == allowed_pass:
            return allowed_pass
    return None


def _extract_target(name: str) -> str | None:
    match = TARGET_EXTRACT_PATTERN.search(name)
    if match and match.end() == len(name):
        return match.group("target")
    return None


def _parse_name_tokens(name: str) -> tuple[str, str]:
    rest = name[5:]
    target = _extract_target(name)
    token = rest.rstrip("_")
    if target:
        idx = rest.rfind(target)
        token = rest[:idx].rstrip("_")
    return token, target  # type: ignore[return-value]


def _describe_name_issue(kind: str, name: str) -> str:
    token, target = _parse_name_tokens(name)
    details: list[str] = []
    if token:
        details.append(f"{kind} token parsed as '{token}'\n    ")
    else:
        details.append(f"missing {kind} token before target\n    ")

    if target:
        details.append(f"target token parsed as '{target}'")
    else:
        details.append(
            f"missing target suffix (expected one of: {TARGET_CHOICES_DISPLAY}"
        )

    details.append(
        "\n    please follow the test naming convention: https://confluence.arm.com/display/MLENG/Executorch+naming+conventions"
    )

    return "".join(details)


def check_test_convention(  # noqa: C901
    tests: list[str],
    allowed_ops: list[str],
    allowed_models: list[str],
    allowed_passes: list[str],
) -> list[str]:
    violations: list[str] = []
    for test in tests:
        path = test.split("::")[0]
        name = test.split("::")[1]
        try:
            assert name[:5] == "test_", f"Unexpected input: {name}"
            if str(os.sep) + "ops" + str(os.sep) in path:
                matched_op, target, delegated = _match_allowed_op_prefix(name)
                if "reject" in name:
                    desc = _describe_name_issue("op", name)
                    violations.append(
                        f"{test}\n    expected test_OP_TARGET_*\n    Use 'not_delegated' instead of 'reject' in {name}\n"
                    )
                if target == "None" or matched_op not in allowed_ops:
                    desc = _describe_name_issue("op", name)
                    violations.append(
                        f"{test}\n    expected test_OP_TARGET_*\n    {desc}\n"
                    )
                    continue
            if str(os.sep) + "models" + str(os.sep) in path:
                token, target = _parse_name_tokens(name)
                if not target or not token:
                    desc = _describe_name_issue("model", name)
                    violations.append(
                        f"{test}\n    expected test_MODEL_TARGET_*\n    {desc}\n"
                    )
                    continue
                matched_model = _match_allowed_model_prefix(token, allowed_models)
                if matched_model is None:
                    desc = _describe_name_issue("model", name)
                    violations.append(
                        f"{test}\n    unknown model prefix '{token}'\n    expected test_MODEL_TARGET_* with MODEL in MODEL_LIST\n    {desc}\n"
                    )
                    continue
                continue
            if str(os.sep) + "passes" + str(os.sep) in path:
                token, target = _parse_name_tokens(name)
                if not target or not token:
                    desc = _describe_name_issue("pass", name)
                    violations.append(
                        f"{test}\n    expected test_PASS_TARGET_*\n    {desc}\n"
                    )
                    continue
                matched_pass = _match_allowed_pass_prefix(token, allowed_passes)
                if matched_pass is None:
                    desc = _describe_name_issue("pass", name)
                    violations.append(
                        f"{test}\n    unknown pass prefix '{token}'\n    expected test_PASS_TARGET_* with PASS in PASS_LIST\n    {desc}\n"
                    )
                    continue
                continue
            if (
                str(os.sep) + "quantizer" + str(os.sep) in path
                or str(os.sep) + "misc" + str(os.sep) in path
            ):
                token, target = _parse_name_tokens(name)
                if not target or not token:
                    desc = _describe_name_issue("name", name)
                    violations.append(
                        f"{test}\n    expected test_*_TARGET_*\n    {desc}\n"
                    )
                    continue
                continue
        except AssertionError as e:
            print(e)
    return violations


if __name__ == "__main__":
    """Parses a list of test names given on the commandline."""

    sys.tracebacklimit = 0  # Do not print stack trace
    tests: list[str] = []
    path_list: list[pathlib.Path] = [pathlib.Path(path) for path in sys.argv[1:]]

    if path_list == []:
        tests = collect_tests(collect_test_files(pathlib.Path("backends/arm/test")))
    else:
        tests = collect_tests(path_list)

    violations = check_test_convention(tests, OP_LIST, MODEL_LIST, PASS_LIST)

    for entry in violations:
        print(entry)

    print(f"Total tests needing renaming: {len(violations)}")

    if violations:
        sys.exit(1)
