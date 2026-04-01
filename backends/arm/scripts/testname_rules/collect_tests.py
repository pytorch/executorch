# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import ast
import logging
import pathlib


LOGGER = logging.getLogger(__name__)

ALLOWED_DIRNAMES = ("misc", "passes", "models", "quantizer", "ops")


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
            LOGGER.warning("skipped missing directory %s", dir_path)

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
            LOGGER.warning("failed to read %s: %s", file_path, error)
            continue

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as error:
            LOGGER.warning("failed to parse %s: %s", file_path, error)
            continue

        TestCollector(file_path, tests).visit(tree)

    return tests
