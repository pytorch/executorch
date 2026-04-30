#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""Validate internal imports for backend example entrypoints.

Entrypoints are discovered from a backend-specific examples root by looking for
Python files that define a standard `if __name__ == "__main__"` block. The
workflow passes backend-specific paths and module prefixes, so the checker can
be reused without adding backend-specific logic here.
"""

import argparse
import ast
import importlib.util
import sys
from pathlib import Path


class ModuleResolver:
    def __init__(self, executorch_root: Path, extra_search_roots: list[Path]) -> None:
        self._executorch_root = executorch_root
        self._extra_search_roots = extra_search_roots
        self._local_module_names = self._discover_local_module_names()

    def _discover_local_module_names(self) -> set[str]:
        names = set()
        for search_root in self._extra_search_roots:
            for child in search_root.iterdir():
                if child.name == "__pycache__" or child.name.startswith("."):
                    continue
                if child.is_dir():
                    names.add(child.name)
                elif child.suffix == ".py":
                    names.add(child.stem)
        return names

    def is_internal_module(self, module_name: str) -> bool:
        if module_name.startswith("executorch."):
            return True
        return module_name.split(".", 1)[0] in self._local_module_names

    def _candidate_base_paths(self, module_name: str) -> list[Path]:
        if module_name.startswith("executorch."):
            return [self._executorch_root.joinpath(*module_name.split(".")[1:])]

        parts = module_name.split(".")
        return [
            search_root.joinpath(*parts) for search_root in self._extra_search_roots
        ]

    def module_source_file(self, module_name: str) -> Path | None:
        for base_path in self._candidate_base_paths(module_name):
            file_path = base_path.with_suffix(".py")
            if file_path.is_file():
                return file_path

            init_path = base_path / "__init__.py"
            if init_path.is_file():
                return init_path

        return None

    def module_exists(self, module_name: str) -> bool:
        if self.module_source_file(module_name) is not None:
            return True

        for base_path in self._candidate_base_paths(module_name):
            if base_path.is_dir():
                return True

        try:
            return importlib.util.find_spec(module_name) is not None
        except (AttributeError, ImportError, ValueError):
            return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        required=True,
        help="Display name for log messages, for example `QNN`.",
    )
    parser.add_argument(
        "--examples-root",
        required=True,
        help="Path to the examples root, relative to ExecuTorch root.",
    )
    parser.add_argument(
        "--module-prefix",
        required=True,
        help="Python module prefix for entrypoints under the examples root.",
    )
    parser.add_argument(
        "--module-search-root",
        action="append",
        default=[],
        help=(
            "Additional directories, relative to ExecuTorch root, that contain "
            "backend-local helper modules imported by examples."
        ),
    )
    parser.add_argument(
        "--skip-path-segment",
        action="append",
        default=["fb", "test", "tests"],
        help="Directory name to skip while discovering entrypoints.",
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


def normalize_module_prefix(module_prefix: str) -> str:
    return module_prefix[:-1] if module_prefix.endswith(".") else module_prefix


def should_skip_path(path: Path, skip_segments: list[str]) -> bool:
    if any(segment in path.parts for segment in skip_segments):
        return True

    stem = path.stem
    return any(
        stem == segment or stem.startswith(f"{segment}_") for segment in skip_segments
    )


def is_main_guard(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False

    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def is_entrypoint(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and is_main_guard(node.test):
            return True
    return False


def discover_entrypoints(
    examples_root: Path,
    skip_segments: list[str],
) -> list[str]:
    entrypoints = []
    for path in sorted(examples_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        if should_skip_path(path.relative_to(examples_root), skip_segments):
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if is_entrypoint(tree):
            entrypoints.append(path.relative_to(examples_root).as_posix())
    return entrypoints


def target_names(node: ast.AST) -> set[str]:
    names = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            names.update(target_names(element))
    return names


def resolve_from_module(
    module_name: str,
    node: ast.ImportFrom,
    *,
    is_package: bool = False,
) -> str:
    if node.level == 0:
        return node.module or ""

    package_name = module_name if is_package else module_name.rpartition(".")[0]
    relative_name = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative_name, package_name)


def collect_names_from_import_from(
    module_name: str,
    node: ast.ImportFrom,
    resolver: ModuleResolver,
    export_cache: dict[str, set[str]],
    *,
    is_package: bool,
) -> set[str]:
    names = set()
    try:
        imported_module = resolve_from_module(
            module_name,
            node,
            is_package=is_package,
        )
    except ImportError:
        imported_module = ""

    for alias in node.names:
        if alias.name == "*":
            if resolver.is_internal_module(imported_module):
                source_file = resolver.module_source_file(imported_module)
                if source_file is not None:
                    names.update(
                        exported_names(
                            imported_module,
                            source_file,
                            resolver,
                            export_cache,
                        )
                    )
            continue

        names.add(alias.asname or alias.name)

    return names


def nested_statement_bodies(node: ast.stmt) -> list[list[ast.stmt]] | None:
    if isinstance(node, ast.If):
        return [node.body, node.orelse]
    if isinstance(node, ast.Try):
        bodies = [node.body, node.orelse, node.finalbody]
        bodies.extend(handler.body for handler in node.handlers)
        return bodies
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
        return [node.body, getattr(node, "orelse", [])]
    if isinstance(node, ast.Match):
        return [case.body for case in node.cases]
    return None


def collect_names_from_node(
    module_name: str,
    node: ast.stmt,
    resolver: ModuleResolver,
    export_cache: dict[str, set[str]],
    *,
    is_package: bool,
) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return collect_names_from_import_from(
            module_name,
            node,
            resolver,
            export_cache,
            is_package=is_package,
        )
    if isinstance(node, ast.Assign):
        names = set()
        for target in node.targets:
            names.update(target_names(target))
        return names
    if isinstance(node, ast.AnnAssign):
        return target_names(node.target)

    bodies = nested_statement_bodies(node)
    if bodies is None:
        return set()

    names = set()
    for body in bodies:
        names.update(
            collect_exported_names(
                module_name,
                body,
                resolver,
                export_cache,
                is_package=is_package,
            )
        )
    return names


def collect_exported_names(
    module_name: str,
    body: list[ast.stmt],
    resolver: ModuleResolver,
    export_cache: dict[str, set[str]],
    *,
    is_package: bool,
) -> set[str]:
    names = set()
    for node in body:
        names.update(
            collect_names_from_node(
                module_name,
                node,
                resolver,
                export_cache,
                is_package=is_package,
            )
        )
    return names


def exported_names(
    module_name: str,
    source_file: Path,
    resolver: ModuleResolver,
    export_cache: dict[str, set[str]],
) -> set[str]:
    cached_names = export_cache.get(module_name)
    if cached_names is not None:
        return cached_names

    names: set[str] = set()
    export_cache[module_name] = names

    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    names.update(
        collect_exported_names(
            module_name,
            tree.body,
            resolver,
            export_cache,
            is_package=source_file.name == "__init__.py",
        )
    )
    return names


def validate_import_from(
    resolver: ModuleResolver,
    module_name: str,
    entrypoint: str,
    node: ast.ImportFrom,
    export_cache: dict[str, set[str]],
) -> tuple[list[str], int]:
    failures = []
    try:
        imported_module = resolve_from_module(module_name, node)
    except ImportError as error:
        failures.append(
            f"{entrypoint}:{node.lineno} relative import could not be resolved: {error}"
        )
        return failures, 0

    if not resolver.is_internal_module(imported_module):
        return failures, 0

    checks = 1
    if not resolver.module_exists(imported_module):
        failures.append(
            f"{entrypoint}:{node.lineno} missing internal module `{imported_module}`"
        )
        return failures, checks

    source_file = resolver.module_source_file(imported_module)
    exported = (
        exported_names(imported_module, source_file, resolver, export_cache)
        if source_file is not None
        else set()
    )

    for alias in node.names:
        if alias.name == "*":
            continue

        submodule_name = f"{imported_module}.{alias.name}"
        if resolver.module_exists(submodule_name):
            checks += 1
            continue

        if source_file is None or alias.name not in exported:
            failures.append(
                f"{entrypoint}:{node.lineno} unresolved internal import "
                f"`{alias.name}` from `{imported_module}`"
            )
        checks += 1

    return failures, checks


def validate_entrypoint(
    resolver: ModuleResolver,
    module_prefix: str,
    examples_root: Path,
    relative_path: str,
    export_cache: dict[str, set[str]],
) -> tuple[list[str], int]:
    entrypoint_path = examples_root / relative_path
    if not entrypoint_path.is_file():
        return [f"{relative_path}: entrypoint not found"], 0

    module_name = f"{module_prefix}.{Path(relative_path).with_suffix('').as_posix().replace('/', '.')}"
    tree = ast.parse(
        entrypoint_path.read_text(encoding="utf-8"), filename=str(entrypoint_path)
    )

    failures = []
    checks = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not resolver.is_internal_module(alias.name):
                    continue

                checks += 1
                if not resolver.module_exists(alias.name):
                    failures.append(
                        f"{relative_path}:{node.lineno} missing internal module `{alias.name}`"
                    )
        elif isinstance(node, ast.ImportFrom):
            import_failures, import_checks = validate_import_from(
                resolver,
                module_name,
                relative_path,
                node,
                export_cache,
            )
            failures.extend(import_failures)
            checks += import_checks

    return failures, checks


def main() -> None:
    if sys.version_info < (3, 10):
        print("Python 3.10+ is required to parse example sources")
        sys.exit(1)

    args = parse_args()
    executorch_root = resolve_executorch_root()
    examples_root = resolve_directory(executorch_root, args.examples_root)
    module_prefix = normalize_module_prefix(args.module_prefix)
    extra_search_roots = [
        resolve_directory(executorch_root, relative_path)
        for relative_path in args.module_search_root
    ]
    resolver = ModuleResolver(executorch_root, extra_search_roots)

    entrypoints = discover_entrypoints(examples_root, args.skip_path_segment)
    if not entrypoints:
        print(f"No {args.name} example entrypoints found under {examples_root}")
        sys.exit(1)

    all_failures = []
    total_checks = 0
    export_cache: dict[str, set[str]] = {}

    for relative_path in entrypoints:
        failures, checks = validate_entrypoint(
            resolver,
            module_prefix,
            examples_root,
            relative_path,
            export_cache,
        )
        all_failures.extend(failures)
        total_checks += checks

    if total_checks == 0:
        print(f"No {args.name} example imports were checked")
        sys.exit(1)

    if all_failures:
        print(
            f"{len(all_failures)} unresolved internal import(s) across "
            f"{len(entrypoints)} {args.name} example entrypoint(s):"
        )
        for failure in all_failures:
            print(f"  FAIL: {failure}")
        sys.exit(1)

    print(
        f"Validated {total_checks} internal import(s) across "
        f"{len(entrypoints)} {args.name} example entrypoint(s)"
    )


if __name__ == "__main__":
    main()
