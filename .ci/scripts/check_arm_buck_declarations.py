#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Check regressions in Arm test-library Buck declarations, without executing Buck.

This is a static declaration lint, not a replacement for Meta's packaged tests.
It models literal Python library declarations and internal (is_oss=False) branches.
Unsupported newly relevant declarations fail as 'cannot validate'. Existing
violations are compared by identity, not line number, against an explicit base.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import functools
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SCOPE = "backends/arm/test"
TEST_MACRO = SCOPE + "/targets.bzl"


class Unsupported(ValueError):
    pass


class Snapshot:
    def __init__(self, files: dict[str, str]):
        self.paths = set(files)
        self._files = files

    def read(self, path: str) -> str:
        return self._files[path]


class GitSnapshot(Snapshot):
    def __init__(self, repo: Path, revision: str | None):
        super().__init__({})
        self.repo = repo
        self.revision = revision
        self._files: dict[str, str] = {}
        if revision is None:
            command = ["ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        else:
            revision = self.git("rev-parse", "--verify", revision + "^{commit}").strip()
            self.revision = revision
            command = ["ls-tree", "-r", "--name-only", "-z", revision]
        self.paths = set(self.git(*command).split("\0")) - {""}
        if revision is None:
            self.paths = {p for p in self.paths if (repo / p).is_file()}

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def read(self, path: str) -> str:
        if path not in self.paths:
            raise KeyError(path)
        if path not in self._files:
            self._files[path] = (
                self.git("show", f"{self.revision}:{path}")
                if self.revision
                else (self.repo / path).read_text(encoding="utf-8")
            )
        return self._files[path]


@functools.lru_cache(maxsize=None)
def glob_match(path: str, pattern: str) -> bool:
    """Buck-style path glob: '*' does not cross '/', '**' can."""
    parts, patterns = tuple(path.split("/")), tuple(pattern.split("/"))

    @functools.lru_cache(maxsize=None)
    def match(i: int, j: int) -> bool:
        if j == len(patterns):
            return i == len(parts)
        if patterns[j] == "**":
            return match(i, j + 1) or (i < len(parts) and match(i + 1, j))
        return (
            i < len(parts)
            and fnmatch.fnmatchcase(parts[i], patterns[j])
            and match(i + 1, j + 1)
        )

    return match(0, 0)


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


def label_for(package: str, name: str) -> str:
    return "//executorch/" + package + ":" + name


def is_scoped_label(label: str) -> bool:
    return label.startswith((label_for(SCOPE, ""), "//executorch/" + SCOPE + "/"))


def normalize(label: str, package: str) -> str | None:
    if label.startswith(":"):
        return label_for(package, label[1:])
    if label.startswith("//executorch/"):
        return label
    # Dependencies outside the in-repo executorch namespace are out of scope.
    return None


@dataclass(frozen=True, order=True)
class Issue:
    # Messages may gain provider suggestions without creating a new violation.
    kind: str = field(compare=False)
    owner: str
    source: str
    detail: str = field(compare=False)
    subject: str = ""
    category: str = "declaration"

    def __str__(self) -> str:
        return f"{self.kind}: {self.owner}: {self.source}: {self.detail}"


@dataclass
class Rule:
    label: str
    kind: str
    declaration_file: str
    sources: set[str] = field(default_factory=set)
    deps: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)


class Declarations:
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.rules: dict[str, Rule] = {}
        self.loaded: set[str] = set()
        self.package_errors: dict[str, list[tuple[str, str]]] = {}

    def package_error(self, package: str, filename: str, error: str) -> None:
        self.package_errors.setdefault(package, []).append((filename, error))

    def load_scope(self) -> None:
        self.load_package(SCOPE)
        for filename in sorted(self.snapshot.paths):
            package, _, name = filename.rpartition("/")
            if name in {"BUCK", "TARGETS"} and (
                package == SCOPE or package.startswith(SCOPE + "/")
            ):
                self.load_package(package)

    def package_files(self, package: str) -> list[str]:
        prefix = package + "/" if package else ""
        result = []
        for path in self.snapshot.paths:
            if not path.startswith(prefix):
                continue
            relative = path[len(prefix) :]
            # A glob cannot reach into another Buck package.
            parents = list(PurePosixPath(relative).parents)[:-1]
            if any(
                prefix + str(p) + "/BUCK" in self.snapshot.paths
                or prefix + str(p) + "/TARGETS" in self.snapshot.paths
                for p in parents
            ):
                continue
            result.append(relative)
        return sorted(result)

    # Each branch corresponds to one explicitly supported AST form.
    # pylint: disable-next=too-many-return-statements
    def value(self, node: ast.AST, env: dict[str, object], package: str):  # noqa: C901
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.value(x, env, package) for x in node.elts]
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if dotted(node) == "runtime.is_oss":
            return env.get("__is_oss__", False)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self.value(node.operand, env, package)
        if isinstance(node, ast.BoolOp):
            for operand in node.values:
                value = self.value(operand, env, package)
                if bool(value) == isinstance(node.op, ast.Or):
                    return value
            return value
        if isinstance(node, ast.IfExp):
            branch = node.body if self.value(node.test, env, package) else node.orelse
            return self.value(branch, env, package)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.value(node.left, env, package) + self.value(
                node.right, env, package
            )
        if isinstance(node, ast.Call) and dotted(node.func) in {"glob", "native.glob"}:
            patterns = self.value(node.args[0], env, package)
            kwargs = {k.arg: self.value(k.value, env, package) for k in node.keywords}
            unknown = set(kwargs) - {"exclude", "exclude_directories"}
            if unknown:
                raise Unsupported(
                    "unsupported glob arguments: " + ", ".join(sorted(unknown))
                )
            excludes = kwargs.get("exclude", [])
            return [
                p
                for p in self.package_files(package)
                if any(glob_match(p, pattern) for pattern in patterns)
                and not any(glob_match(p, pattern) for pattern in excludes)
            ]
        raise Unsupported(ast.unparse(node))

    def strings(self, node: ast.AST, env: dict[str, object], package: str) -> list[str]:
        value = self.value(node, env, package)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise Unsupported("expected a static list of strings: " + ast.unparse(node))
        return value

    def parse_rule(  # noqa: C901
        self, call: ast.Call, env: dict[str, object], package: str, filename: str
    ) -> None:
        kwargs = {k.arg: k.value for k in call.keywords}
        function = dotted(call.func)
        if function == "non_fbcode_target":
            return
        kind = dotted(kwargs.get("_kind", call.func)).split(".")[-1]
        if kind not in {
            "python_library",
            "python_binary",
            "python_test",
            "python_pytest",
            "python_unittest",
            "cxx_library",
            "cxx_python_extension",
            "cxx_test",
            "export_file",
            "filegroup",
        }:
            if "name" in kwargs and "srcs" in kwargs:
                self.package_error(package, filename, "unsupported rule kind: " + kind)
            return
        try:
            name = self.value(kwargs["name"], env, package)
            if not isinstance(name, str):
                raise Unsupported("non-string target name")
        except (KeyError, Unsupported) as error:
            self.package_error(package, filename, f"target name: {error}")
            return
        rule = Rule(label_for(package, name), kind, filename)
        for attribute in ("srcs", "deps", "exported_deps", "fbcode_deps"):
            if attribute not in kwargs:
                continue
            try:
                values = self.strings(kwargs[attribute], env, package)
                if attribute == "srcs":
                    for source in values:
                        if source.startswith((":", "//", "@")):
                            if kind == "python_library":
                                raise Unsupported("source target reference " + source)
                        elif source.endswith(".py"):
                            rule.sources.add(package + "/" + source)
                else:
                    rule.deps.update(
                        target
                        for value in values
                        if (target := normalize(value, package)) is not None
                    )
            except Unsupported as error:
                rule.errors.append(f"{attribute}: {error}")
        if rule.label in self.rules:
            rule.errors.append("multiple declarations for the same target")
        self.rules[rule.label] = rule

    def load_package(self, package: str) -> None:
        if package in self.loaded:
            return
        self.loaded.add(package)
        filename = next(
            (
                package + "/" + name
                for name in ("BUCK", "TARGETS")
                if package + "/" + name in self.snapshot.paths
            ),
            None,
        )
        if filename is None:
            self.package_error(package, package + "/BUCK", "no in-repo build file")
            return
        self.parse_file(filename, package, set())

    # Keep declaration traversal and its environment in a single scope.
    # pylint: disable-next=too-many-locals,too-many-statements
    def parse_file(  # noqa: C901
        self, filename: str, package: str, seen: set, selected: set[str] | None = None
    ) -> None:
        key = (filename, tuple(sorted(selected or [])))
        if key in seen:
            return
        seen.add(key)
        try:
            tree = ast.parse(self.snapshot.read(filename), filename=filename)
        except (KeyError, SyntaxError) as error:
            self.package_error(package, filename, f"cannot read {filename}: {error}")
            return
        env: dict[str, object] = {}
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        imports: dict[str, tuple[str, str]] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and dotted(node.value.func) == "load"
            ):
                args = node.value.args
                if (
                    args
                    and isinstance(args[0], ast.Constant)
                    and isinstance(args[0].value, str)
                ):
                    source = args[0].value
                    # Only local declaration helpers; never load external macros.
                    if source.startswith(":") or not source.startswith(("@", "//")):
                        helper = str(
                            PurePosixPath(filename).parent / source.lstrip(":")
                        )
                        if helper in self.snapshot.paths:
                            imports.update(
                                (arg.value, (helper, arg.value))
                                for arg in args[1:]
                                if isinstance(arg, ast.Constant)
                                and isinstance(arg.value, str)
                            )
                            imports.update(
                                (k.arg, (helper, k.value.value))
                                for k in node.value.keywords
                                if isinstance(k.value, ast.Constant)
                                and isinstance(k.value.value, str)
                            )

        called: set[str] = set()
        visiting: set[str] = set()

        def call_name(call: ast.Call) -> str:
            return dotted(
                next((k.value for k in call.keywords if k.arg == "_kind"), call.func)
            )

        def visit_helper(name: str) -> None:
            function = functions.get(name)
            if function is None:
                return
            args = function.args
            if (
                args.posonlyargs
                or args.args
                or args.kwonlyargs
                or args.vararg
                or args.kwarg
            ):
                self.package_error(
                    package, filename, "parameterized declaration helper: " + name
                )
            elif name in visiting:
                self.package_error(
                    package, filename, "recursive declaration helper: " + name
                )
            else:
                visiting.add(name)
                visit(function.body, dict(env))
                visiting.remove(name)

        def visit(statements: list[ast.stmt], local_env: dict[str, object]) -> None:
            for node in statements:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    try:
                        local_env[node.targets[0].id] = self.value(
                            node.value, local_env, package
                        )
                    except Unsupported:
                        local_env.pop(node.targets[0].id, None)
                elif isinstance(node, ast.If):
                    try:
                        branch = (
                            node.body
                            if self.value(node.test, local_env, package)
                            else node.orelse
                        )
                        visit(branch, local_env)
                    except Unsupported as error:
                        if any(
                            isinstance(x, ast.Call)
                            and dotted(x.func) != "non_fbcode_target"
                            and (
                                call_name(x) in functions
                                or call_name(x) in imports
                                or call_name(x).split(".")[-1] == "python_library"
                            )
                            for x in ast.walk(node)
                        ):
                            self.package_error(
                                package, filename, f"conditional declaration: {error}"
                            )
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if dotted(call.func) == "non_fbcode_target":
                        continue
                    name = call_name(call)
                    if name in functions or name in imports:
                        if call.args or any(k.arg != "_kind" for k in call.keywords):
                            self.package_error(
                                package,
                                filename,
                                "parameterized declaration helper: " + name,
                            )
                        elif name in functions:
                            visit_helper(name)
                        else:
                            called.add(name)
                    else:
                        self.parse_rule(call, local_env, package, filename)
                elif isinstance(node, ast.For):
                    kinds = {
                        dotted(
                            next(
                                (k.value for k in call.keywords if k.arg == "_kind"),
                                call.func,
                            )
                        ).split(".")[-1]
                        for call in ast.walk(node)
                        if isinstance(call, ast.Call)
                    }
                    known_test_loop = (
                        isinstance(node.target, ast.Name)
                        and node.target.id == "test_file"
                        and isinstance(node.iter, ast.Name)
                        and node.iter.id == "test_files"
                        and "python_pytest" in kinds
                        and "python_library" not in kinds
                    )
                    if not known_test_loop:
                        self.package_error(
                            package,
                            filename,
                            "unsupported declaration loop: "
                            + ast.unparse(node.target)
                            + " in "
                            + ast.unparse(node.iter),
                        )

        visit(tree.body, env)
        for name in sorted(selected or set()):
            visit_helper(name)
        needed: dict[str, set[str]] = {}
        for name in sorted(called):
            helper, original = imports[name]
            needed.setdefault(helper, set()).add(original)
        for helper, names in needed.items():
            self.parse_file(helper, package, seen, names)

    def get(self, label: str) -> Rule | None:
        package = label.removeprefix("//executorch/").split(":", 1)[0]
        self.load_package(package)
        return self.rules.get(label)

    def owners(self, source: str) -> set[str]:
        for parent in PurePosixPath(source).parents:
            package = str(parent)
            if package == ".":
                continue
            if any(
                package + "/" + name in self.snapshot.paths
                for name in ("BUCK", "TARGETS")
            ):
                self.load_package(package)
                return {
                    r.label
                    for r in self.rules.values()
                    if r.kind == "python_library" and source in r.sources
                }
        return set()

    def reachable(self, start: Rule) -> tuple[set[str], list[str]]:
        visited, pending, errors = set(), [start.label], []
        while pending:
            label = pending.pop()
            if label in visited:
                continue
            visited.add(label)
            rule = self.get(label)
            if rule is None:
                errors.append(label + " (no supported declaration)")
                continue
            if rule.kind != "python_library":
                continue
            errors.extend(label + " (" + error + ")" for error in rule.errors)
            pending.extend(rule.deps)
        return visited, sorted(set(errors))


def python_imports(  # noqa: C901
    source: str, filename: str, snapshot: Snapshot
) -> set[tuple[str, str | None]]:
    results: set[tuple[str, str | None]] = set()

    def resolve(module: str) -> str | None:
        if not module.startswith("executorch."):
            return None
        path = module.removeprefix("executorch.").replace(".", "/")
        return next(
            (
                candidate
                for candidate in (path + ".py", path + "/__init__.py")
                if candidate in snapshot.paths
            ),
            None,
        )

    class Imports(ast.NodeVisitor):
        # ast.NodeVisitor dispatch requires these method names.
        # pylint: disable=invalid-name
        def visit_If(self, node: ast.If) -> None:
            if dotted(node.test) in {"TYPE_CHECKING", "typing.TYPE_CHECKING"}:
                for child in node.orelse:
                    self.visit(child)
            else:
                self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name.startswith("executorch."):
                    results.add((alias.name, resolve(alias.name)))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            if node.level:
                package = ["executorch"] + list(PurePosixPath(filename).parent.parts)
                module = ".".join(
                    package[: len(package) - node.level + 1]
                    + ([module] if module else [])
                )
            if not module.startswith("executorch."):
                return
            for alias in node.names:
                child = module + "." + alias.name
                child_file = resolve(child)
                results.add(
                    (child, child_file) if child_file else (module, resolve(module))
                )

    Imports().visit(ast.parse(source, filename=filename))
    return results


def is_test_file(source: str, filename: str) -> bool:
    tree = ast.parse(source, filename=filename)
    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, function_types) and node.name.startswith("test_"):
            if not any(
                dotted(d.func if isinstance(d, ast.Call) else d).endswith("fixture")
                for d in node.decorator_list
            ):
                return True
        if isinstance(node, ast.ClassDef) and (
            node.name.startswith("Test")
            or any(dotted(base).endswith("TestCase") for base in node.bases)
        ):
            if any(
                isinstance(child, function_types) and child.name.startswith("test")
                for child in node.body
            ):
                return True
    return False


def registered_tests(  # noqa: C901
    declarations: Declarations,
) -> tuple[set[str], list[str]]:
    try:
        tree = ast.parse(declarations.snapshot.read(TEST_MACRO), filename=TEST_MACRO)
    except (KeyError, SyntaxError) as error:
        return set(), [str(error)]
    errors = []
    registrations: set[str] = set()

    def visit(statements: list[ast.stmt], env: dict[str, object]) -> None:
        for node in statements:
            if isinstance(node, ast.FunctionDef) and node.name == "define_arm_tests":
                visit(node.body, env)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    env[node.targets[0].id] = declarations.value(node.value, env, SCOPE)
                except Unsupported as error:
                    env.pop(node.targets[0].id, None)
                    if node.targets[0].id == "test_files":
                        errors.append(str(error))
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "test_files"
            ):
                try:
                    if not isinstance(node.op, ast.Add):
                        raise Unsupported(ast.unparse(node))
                    env["test_files"] = env.get(
                        "test_files", []
                    ) + declarations.strings(node.value, env, SCOPE)
                except Unsupported as error:
                    errors.append(str(error))
            elif isinstance(node, ast.If):
                try:
                    visit(
                        (
                            node.body
                            if declarations.value(node.test, env, SCOPE)
                            else node.orelse
                        ),
                        env,
                    )
                except Unsupported as error:
                    if any(
                        isinstance(x, ast.Name) and x.id == "test_files"
                        for x in ast.walk(node)
                    ):
                        errors.append(str(error))
            elif isinstance(node, ast.For):
                if any(
                    isinstance(x, ast.Name)
                    and x.id == "test_files"
                    and isinstance(x.ctx, ast.Store)
                    for x in ast.walk(node)
                ):
                    errors.append("loop-modified test_files")

    # A test may intentionally be registered only for one build environment.
    for is_oss in (False, True):
        env = {"__is_oss__": is_oss}
        visit(tree.body, env)
        registrations.update(SCOPE + "/" + p for p in env.get("test_files", []))
    return registrations, errors


# pylint: disable-next=too-many-locals
def analyze(snapshot: Snapshot) -> set[Issue]:  # noqa: C901
    declarations = Declarations(snapshot)
    declarations.load_scope()
    issues: set[Issue] = {
        Issue("cannot validate", package, filename, error, error)
        for package, errors in declarations.package_errors.items()
        for filename, error in errors
    }
    libraries = [
        rule
        for rule in declarations.rules.values()
        if rule.kind == "python_library" and is_scoped_label(rule.label)
    ]
    for library in libraries:
        for error in library.errors:
            issues.add(
                Issue(
                    "cannot validate",
                    library.label,
                    library.declaration_file,
                    error,
                    error.split(":", 1)[0],
                )
            )
        reachable, incomplete = declarations.reachable(library)
        for error in incomplete:
            dependency = error.split(" ", 1)[0]
            if is_scoped_label(dependency):
                issues.add(
                    Issue(
                        "cannot validate",
                        library.label,
                        library.declaration_file,
                        "unresolved scoped dependency: " + error,
                        dependency,
                        "declaration",
                    )
                )
        for filename in sorted(library.sources):
            try:
                imports = python_imports(snapshot.read(filename), filename, snapshot)
            except (KeyError, SyntaxError) as error:
                issues.add(
                    Issue("cannot validate", library.label, filename, str(error))
                )
                continue
            for module, imported_file in imports:
                if imported_file in library.sources:
                    continue
                if imported_file is None:
                    issues.add(
                        Issue(
                            "cannot validate",
                            library.label,
                            filename,
                            "no in-repo source for " + module,
                            module,
                            "dependency",
                        )
                    )
                    continue
                owners = declarations.owners(imported_file)
                if owners & reachable:
                    continue
                if not owners:
                    issues.add(
                        Issue(
                            "cannot validate",
                            library.label,
                            filename,
                            "no supported python_library provider for " + module,
                            module,
                            "dependency",
                        )
                    )
                    continue
                detail = module + " requires " + " or ".join(sorted(owners))
                if incomplete:
                    detail += "; unresolved dependency declarations: " + ", ".join(
                        incomplete
                    )
                    kind = "cannot validate"
                else:
                    kind = "missing dependency"
                issues.add(
                    Issue(kind, library.label, filename, detail, module, "dependency")
                )
    registrations, errors = registered_tests(declarations)
    for error in errors:
        issues.add(
            Issue(
                "cannot validate",
                "Arm test registration",
                TEST_MACRO,
                error,
                error,
                "registration",
            )
        )
    for filename in sorted(snapshot.paths):
        if filename.startswith(SCOPE + "/") and filename.endswith(".py"):
            try:
                actual_test = is_test_file(snapshot.read(filename), filename)
            except SyntaxError as error:
                issues.add(
                    Issue(
                        "cannot validate", "Arm test registration", filename, str(error)
                    )
                )
                continue
            if actual_test and filename not in registrations:
                # A nested package may declare its own ordinary Python test.
                declarations.owners(filename)  # Load its nearest build package.
                explicitly_registered = any(
                    rule.kind in {"python_test", "python_pytest", "python_unittest"}
                    and filename in rule.sources
                    for rule in declarations.rules.values()
                )
                if not explicitly_registered:
                    issues.add(
                        Issue(
                            "unregistered test",
                            "Arm test registration",
                            filename,
                            "not in define_arm_tests test_files or a supported explicit Python test target",
                            "",
                            "registration",
                        )
                    )
    return issues


def compare(base: Snapshot, head: Snapshot) -> tuple[set[Issue], int]:
    baseline, current = analyze(base), analyze(head)
    previous = Declarations(base)
    previous.load_scope()
    present = Declarations(head)
    present.load_scope()
    for rule in previous.rules.values():
        if rule.kind != "python_library" or not is_scoped_label(rule.label):
            continue
        for source in rule.sources & head.paths:
            present.owners(source)
            if not any(
                source in target.sources
                and target.kind
                in {"python_library", "python_test", "python_pytest", "python_unittest"}
                for target in present.rules.values()
            ):
                current.add(
                    Issue(
                        "cannot validate",
                        rule.label,
                        source,
                        "previously declared Python source has no supported owner in the head snapshot",
                        "source ownership",
                        "ownership",
                    )
                )
    return current - baseline, len(current & baseline)


def print_report(regressions: set[Issue], unchanged: int) -> None:
    status = "FAIL" if regressions else "PASS"
    print(f"Arm Python Buck declaration check: {status}")
    print(f"  New issues: {len(regressions)}")
    print(f"  Existing issues unchanged: {unchanged}")
    wrapper = textwrap.TextWrapper(
        width=100, break_long_words=False, break_on_hyphens=False
    )
    for index, issue in enumerate(sorted(regressions), start=1):
        print(f"\n[{index}/{len(regressions)}] {issue.kind.capitalize()}")
        print(f"  File: {issue.source}")
        owner_label = "Target" if issue.owner.startswith("//") else "Scope"
        print(f"  {owner_label}: {issue.owner}")
        detail, _, unresolved = issue.detail.partition(
            "; unresolved dependency declarations: "
        )
        wrapper.initial_indent = "  Detail: "
        wrapper.subsequent_indent = "          "
        print(wrapper.fill(detail))
        if unresolved:
            print("  Unresolved dependency declarations:")
            wrapper.initial_indent = "    "
            wrapper.subsequent_indent = "    "
            print(wrapper.fill(unresolved))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--base", required=True, help="explicit pre-change Git revision"
    )
    parser.add_argument("--head", help="Git revision; omit to inspect the working tree")
    args = parser.parse_args(argv)
    try:
        regressions, unchanged = compare(
            GitSnapshot(args.repo.resolve(), args.base),
            GitSnapshot(args.repo.resolve(), args.head),
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Cannot validate Arm Python Buck declarations: {error}", file=sys.stderr)
        return 2
    print_report(regressions, unchanged)
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
