# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the test modules the full wheel drops.

The wheel used to carry every test file in the repository, about 9.7 MB of Python that nothing
in an installed wheel can reach. A test case is only ever loaded by pytest from a path in the
checkout, never through the installed name, so shipping it buys nothing.

Shared helpers are the opposite. The suites here import each other by installed name, for
example `from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP`, so a
helper has to ship or collection breaks. That is why the keep set is computed from the import
graph and not from file names: `test_pipeline.py` and `test_add.py` are indistinguishable by
name and only one of them can go.

setup.py is read rather than imported. It calls setup() at module scope, so importing it under a
test runner hands setup() the runner's own arguments and the session dies on an invalid command
name.
"""

import ast
import functools
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from setuptools import find_namespace_packages

SETUP_PY = Path(__file__).resolve().parents[3] / "setup.py"
REPO_ROOT = SETUP_PY.parent

# Enough of setup.py to exercise the keep set, and nothing that builds anything.
_WANTED = (
    "_WALK_SKIP_DIRS",
    "_TEST_DIR_NAMES",
    "_CI_ENTRY_POINTS",
    "_CI_ENTRY_POINT_DIRS",
    "_SHADER_TEMPLATE_MARKERS",
    "_is_shader_template",
    "_VENDORED_DIR_NAMES",
    "_VENDORED_SUBMODULE_FALLBACK",
    "_top_level_package_dirs",
    "_first_party_module",
    "_is_test_module",
    "_module_name",
    "_import_targets",
    "_scan_imports",
    "_GENERATED_DIR_NAMES",
    "_unshipped_directories",
    "_import_graph",
    "_reachable_test_modules",
    "_vendored_prefixes",
    "_is_vendored_path",
    "_full_packages",
    "_minimal_packages",
)


def _setup_py_module() -> ast.Module:
    # Name the encoding: these tests are collected on Windows too (pytest-windows.ini line 19),
    # where the default is cp1252, and setup.py holds a non-ascii apostrophe that would decode to
    # the wrong characters without a word rather than raising.
    return ast.parse(SETUP_PY.read_text(encoding="utf-8"))


def _load_from_setup_py() -> Dict[str, object]:
    """Run only the named definitions from setup.py, not its build logic."""
    selected: List[ast.stmt] = []
    found: Set[str] = set()
    for node in _setup_py_module().body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in _WANTED
            }
            if names:
                selected.append(node)
                found |= names

    assert found == set(
        _WANTED
    ), f"setup.py no longer defines {sorted(set(_WANTED) - found)}, so this test checks nothing"

    namespace: Dict[str, object] = {
        "__file__": str(SETUP_PY),
        "ast": ast,
        "os": os,
        "Path": Path,
        "functools": functools,
        "subprocess": subprocess,
        "Dict": Dict,
        "FrozenSet": FrozenSet,
        "List": List,
        "Optional": Optional,
        "Set": Set,
        "Tuple": Tuple,
        "find_namespace_packages": find_namespace_packages,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SETUP_PY), "exec"),
        namespace,
    )
    return namespace


_NAMESPACE = _load_from_setup_py()
_is_test_module = _NAMESPACE["_is_test_module"]
_import_graph = _NAMESPACE["_import_graph"]
_reachable_test_modules = _NAMESPACE["_reachable_test_modules"]
_CI_ENTRY_POINTS = _NAMESPACE["_CI_ENTRY_POINTS"]


@functools.lru_cache(maxsize=None)
def _graph() -> Tuple[Set[str], Dict[str, Set[str]], Set[str]]:
    return _import_graph(REPO_ROOT / "src" / "executorch")


def _a_dropped_test_module() -> str:
    """A real test module the keep set excludes, so the wiring tests assert on real data.

    Must be a leaf inside a test package, because find_package_modules is only given a chance to
    drop something when the package it is asked about is itself under a test directory.
    """
    modules, _edges, _dynamic = _graph()
    keep = _reachable_test_modules()
    dropped = sorted(
        name
        for name in modules
        if name not in keep
        and _is_test_module(name.rsplit(".", 1)[0])
        and name.rsplit(".", 1)[1] != "__init__"
    )
    assert dropped, "nothing is dropped, so the wiring tests would be vacuous"
    return dropped[0]


_UNREACHABLE_TEST_MODULE = _a_dropped_test_module()


@functools.lru_cache(maxsize=None)
def _reachable_from_imports_only() -> FrozenSet[str]:
    """The keep set the import graph produces on its own, with no directory entries applied.

    Used to tell a load-bearing directory entry from a redundant one: a module the graph already
    reaches would ship whether or not its directory is listed.
    """
    modules, edges, dynamic = _graph()
    referenced = set(dynamic)
    referenced.update(_CI_ENTRY_POINTS)
    for targets in edges.values():
        referenced.update(targets)
    return frozenset(name for name in referenced if _is_test_module(name)) & modules


@functools.lru_cache(maxsize=None)
def _entry_point_dir_drivers() -> Dict[str, str]:
    """Why each `_CI_ENTRY_POINT_DIRS` entry exists, as the file that drives it.

    These directories cannot be re-derived from the source, which is the whole reason they are
    listed by hand: each is walked by something that never spells out a module name, so there is
    no import to find and no literal to grep for. What CAN be checked is that the thing doing the
    walking still exists and still refers to the directory. If a driver is deleted or stops
    mentioning its directory, the entry has outlived its reason and this pairing fails.

    Keyed by the dotted prefix, valued by a repository-relative path.
    """
    return {
        "executorch.backends.mlx.custom_kernel_ops": ".github/workflows/mlx.yml",
        "executorch.backends.webgpu.test": "backends/webgpu/scripts/test_webgpu_native_ci.sh",
        "executorch.backends.test.suite": "backends/test/suite/runner.py",
        "executorch.examples.models.llava.test": "examples/models/llava/README.md",
    }


@functools.lru_cache(maxsize=None)
def _tracked_shell_scripts() -> Tuple[str, ...]:
    """Shell scripts this repository actually owns, as repository-relative paths.

    Asked of git rather than found by walking. CI checks other repositories out INSIDE this one,
    for example a `pytorch/` sibling clone, and a walk cannot tell those files from ours. It found
    `pytorch/.ci/pytorch/test.sh` and reported a module belonging to a different project, so the
    walk failed on CI while passing in every local checkout.

    An archive with no git available yields nothing, which makes this test vacuous rather than
    wrong. It is a drift guard, so silence in an environment that cannot check is the safe way to
    fail.
    """
    try:
        listed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "*.sh"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        # No git on PATH, as in an unpacked source archive.
        return ()
    if listed.returncode:
        return ()
    return tuple(name for name in listed.stdout.split("\0") if name)


def _fake_prune(build_lib, source_root):
    """A CustomBuildPy whose prune runs, with build_lib and the source tree given directly.

    The real method resolves the source tree from setup.py's own location, so the lifted body is
    bound to a stand-in whose __file__ points at the fixture instead.
    """
    classes = [
        node
        for node in _setup_py_module().body
        if isinstance(node, ast.ClassDef) and node.name == "CustomBuildPy"
    ]
    assert len(classes) == 1, "setup.py no longer defines CustomBuildPy"
    bodies = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_prune_unstaged_files"
    ]
    assert len(bodies) == 1, "the stale-file prune is gone"

    namespace = {
        "os": os,
        "Path": Path,
        "__file__": str(source_root.parent / "setup.py"),
    }
    exec(compile(ast.unparse(bodies[0]), "prune", "exec"), namespace)

    class Stub:
        editable_mode = False
        packages = ["executorch", "executorch.pkg"]

        def __init__(self):
            self.build_lib = str(build_lib)

        def find_all_modules(self):
            # Deliberately omits stale.py, which is what marks it unwanted.
            return [("executorch", "__init__", ""), ("executorch.pkg", "__init__", "")]

        def get_package_dir(self, package):
            return str(source_root / Path(*package.split(".")))

        def find_data_files(self, package, src_dir):
            return []

    Stub._prune_unstaged_files = namespace["_prune_unstaged_files"]
    return Stub()


def _fake_build_py():
    """A CustomBuildPy whose overrides run, without configuring a real distribution.

    The overrides are lifted from setup.py and bound to a stand-in so they can be CALLED. The
    point is to exercise the real bodies: a test that only reads their syntax passes on code that
    never runs, which is the hole this helper exists to close.
    """
    classes = [
        node
        for node in _setup_py_module().body
        if isinstance(node, ast.ClassDef) and node.name == "CustomBuildPy"
    ]
    assert len(classes) == 1, "setup.py no longer defines CustomBuildPy"
    wanted = ("find_package_modules", "find_data_files")
    overrides = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in overrides} == set(
        wanted
    ), f"CustomBuildPy no longer overrides {sorted(set(wanted) - {n.name for n in overrides})}"

    package = _UNREACHABLE_TEST_MODULE.rsplit(".", 1)[0]
    leaf = _UNREACHABLE_TEST_MODULE.rsplit(".", 1)[1]
    modules = [(package, "__init__", "x"), (package, leaf, "y")]

    class Stub:
        editable_mode = False

        def __init__(self) -> None:
            self._data_files_to_return: List[str] = []

        # Stands in for build_py's own implementations, which need a configured distribution.
        def _super_find_package_modules(self, _package, _package_dir):
            return list(modules)

        def _super_find_data_files(self, _package, _src_dir):
            return list(self._data_files_to_return)

    namespace = dict(_NAMESPACE)
    namespace["os"] = os
    # `super()` needs a real base, so give the lifted bodies one that returns the fixtures above.
    source = "\n".join(
        ast.unparse(node)
        .replace(
            "super().find_package_modules(package, package_dir)",
            "self._super_find_package_modules(package, package_dir)",
        )
        .replace(
            "super().find_data_files(package, src_dir)",
            "self._super_find_data_files(package, src_dir)",
        )
        for node in overrides
    )
    exec(compile(source, "overrides", "exec"), namespace)
    for name in wanted:
        setattr(Stub, name, namespace[name])
    return Stub(), package, modules


class TestDroppedTestModules(unittest.TestCase):
    def test_something_is_actually_dropped(self) -> None:
        """The rule removes a substantial number of modules.

        Without this, every assertion below is vacuous on a keep set that happens to contain
        everything, and the whole change could be reverted with the suite still green.
        """
        modules, _, _ = _graph()
        tests = {name for name in modules if _is_test_module(name)}
        keep = _reachable_test_modules()
        self.assertGreater(len(tests), 500, "no test modules discovered at all")
        self.assertLess(
            len(keep),
            len(tests) // 2,
            f"keeping {len(keep)} of {len(tests)} test modules, so almost nothing is dropped",
        )

    def test_shared_helpers_are_kept(self) -> None:
        """Modules the suites import by installed name still ship.

        These are the ones whose removal breaks collection rather than a single test. Each is
        imported from outside its own directory, which is what makes the installed name matter.
        """
        keep = _reachable_test_modules()
        for helper in (
            "executorch.backends.arm.test.tester.test_pipeline",
            "executorch.backends.xnnpack.test.tester.tester",
            "executorch.backends.test.harness.stages",
            "executorch.backends.test.graph_builder",
            "executorch.exir.backend.test.op_partitioner_demo",
        ):
            self.assertIn(helper, keep)

    def test_leaf_cases_are_dropped(self) -> None:
        """A test case nothing imports does not ship.

        Chosen from different suites, because one backend getting this right says nothing about
        the others.
        """
        keep = _reachable_test_modules()
        modules, _, _ = _graph()
        for leaf in (
            "executorch.backends.arm.test.ops.test_add",
            "executorch.backends.xnnpack.test.ops.test_bilinear2d",
        ):
            self.assertIn(
                leaf, modules, f"{leaf} no longer exists, pick another example"
            )
            self.assertNotIn(leaf, keep)

    def test_relative_imports_are_followed(self) -> None:
        """A submodule reached only by a relative import is kept.

        backends/test/harness/stages/__init__.py does `from .export import Export`, so treating
        a relative import as reaching nothing new drops stages.export and breaks every importer
        of that package. This is a regression guard: it failed exactly that way once.
        """
        self.assertIn(
            "executorch.backends.test.harness.stages.export", _reachable_test_modules()
        )

    def test_dynamic_imports_are_followed(self) -> None:
        """A module named only as a string to importlib is kept.

        backends/mlx/test/run_all_tests.py does
        `importlib.import_module(".test_ops", package=__package__)`, which an import scan that
        only reads import statements cannot see. Note test_ops is also named like a leaf, so a
        file name rule would drop it.
        """
        self.assertIn(
            "executorch.backends.mlx.test.test_ops", _reachable_test_modules()
        )

    def test_ci_entry_points_are_kept(self) -> None:
        """The modules only a workflow names are kept."""
        keep = _reachable_test_modules()
        for name in _CI_ENTRY_POINTS:
            self.assertIn(name, keep)

    def test_ci_entry_points_still_match_the_workflows(self) -> None:
        """The hand-written CI list has not drifted from what the workflows actually run.

        The list is explicit rather than scanned at build time, because a source distribution
        carries no .github directory and a scan there would silently keep nothing. The cost of
        being explicit is drift, so it is checked here instead.
        """
        pattern = re.compile(r"executorch(?:\.[A-Za-z0-9_]+)+")
        referenced: Set[str] = set()
        # The whole tree, not just .github and .ci. A workflow often calls a script that lives
        # beside the backend it tests, and those name modules too:
        # backends/webgpu/scripts/test_webgpu_native_ci.sh runs six of them by dotted name.
        skip = {
            ".git",
            "pip-out",
            "cmake-out",
            "third-party",
            "third_party",
            "__pycache__",
        }
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for filename in filenames:
                # Markdown too: a README documenting `python -m executorch.x.test.y` is a
                # promise to users, and dropping that module breaks the documented command.
                # Python as well, because several modules document their own `python -m`
                # invocation in a docstring rather than in a README, and that is the same
                # promise written somewhere else.
                if not filename.endswith(
                    (".yml", ".yaml", ".sh", ".ps1", ".md", ".py")
                ):
                    continue
                path = Path(dirpath) / filename
                if path.resolve() == Path(__file__).resolve():
                    # This file names dropped modules as examples of what the rule removes, so
                    # reading itself would report them as promised and contradict its own tests.
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                referenced.update(pattern.findall(text))

        modules, _, _ = _graph()

        # A reference like executorch.a.test.b.SomeClass.some_method is one dotted run to the
        # regex, and it is not a module, so trim each match back to its longest real module
        # prefix. Without this the class-suffixed entries silently drop out of the comparison
        # and the guard protects fewer names than it appears to.
        def longest_module(name: str) -> str:
            parts = name.split(".")
            while parts:
                candidate = ".".join(parts)
                if candidate in modules:
                    return candidate
                parts.pop()
            return name

        expected = {
            trimmed
            for trimmed in (longest_module(name) for name in referenced)
            if _is_test_module(trimmed) and trimmed in modules
        }
        missing = sorted(expected - set(_CI_ENTRY_POINTS) - _reachable_test_modules())
        self.assertEqual(
            missing,
            [],
            f"a workflow names these test modules but nothing keeps them: {missing}",
        )

    def test_parent_packages_of_kept_modules_are_kept(self) -> None:
        """Every kept module's package chain is kept, or the dotted path cannot resolve."""
        keep = _reachable_test_modules()
        modules, _, _ = _graph()
        for name in keep:
            parts = name.split(".")
            for end in range(2, len(parts)):
                parent = ".".join(parts[:end])
                if _is_test_module(parent) and parent in modules:
                    self.assertIn(parent, keep, f"{parent} missing but {name} is kept")

    def test_shader_templates_do_not_ship(self) -> None:
        """Shader codegen inputs are dropped, and op definitions are not.

        The cmake build expands these into SPIR-V and WGSL headers, so the wheel already carries
        the compiled result. Matched on content, so the two examples below are the real
        distinction: one is a template, the other is read at run time through
        importlib.resources and must survive.
        """
        is_template = _NAMESPACE["_is_shader_template"]
        self.assertTrue(
            is_template("backends/vulkan/runtime/graph/ops/glsl/adamw_step.yaml")
        )
        self.assertTrue(
            is_template("backends/webgpu/runtime/ops/binary_op/binary_op.yaml")
        )
        for needed in (
            "exir/dialects/edge/edge.yaml",
            "kernels/portable/functions.yaml",
            "backends/cadence/aot/functions.yaml",
        ):
            self.assertFalse(is_template(needed), f"{needed} would stop shipping")

    def test_build_py_is_wired_to_the_custom_class(self) -> None:
        """setup() receives CustomBuildPy, not the stock build_py.

        Every other test here exercises the class directly, so all of them stay green when the
        cmdclass entry is pointed back at setuptools' own build_py. That single edit disables
        the module filter, the data file filter and the prune at once, and the wheel then ships
        everything again.
        """
        assignments = [
            node
            for node in ast.walk(_setup_py_module())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setup"
        ]
        self.assertEqual(len(assignments), 1, "expected exactly one setup() call")

        mapping = [kw.value for kw in assignments[0].keywords if kw.arg == "cmdclass"]
        self.assertEqual(len(mapping), 1, "setup() no longer passes cmdclass")
        wired = {
            key.value: value.id
            for key, value in zip(mapping[0].keys, mapping[0].values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Name)
        }
        self.assertEqual(
            wired.get("build_py"),
            "CustomBuildPy",
            "build_py is not wired to CustomBuildPy, so none of the filters run",
        )

    def test_both_package_lists_are_anchored_on_this_file(self) -> None:
        """Neither package list depends on the working directory.

        A cwd-relative `where` returns nothing when the build runs from anywhere but the
        repository root, and an empty package list makes the prune treat every staged file as
        unwanted. The full list was anchored for this reason; the minimal one has to match.

        Both lists are CALLED from a directory that is not the repository root, because reading
        the syntax of the `where=` argument only proves it is not a literal. Swapping the anchor
        for `Path.cwd()` leaves the syntax test green and breaks every build started elsewhere.
        """
        original = os.getcwd()
        os.chdir(tempfile.gettempdir())
        try:
            full = _NAMESPACE["_full_packages"]()
            minimal = _NAMESPACE["_minimal_packages"]()
        finally:
            os.chdir(original)
        self.assertIn("executorch", full)
        self.assertGreater(
            len(full), 100, "the full list collapsed when built from another directory"
        )
        self.assertIn("executorch", minimal)
        self.assertGreater(
            len(minimal),
            1,
            "the minimal list collapsed when built from another directory",
        )

    def test_stale_staged_files_are_pruned(self) -> None:
        """A rebuild removes what an earlier build staged and this one does not want.

        build_py only copies, so without this a second build into the same directory keeps
        every file the first one put there and the wheel packages it. The failure is silent:
        the build succeeds and the wheel quietly contains the dropped files.

        The prune is CALLED against a real staging directory, because checking that the method
        and its call site exist leaves an early `return` inside the body undetected, and the
        prune then does nothing while this test stays green.
        """
        staging = Path(tempfile.mkdtemp(prefix="prunetest-"))
        self.addCleanup(shutil.rmtree, staging, ignore_errors=True)
        source = staging / "src"
        (source / "executorch" / "pkg").mkdir(parents=True)
        for name in ("executorch/__init__.py", "executorch/pkg/__init__.py"):
            (source / name).write_text("")
        # Exists in the source tree and is NOT in build_py's file list, so the prune wants it
        # gone. That is the whole contract.
        (source / "executorch" / "pkg" / "stale.py").write_text(
            "# left by an earlier build\n"
        )
        build_lib = staging / "lib"
        shutil.copytree(source, build_lib)
        # Generated by a later build command, absent from src/, and must survive.
        (build_lib / "executorch" / "pkg" / "generated.py").write_text(
            "# from a template\n"
        )

        command = _fake_prune(build_lib, source)
        command._prune_unstaged_files()

        remaining = sorted(p.name for p in (build_lib / "executorch" / "pkg").iterdir())
        self.assertNotIn(
            "stale.py",
            remaining,
            "the prune left a file the current build does not want",
        )
        self.assertIn(
            "generated.py",
            remaining,
            "the prune deleted a file another command generated",
        )
        self.assertIn("__init__.py", remaining, "the prune deleted a wanted module")

    def test_build_py_applies_the_keep_set(self) -> None:
        """The drop is actually wired into the build, checked by CALLING the override.

        An earlier version of this test read the override's syntax tree instead. That passes on
        code that is present but never runs, so an early `return modules` at the top of the
        override left the filter dead with every assertion here still true. Build a real command
        and look at what it returns.
        """
        command, package, modules = _fake_build_py()
        result = command.find_package_modules(package, "unused")
        returned = {entry[1] for entry in result}
        offered = {entry[1] for entry in modules}
        self.assertIn("__init__", returned, "a kept package must still import")
        self.assertTrue(
            offered - returned,
            "find_package_modules returned everything it was offered, so nothing is dropped",
        )
        self.assertNotIn(
            _UNREACHABLE_TEST_MODULE.rsplit(".", 1)[1],
            returned,
            "an unreachable test module was not dropped",
        )

    def test_shader_filter_is_wired_into_find_data_files(self) -> None:
        """The shader classifier is actually CALLED, not merely correct.

        test_shader_templates_do_not_ship above checks the predicate. That is not the same thing:
        deleting the filtering line in find_data_files leaves the predicate perfect and unused,
        and every shader template ships again.
        """
        command, _package, _modules = _fake_build_py()
        template = "backends/vulkan/runtime/graph/ops/glsl/adamw_step.yaml"
        needed = "kernels/portable/functions.yaml"
        root = str(REPO_ROOT)
        command._data_files_to_return = [
            os.path.join(root, template),
            os.path.join(root, needed),
        ]
        kept = command.find_data_files("executorch", root)
        self.assertNotIn(
            os.path.join(root, template),
            kept,
            "find_data_files does not drop shader templates, so the filter is not wired in",
        )
        self.assertIn(
            os.path.join(root, needed),
            kept,
            "find_data_files dropped a yaml the runtime reads",
        )

    def test_ci_entry_point_dirs_are_all_load_bearing(self) -> None:
        """Every listed directory still has a driver, and still keeps something.

        Nothing referenced `_CI_ENTRY_POINT_DIRS`, so an entry could be deleted with the whole
        suite green: removing the backend suite line silently stopped 86 modules shipping. Two
        checks close that. Each entry must be paired with the file that walks it, which fails when
        an entry is added or removed without updating the pairing, and each entry must keep modules
        the import graph cannot reach on its own, which fails when an entry becomes dead weight.
        """
        listed = set(_NAMESPACE["_CI_ENTRY_POINT_DIRS"])
        self.assertTrue(listed, "the list is empty, so nothing is protected")

        drivers = _entry_point_dir_drivers()
        self.assertEqual(
            listed,
            set(drivers),
            "_CI_ENTRY_POINT_DIRS and its list of drivers disagree. Add the new entry with the "
            "file that walks it, or drop the driver for the entry that went away",
        )

        for prefix, driver in sorted(drivers.items()):
            path = REPO_ROOT / driver
            self.assertTrue(
                path.is_file(),
                f"{prefix} is kept for {driver}, which no longer exists, so the entry may be "
                "obsolete",
            )
            tail = prefix.split(".")[-1]
            self.assertIn(
                tail,
                path.read_text(encoding="utf-8", errors="replace"),
                f"{driver} no longer mentions {tail}, so it may have stopped driving {prefix}",
            )

        # And the other direction: an entry that keeps nothing new is dead weight.
        reached_anyway = _reachable_from_imports_only()
        keep = _reachable_test_modules()
        for entry in sorted(listed):
            covered = {
                name for name in keep if name == entry or name.startswith(f"{entry}.")
            }
            self.assertTrue(
                covered - reached_anyway,
                f"{entry} keeps nothing the import graph does not already reach, so the entry "
                "is redundant and should be removed",
            )

    def test_ci_entry_points_cover_constructed_module_names(self) -> None:
        """A runner that BUILDS a dotted name is covered too.

        The drift test above searches for a literal dotted name, so it cannot see a script that
        assembles one, and a directory whose tests are only reached that way would be dropped
        with nothing to warn about.

        A script that runs from a checkout by design is exempt, and says so in its own header.
        `backends/apple/coreai/run_all_tests.sh` is the current example: it cds to the repository
        root, so it always finds the files on disk and never needs them installed.
        """
        pattern = re.compile(r"find\s+([A-Za-z0-9_./-]+)\s+-name\s+'?test_\*\.py'?")
        keep = _reachable_test_modules()
        listed = _NAMESPACE["_CI_ENTRY_POINT_DIRS"]
        unprotected = []
        for relative in _tracked_shell_scripts():
            path = REPO_ROOT / relative
            text = path.read_text(encoding="utf-8", errors="replace")
            walked = pattern.findall(text)
            if not walked:
                continue
            if "not a landing artifact" in text:
                continue
            for entry in walked:
                dotted = "executorch." + entry.strip("./").replace("/", ".")
                covered = any(
                    dotted == prefix or dotted.startswith(f"{prefix}.")
                    for prefix in listed
                ) or any(name.startswith(f"{dotted}.") for name in keep)
                if not covered:
                    unprotected.append(f"{relative} -> {dotted}")
        self.assertEqual(
            unprotected,
            [],
            "a script discovers test modules under these paths by building dotted names, and "
            "nothing keeps them. Either add the directory to _CI_ENTRY_POINT_DIRS in setup.py, "
            "or say in the script's header that it is not a landing artifact if it only ever "
            f"runs from a checkout: {unprotected}",
        )

    def test_unprefixed_first_party_imports_count_as_references(self) -> None:
        """`from backends.x import y` keeps y, the same as the prefixed spelling.

        This repository imports itself both ways: most code says `executorch.backends.x`, but the
        Arm suites say `backends.arm.test...`, which resolves because the repository root is on
        sys.path. Both name the same file. Following only the prefixed spelling dropped two shared
        helpers with eight importers between them, which is the invariant this change exists to
        preserve.
        """
        first_party = _NAMESPACE["_first_party_module"]
        self.assertEqual(
            first_party("backends.arm.test.common"),
            "executorch.backends.arm.test.common",
        )
        self.assertEqual(
            first_party("executorch.exir.tests.common"), "executorch.exir.tests.common"
        )
        # A third-party module whose first component is not one of ours stays out.
        self.assertIsNone(first_party("torch.nn.functional"))
        self.assertIsNone(first_party("numpy"))

        keep = _reachable_test_modules()
        for helper in (
            "executorch.backends.arm.test._custom_vgf_test_utils",
            "executorch.backends.arm.test.runtime._vgf_runtime_test_utils",
        ):
            self.assertIn(
                helper,
                keep,
                f"{helper} is imported without the executorch prefix and must still ship",
            )

    def test_importers_outside_the_shipped_tree_are_followed(self) -> None:
        """A file the wheel does not carry can still import one that it does.

        `src/executorch` is a subset of the checkout, so an importer in a directory that is never
        packaged is invisible to a walk of the shipped tree alone. Its imports still have to keep
        their targets: test/end2end/test_end2end.py imports two model helpers out of exir/tests.
        """
        keep = _reachable_test_modules()
        importer = REPO_ROOT / "test" / "end2end" / "test_end2end.py"
        self.assertTrue(
            importer.is_file(), "this test needs a different example importer"
        )
        for helper in (
            "executorch.exir.tests.dynamic_shape_models",
            "executorch.exir.tests.transformer",
        ):
            self.assertIn(
                helper,
                keep,
                f"{helper} is imported from outside the shipped tree and must still ship",
            )

    def test_vendored_trees_are_not_read_as_import_evidence(self) -> None:
        """A vendored submodule's own imports do not keep anything.

        The package list excludes vendored trees, so nothing in one ships. The import scan has to
        agree, or the two disagree about the same directory: a submodule checked out under an
        ordinary name, rather than under `third-party`, was read as first-party and its imports
        kept test modules the wheel never carries.

        Skipping by directory name alone is not enough, which is why this asserts on the scan's
        output rather than on the skip list.
        """
        modules, _edges, _dynamic = _graph()
        is_vendored = _NAMESPACE["_is_vendored_path"]
        vendored = sorted(
            name for name in modules if is_vendored(name.replace(".", "/"))
        )
        self.assertEqual(
            vendored,
            [],
            "the import scan read these vendored modules as first-party, so their imports can "
            f"keep test modules nothing shipped reaches: {vendored[:5]}",
        )

    def test_generated_directories_are_not_read_as_import_evidence(self) -> None:
        """A build tree or an in-tree virtualenv does not vote on what ships.

        Those hold an INSTALLED copy of this package, so reading one lets the last wheel decide
        what the next carries: a file that shipped once keeps itself alive. A clean checkout has
        none of them, so the guard is exercised here by creating one.
        """
        unshipped = _NAMESPACE["_unshipped_directories"]
        root = REPO_ROOT / "src" / "executorch"
        planted = REPO_ROOT / ".venv"
        created = not planted.exists()
        if created:
            (planted / "lib").mkdir(parents=True)
            self.addCleanup(shutil.rmtree, planted, ignore_errors=True)
        walked = {entry.name for entry in unshipped(root)}
        self.assertNotIn(
            ".venv",
            walked,
            "a generated directory is read as import evidence, so an installed copy of this "
            "package can keep test modules alive across builds",
        )
        self.assertIn(
            "test", walked, "the guard also dropped a real unshipped directory"
        )

    def test_editable_installs_are_left_alone(self) -> None:
        """An editable install still exposes every test module.

        It maps the package root to a directory, so the suites resolve from the source tree
        whatever is listed, and dropping modules there would only make the two install modes
        disagree for no benefit.
        """
        classes = [
            node
            for node in _setup_py_module().body
            if isinstance(node, ast.ClassDef) and node.name == "CustomBuildPy"
        ]
        overrides = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == "find_package_modules"
        ]
        source = ast.unparse(overrides[0])
        self.assertIn("editable_mode", source)


if __name__ == "__main__":
    unittest.main()
