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
import subprocess
import unittest
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

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
    "_is_test_module",
    "_module_name",
    "_import_targets",
    "_import_graph",
    "_reachable_test_modules",
    "_vendored_prefixes",
    "_is_vendored_path",
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
                if not filename.endswith((".yml", ".yaml", ".sh", ".ps1", ".md")):
                    continue
                path = Path(dirpath) / filename
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

    def test_build_py_applies_the_keep_set(self) -> None:
        """The drop is actually wired into the build.

        Without this, the keep set could be perfect and unused, and every test above would still
        pass while the wheel shipped everything.
        """
        classes = [
            node
            for node in _setup_py_module().body
            if isinstance(node, ast.ClassDef) and node.name == "CustomBuildPy"
        ]
        self.assertEqual(len(classes), 1, "setup.py no longer defines CustomBuildPy")

        overrides = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == "find_package_modules"
        ]
        self.assertEqual(
            len(overrides),
            1,
            "CustomBuildPy does not override find_package_modules, so nothing is dropped",
        )
        calls = [
            node
            for node in ast.walk(overrides[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_reachable_test_modules"
        ]
        self.assertTrue(calls, "the override does not consult the keep set")

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
