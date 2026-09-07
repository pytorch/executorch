# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the packages the full wheel publishes.

The wheel used to carry the Python files and codegen scripts of every vendored third-party
checkout, because the full build passed no `packages` list and setuptools then discovered
everything under src/executorch. Those files exist to build the C++ targets, so nothing in
an installed wheel imports them.

Asserting on the discovery result rather than on a built wheel, because the behaviour under
test is a pure function of the source tree plus the exclude patterns, and a full build takes
minutes to exercise one filter. `.ci/scripts/test_minimal_wheel.sh` already covers the
built-artifact side for the minimal wheel.

setup.py is read rather than imported. It calls setup() at module scope, so importing it under
a test runner hands setup() the runner's own arguments and the session dies on an invalid
command name.
"""

import ast
import functools
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

from setuptools import find_namespace_packages

SETUP_PY = Path(__file__).resolve().parents[3] / "setup.py"
# Discovery is anchored on this file's location, not on the working directory, so the result
# does not depend on where the runner was started.
PACKAGE_ROOT = str(SETUP_PY.parent / "src")


def _setup_py_module() -> ast.Module:
    return ast.parse(SETUP_PY.read_text())


def _load_from_setup_py() -> Dict[str, object]:
    """The vendored-path helpers and the package list builder, from setup.py's source.

    Only those definitions are executed, so none of setup.py's module level build logic runs.
    """
    wanted = (
        "_VENDORED_DIR_NAMES",
        "_vendored_prefixes",
        "_is_vendored_path",
        "_full_packages",
    )

    selected: List[ast.stmt] = []
    found = set()
    for node in _setup_py_module().body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in wanted
            }
            if names:
                selected.append(node)
                found |= names

    assert found == set(
        wanted
    ), f"setup.py no longer defines {sorted(set(wanted) - found)}, so this test checks nothing"

    namespace: Dict[str, object] = {
        "__file__": str(SETUP_PY),
        "Path": Path,
        "List": List,
        "Tuple": Tuple,
        "functools": functools,
        "find_namespace_packages": find_namespace_packages,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SETUP_PY), "exec"),
        namespace,
    )
    return namespace


_NAMESPACE = _load_from_setup_py()
_VENDORED_DIR_NAMES = _NAMESPACE["_VENDORED_DIR_NAMES"]
_vendored_prefixes = _NAMESPACE["_vendored_prefixes"]
_is_vendored_path = _NAMESPACE["_is_vendored_path"]
_full_packages = _NAMESPACE["_full_packages"]


def _discover(exclude_vendored: bool) -> List[str]:
    """Package discovery over the real tree, with and without the exclusion."""
    patterns = (
        [
            pattern
            for name in sorted(_VENDORED_DIR_NAMES)
            for pattern in (f"*.{name}", f"*.{name}.*")
        ]
        if exclude_vendored
        else []
    )
    return sorted(
        find_namespace_packages(
            where=PACKAGE_ROOT,
            include=["executorch", "executorch.*"],
            exclude=patterns,
        )
    )


def _vendored(packages: List[str]) -> List[str]:
    return [
        package for package in packages if _is_vendored_path(package.replace(".", "/"))
    ]


class TestFullWheelPackages(unittest.TestCase):
    def test_the_tree_has_vendored_packages_to_exclude(self) -> None:
        """Fail rather than skip when there is nothing to exclude.

        Every other test here is vacuous on a tree with no vendored checkouts: an empty
        package list contains no vendored package, so the exclusion would look correct even
        if it had been deleted. Assert the premise instead of quietly passing on it.
        """
        discovered = _discover(exclude_vendored=False)
        self.assertNotEqual(
            discovered, [], f"no packages discovered under {PACKAGE_ROOT}"
        )
        self.assertNotEqual(
            _vendored(discovered),
            [],
            "no vendored third-party packages in this tree, so the exclusion below cannot "
            "be shown to do anything. Initialize the submodules before running this.",
        )

    def test_no_vendored_package_ships(self) -> None:
        """No package under a vendored third-party checkout is published."""
        leaked = _vendored(_full_packages())
        # Only the count and a few names, because a regression here leaks hundreds of
        # packages and the default diff would bury the message.
        self.assertEqual(
            len(leaked),
            0,
            f"the wheel would publish {len(leaked)} vendored packages, "
            f"e.g. {leaked[:3]}",
        )

    def test_the_exclusion_is_load_bearing(self) -> None:
        """Discovery without the exclusion finds the packages the exclusion removes."""
        self.assertLess(
            len(_discover(exclude_vendored=True)),
            len(_discover(exclude_vendored=False)),
            "the exclusion dropped nothing, so it is no longer doing any work",
        )

    def test_setup_passes_the_package_list(self) -> None:
        """The helper is actually wired into the full build.

        Without this, every test above still passes when the assignment that hands the list
        to setuptools is deleted, which is the whole of the change. The sibling wheel test
        asserts its own wiring the same way and for the same reason.
        """
        assigned = [
            node
            for node in ast.walk(_setup_py_module())
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "setup_kwargs"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "packages"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_full_packages"
        ]
        self.assertEqual(
            len(assigned),
            1,
            "setup.py does not assign _full_packages() to setup_kwargs['packages'], "
            "so the full build falls back to discovering every package",
        )

    def test_is_vendored_path_matches_whole_components(self) -> None:
        """The filter matches a path component, not a substring."""
        self.assertTrue(
            _is_vendored_path(
                "src/executorch/backends/xnnpack/third-party/XNNPACK/a.py"
            )
        )
        self.assertTrue(_is_vendored_path("src/executorch/x/third_party/y.yaml"))
        self.assertFalse(_is_vendored_path("src/executorch/exir/program/_program.py"))
        # "third-party" as part of a longer name is a different directory.
        self.assertFalse(_is_vendored_path("src/executorch/x/third-party-tools/y.py"))

    def test_submodules_outside_a_vendored_dir_are_recognized(self) -> None:
        """A submodule checked out under an ordinary name is still another repository.

        These are not matched by the directory name, so they are read from .gitmodules. Their
        nested copies also cannot satisfy the imports the code uses: the FACTO helper imports
        facto.specdb from the top level, and the tokenizers ship as a declared dependency.
        """
        prefixes = _vendored_prefixes()
        self.assertIn("backends/cadence/utils/FACTO", prefixes)
        self.assertIn("extension/llm/tokenizers", prefixes)
        for prefix in ("backends/cadence/utils/FACTO", "extension/llm/tokenizers"):
            self.assertTrue(_is_vendored_path(f"executorch/{prefix}"))
            self.assertTrue(_is_vendored_path(f"src/executorch/{prefix}/setup.py"))
        self.assertFalse(
            _is_vendored_path("executorch/extension/llm/custom_ops/op_sdpa.py")
        )

    def test_root_level_submodules_are_not_listed(self) -> None:
        """A submodule at the repository root is not a wheel path.

        Those are build tooling, never copied into the package, and listing one would put a
        bare single-word name into the matcher. That would then drop any directory sharing the
        name, anywhere in the tree, which is a much wider rule than intended.
        """
        for prefix in _vendored_prefixes():
            self.assertIn(
                "/",
                prefix,
                f"{prefix!r} is a root-level submodule and must not be listed",
            )
        self.assertFalse(_is_vendored_path("executorch/some/nested/shim"))


if __name__ == "__main__":
    unittest.main()
