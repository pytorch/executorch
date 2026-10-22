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
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

from setuptools import find_namespace_packages
from setuptools.command.build_py import build_py

SETUP_PY = Path(__file__).resolve().parents[3] / "setup.py"
REPO_ROOT = SETUP_PY.parent
# Discovery is anchored on this file's location, not on the working directory, so the result
# does not depend on where the runner was started.
PACKAGE_ROOT = str(SETUP_PY.parent / "src")


# The helpers this test drives, shared by both loaders below.
_HELPERS = (
    # _VENDORED_DIR_NAMES is not used directly below, but _is_vendored_path closes over it.
    "_VENDORED_DIR_NAMES",
    "_VENDORED_SUBMODULE_FALLBACK",
    "_vendored_prefixes",
    "_is_vendored_path",
    # CustomBuildPy calls this, so the class cannot be exec'd without it.
    "_SHADER_TEMPLATE_MARKERS",
    "_is_shader_template",
    "_full_packages",
)


def _setup_py_module() -> ast.Module:
    # Name the encoding: these tests are collected on Windows too (pytest-windows.ini line 19),
    # where the default is cp1252, and setup.py holds a non-ascii apostrophe that would decode to
    # the wrong characters without a word rather than raising.
    return ast.parse(SETUP_PY.read_text(encoding="utf-8"))


def _load_from_setup_py(root: Path = None) -> Dict[str, object]:
    """The vendored-path helpers and the package list builder, from setup.py's source.

    Only those definitions are executed, so none of setup.py's module level build logic runs.
    """
    wanted = _HELPERS

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
        "__file__": str((root or SETUP_PY.parent) / "setup.py"),
        "Path": Path,
        "List": List,
        "Tuple": Tuple,
        "functools": functools,
        "subprocess": subprocess,
        "find_namespace_packages": find_namespace_packages,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SETUP_PY), "exec"),
        namespace,
    )
    return namespace


@functools.lru_cache(maxsize=None)
def _load_build_py() -> Dict[str, object]:
    """CustomBuildPy plus the helpers it calls, so analyze_manifest can be driven directly.

    Only the class body and those helpers run. Its methods reference names from setup.py's own
    imports, so the ones analyze_manifest touches are supplied here.
    """
    wanted = {"CustomBuildPy"} | set(_HELPERS)
    selected: List[ast.stmt] = []
    for node in _setup_py_module().body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted
            for target in node.targets
        ):
            selected.append(node)

    namespace: Dict[str, object] = {
        "__file__": str(SETUP_PY),
        "os": os,
        "ast": ast,
        "Path": Path,
        "functools": functools,
        "subprocess": subprocess,
        "build_py": build_py,
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
_vendored_prefixes = _NAMESPACE["_vendored_prefixes"]
_is_vendored_path = _NAMESPACE["_is_vendored_path"]
_full_packages = _NAMESPACE["_full_packages"]


def _discovered_packages() -> List[str]:
    """Everything setuptools finds, before any of this change's filtering."""
    return sorted(
        find_namespace_packages(
            where=PACKAGE_ROOT, include=["executorch", "executorch.*"]
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
        discovered = _discovered_packages()
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
        """No package from another repository is published.

        Compares against what discovery finds rather than re-filtering the helper's own output.
        Filtering the result with the same predicate the helper already applied is a tautology:
        it is empty whatever the helper did, so it would pass even with the exclusion removed.
        """
        discovered = set(_discovered_packages())
        shipped = set(_full_packages())
        dropped = discovered - shipped

        leaked = sorted(shipped & set(_vendored(discovered)))
        # Only the count and a few names, because a regression here leaks hundreds of
        # packages and the default diff would bury the message.
        self.assertEqual(
            len(leaked),
            0,
            f"the wheel would publish {len(leaked)} vendored packages, "
            f"e.g. {leaked[:3]}",
        )
        # And the helper really removed them, rather than discovery never having found them.
        self.assertEqual(
            dropped,
            set(_vendored(discovered)),
            "the set the helper drops is not the set of vendored packages on disk",
        )

    def test_the_exclusion_is_load_bearing(self) -> None:
        """Discovery without the exclusion finds the packages the exclusion removes."""
        self.assertLess(
            len(_full_packages()),
            len(_discovered_packages()),
            "the exclusion dropped nothing, so it is no longer doing any work",
        )

    def test_setup_passes_the_package_list(self) -> None:
        """The helper is actually wired into the full build.

        Without this, every test above still passes when the assignment that hands the list
        to setuptools is deleted, which is the whole of the change. The sibling wheel test
        asserts its own wiring the same way and for the same reason.

        The search is limited to the else branch of the minimal-build check, because an
        unrestricted walk also matches an assignment that can never run: moved into the
        minimal branch it is overwritten by the next line, and wrapped in a false condition
        it is dead, and both of those leave the full wheel discovering everything.
        """
        minimal_checks = [
            node
            for node in _setup_py_module().body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "_is_minimal_build"
        ]
        self.assertEqual(
            len(minimal_checks),
            1,
            "expected exactly one module level `if _is_minimal_build():`",
        )

        assigned = [
            node
            for node in minimal_checks[0].orelse
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

    def test_first_party_packages_still_ship(self) -> None:
        """A named first-party package survives the exclusion.

        Every other test here asks whether unwanted packages left. This one asks whether
        wanted ones stayed, which is the failure mode a too-greedy filter produces and the
        one nothing else would notice.
        """
        packages = _full_packages()
        for package in (
            "executorch.exir",
            "executorch.backends.xnnpack",
            "executorch.extension.pybindings",
            "executorch.devtools",
        ):
            self.assertIn(package, packages)

    def test_only_submodule_sections_are_read(self) -> None:
        """A `path` line outside a submodule section is not an exclusion prefix.

        Written against a file with a stray entry rather than by comparing git's output to git's
        own output. That comparison holds for any reader on today's clean file, so it would pass
        just as well for a line scanner that accepts `path =` from any section, which is the
        failure this is meant to catch: one stray line silently removes a real package.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitmodules").write_text(
                '[submodule "real"]\n'
                "\tpath = extension/llm/tokenizers\n"
                "[core]\n"
                "\tpath = executorch/exir\n"
            )
            self.assertEqual(
                _load_from_setup_py(root)["_vendored_prefixes"](),
                ("extension/llm/tokenizers",),
                "a path line outside a submodule section became an exclusion prefix",
            )

    def test_prefixes_are_normalized(self) -> None:
        """A legal but unusual spelling in .gitmodules still matches the real directory.

        git treats a trailing slash, a leading ./ and a doubled separator as the same path,
        so storing the raw text would silently disable the exclusion for that entry. Asserted
        against a written file rather than against today's values, because today's are already
        tidy and would pass either way.
        """
        for spelling in (
            "extension/llm/tokenizers/",
            "./extension/llm/tokenizers",
            "extension//llm/tokenizers",
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                # Only the path is read, so the entry needs no url.
                (root / ".gitmodules").write_text(
                    f'[submodule "t"]\n\tpath = {spelling}\n'
                )
                # The helper reads .gitmodules beside its own setup.py, so it is loaded
                # against the temporary tree rather than the real one.
                prefixes = _load_from_setup_py(root)["_vendored_prefixes"]()
                self.assertEqual(
                    prefixes,
                    ("extension/llm/tokenizers",),
                    f"{spelling!r} did not normalize",
                )

    def test_the_fallback_matches_gitmodules(self) -> None:
        """The hardcoded fallback still lists the same submodules the file does.

        It is only used when .gitmodules cannot be read, which is the case in a source
        distribution, so nothing else would notice it drifting out of date.
        """
        self.assertEqual(
            _vendored_prefixes(), _NAMESPACE["_VENDORED_SUBMODULE_FALLBACK"]
        )

        # And it is actually returned when the file is missing, which is the only case it
        # exists for. Without this the fallback could be replaced by an empty tuple and the
        # comparison above would still hold.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                _load_from_setup_py(Path(tmp))["_vendored_prefixes"](),
                _NAMESPACE["_VENDORED_SUBMODULE_FALLBACK"],
                "with no .gitmodules the submodule exclusion silently does nothing",
            )

    def test_a_submodule_name_with_a_space_is_read(self) -> None:
        """A submodule whose NAME contains a space still yields its path.

        git prints "<key> <value>" and permits spaces in the name, so splitting on the first
        space truncates the key and leaves a value that matches no directory, turning the
        exclusion off for that entry.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitmodules").write_text(
                '[submodule "my module"]\n\tpath = extension/llm/tokenizers\n'
            )
            self.assertEqual(
                _load_from_setup_py(root)["_vendored_prefixes"](),
                ("extension/llm/tokenizers",),
            )

    def test_a_broken_gitmodules_falls_back(self) -> None:
        """An unreadable .gitmodules reaches the fallback rather than excluding nothing.

        git exits non-zero with empty output on a bad section header or on conflict markers.
        Reading that as "this repository has no submodules" would turn the exclusion off with
        no warning, which is the one failure the fallback exists to prevent.
        """
        for broken in (
            '[submodule "x"\n\tpath = extension/llm/tokenizers\n',
            '<<<<<<< HEAD\n[submodule "x"]\n\tpath = a/b\n=======\n',
            "",
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / ".gitmodules").write_text(broken)
                prefixes = _load_from_setup_py(root)["_vendored_prefixes"]()
                self.assertEqual(
                    prefixes,
                    _NAMESPACE["_VENDORED_SUBMODULE_FALLBACK"],
                    f"a broken .gitmodules ({broken[:20]!r}) silently excluded nothing",
                )

    def test_manifest_filter_actually_drops_vendored_data(self) -> None:
        """The data-file half of the fix removes files, through the real build code path.

        `packages` only governs Python modules. Non-Python files arrive through the
        package_data manifest, and setuptools attributes a file under an unlisted directory to
        its nearest listed parent, so vendored data returns unless the manifest is filtered too.

        Drives CustomBuildPy.analyze_manifest itself rather than reimplementing the filter here.
        Checking the predicate in isolation is not enough: inverting the editable guard or
        short-circuiting the condition leaves the predicate correct and the build unfiltered,
        and both of those left an earlier version of this test green.
        """
        namespace = _load_build_py()
        build_py_class = namespace["CustomBuildPy"]

        vendored = (
            "src/executorch/backends/xnnpack/third-party/generate-cpuinfo-wrappers.py"
        )
        # A shader template goes through this same filter, and it needs its own example here:
        # deleting the shader line leaves the vendored assertions below green, so the manifest
        # half of the shader fix was unprotected.
        shader = "src/executorch/backends/vulkan/runtime/graph/ops/glsl/adamw_step.yaml"
        ordinary = "setup.py"
        # All of them have to exist on disk, because the filter also drops anything that is not a
        # file, and a missing path would be removed for that reason instead of this one.
        for relative in (vendored, shader, ordinary):
            self.assertTrue(
                (REPO_ROOT / relative).is_file(),
                f"{relative} is gone, so this test needs a different example",
            )

        # analyze_manifest calls up into setuptools first, which needs the full command
        # machinery. Only the filtering after that call is under test, so the parent's method
        # is replaced with a no-op for the duration and the manifest seeded directly. This runs
        # the shipped code path rather than a copy of it, which is the point: a filter that has
        # been turned off still reads correctly in the source.
        parent = build_py_class.__mro__[1]
        original = parent.analyze_manifest
        parent.analyze_manifest = lambda self: None
        try:
            stub = build_py_class.__new__(build_py_class)
            stub.editable_mode = False
            stub.manifest_files = {"executorch": [vendored, shader, ordinary]}
            stub.analyze_manifest()
            kept = stub.manifest_files["executorch"]
        finally:
            parent.analyze_manifest = original

        self.assertNotIn(
            vendored, kept, "a vendored data file survived the manifest filter"
        )
        self.assertNotIn(shader, kept, "a shader template survived the manifest filter")
        self.assertIn(ordinary, kept, "the filter dropped an ordinary file")

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
