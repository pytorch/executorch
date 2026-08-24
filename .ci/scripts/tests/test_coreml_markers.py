# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests that the Core ML availability rules agree.

Where coremltools can be installed is written down three times: as a dependency marker in
setup.py, as a condition in conftest.py that drops the Core ML tests from collection, and
as `is_supported_platform_for_coreml_lowering` in export/utils.py, which callers use as an
import guard. They are written in different languages, PEP 508 and Python, so nothing but a
comment keeps them together. When they drift, the wheel declares a package on a platform
the test run assumes is absent, or the other way round, and pytest fails at collection with
a bare ModuleNotFoundError.

The rules are read out of the three files rather than restated here, so a change to any one
of them is exercised by this test instead of being duplicated a fourth time.
"""

import ast
import fnmatch
import re
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from packaging.markers import Marker

REPO_ROOT = Path(__file__).resolve().parents[3]

# platform_system, platform_machine, sys_platform
PLATFORMS = [
    ("Darwin", "arm64", "darwin"),
    ("Darwin", "x86_64", "darwin"),
    ("Linux", "x86_64", "linux"),
    ("Linux", "aarch64", "linux"),
    ("Linux", "armv7l", "linux"),
    ("Linux", "ppc64le", "linux"),
    ("Linux", "s390x", "linux"),
    ("Windows", "AMD64", "win32"),
    ("Windows", "ARM64", "win32"),
]

# Every version in requires-python, plus one past the end.
PYTHONS = [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15)]


class _VersionInfo(NamedTuple):
    """Stands in for sys.version_info: compares as a tuple, has .major/.minor."""

    major: int
    minor: int


def _normalise(name: str) -> str:
    # PEP 503: runs of -, _ and . collapse to a single -, then casefold.
    return re.sub(r"[-_.]+", "-", name).lower()


def _project_name(requirement: str) -> str:
    # Strip everything a PEP 508 requirement can carry after the project name: an extras
    # list, a version specifier, an environment marker, or a URL.
    head = re.split(r"[\[<>=!~;@\s]", requirement.strip(), maxsplit=1)[0]
    return _normalise(head)


def _base_dependency(name: str) -> str:
    """Return the requirement string for `name` from setup.py's _base_dependencies()."""
    # Compare PEP 503 normalised project names rather than using startswith, which would
    # match a longer name that merely begins with this one (scikit-learn against
    # scikit-learn-intelex) and would also match the function's own docstring, since that
    # is just another string constant in the body.
    wanted = _normalise(name)
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_base_dependencies"
        ):
            continue
        for constant in ast.walk(node):
            if not (
                isinstance(constant, ast.Constant) and isinstance(constant.value, str)
            ):
                continue
            if _project_name(constant.value) == wanted:
                return constant.value
    raise AssertionError(f"no {name} requirement in setup.py _base_dependencies()")


def _conftest_condition() -> str:
    """Return the source of conftest.py's _coremltools_is_declared expression."""
    tree = ast.parse((REPO_ROOT / "conftest.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_coremltools_is_declared"
            for target in node.targets
        ):
            return ast.unparse(node.value)
    raise AssertionError("no _coremltools_is_declared assignment in conftest.py")


def _coreml_ignore_globs() -> list[str]:
    """Return the globs conftest.py adds when _coremltools_is_declared is false.

    Reads the `if not _coremltools_is_declared:` statement rather than the module's
    resulting state, because importing conftest.py here would evaluate the condition
    for the interpreter running this test instead of for the platforms under test.
    """
    tree = ast.parse((REPO_ROOT / "conftest.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # `if not _coremltools_is_declared:`
        if not (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name)
            and test.operand.id == "_coremltools_is_declared"
        ):
            continue
        globs: list[str] = []
        for stmt in node.body:
            # `collect_ignore_glob += [...]` or `collect_ignore_glob = [...]`
            targets = (
                [stmt.target]
                if isinstance(stmt, ast.AugAssign)
                else stmt.targets if isinstance(stmt, ast.Assign) else []
            )
            if not any(
                isinstance(t, ast.Name) and t.id == "collect_ignore_glob"
                for t in targets
            ):
                continue
            for element in ast.walk(stmt.value):
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    globs.append(element.value)
        if globs:
            return globs
    raise AssertionError(
        "conftest.py does not gate collect_ignore_glob on `not _coremltools_is_declared`, "
        "so the Core ML tests are collected even where coremltools is not installed"
    )


def _marker_says_installed(
    requirement: str, system, machine, sys_platform, python
) -> bool:
    _, _, marker = requirement.partition(";")
    assert marker.strip(), f"{requirement} has no environment marker"
    return Marker(marker).evaluate(
        {
            "platform_system": system,
            "platform_machine": machine,
            "sys_platform": sys_platform,
            "python_version": f"{python[0]}.{python[1]}",
            "python_full_version": f"{python[0]}.{python[1]}.0",
        }
    )


def _conftest_says_installed(machine, sys_platform, python) -> bool:
    return bool(
        eval(  # the expression is read out of conftest.py, not taken from input
            compile(_conftest_condition(), "<conftest.py>", "eval"),
            {
                "sys": SimpleNamespace(platform=sys_platform, version_info=python),
                "platform": SimpleNamespace(machine=lambda: machine),
            },
        )
    )


def _support_helper_says_supported(system, machine, python) -> bool:
    """Evaluate export/utils.py's is_supported_platform_for_coreml_lowering().

    The function is extracted and run against stub platform/sys modules rather than
    imported, because importing export.utils pulls in torch and would answer for the
    interpreter running this test instead of for the platform under test.
    """
    tree = ast.parse((REPO_ROOT / "export" / "utils.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "is_supported_platform_for_coreml_lowering"
        ):
            namespace = {
                "platform": SimpleNamespace(
                    system=lambda: system, machine=lambda: machine
                ),
                # A namedtuple-like value so both `>= (3, 14)` and `.major`/`.minor`
                # work, matching what sys.version_info supports.
                "sys": SimpleNamespace(version_info=_VersionInfo(*python)),
                "logging": SimpleNamespace(info=lambda *a, **k: None),
            }
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, "<export/utils.py>", "exec"), namespace)
            return bool(namespace["is_supported_platform_for_coreml_lowering"]())
    raise AssertionError(
        "no is_supported_platform_for_coreml_lowering in export/utils.py"
    )


@pytest.mark.parametrize("system,machine,sys_platform", PLATFORMS)
@pytest.mark.parametrize("python", PYTHONS)
def test_conftest_matches_the_coremltools_marker(system, machine, sys_platform, python):
    declared = _marker_says_installed(
        _base_dependency("coremltools"), system, machine, sys_platform, python
    )
    collected = _conftest_says_installed(machine, sys_platform, python)
    assert declared == collected, (
        f"on {system}/{machine} with Python {python[0]}.{python[1]} setup.py "
        f"{'declares' if declared else 'does not declare'} coremltools while conftest.py "
        f"{'collects' if collected else 'skips'} the Core ML tests, so pytest will either fail "
        "at collection on a missing import or silently skip tests that could have run"
    )


@pytest.mark.parametrize("system,machine,sys_platform", PLATFORMS)
@pytest.mark.parametrize("python", PYTHONS)
def test_scikit_learn_follows_coremltools(system, machine, sys_platform, python):
    # scikit-learn is there for coremltools' palettization, so it is only useful where
    # coremltools is. Nothing else in the wheel imports scikit-learn itself; scipy, which
    # used to arrive as its transitive dependency, is declared in requirements-examples.txt
    # for the code that does import it.
    assert _marker_says_installed(
        _base_dependency("scikit-learn"), system, machine, sys_platform, python
    ) == _marker_says_installed(
        _base_dependency("coremltools"), system, machine, sys_platform, python
    )


def test_the_platforms_core_ml_is_used_on_are_still_covered():
    # A guard on the parametrized tests above: they would pass just as happily if both
    # rules said "never", which would quietly stop running the Core ML tests everywhere.
    requirement = _base_dependency("coremltools")
    assert _marker_says_installed(requirement, "Darwin", "arm64", "darwin", (3, 12))
    assert _marker_says_installed(requirement, "Linux", "x86_64", "linux", (3, 12))
    assert not _marker_says_installed(requirement, "Linux", "aarch64", "linux", (3, 12))
    assert not _marker_says_installed(requirement, "Windows", "AMD64", "win32", (3, 12))


def test_conftest_acts_on_the_condition():
    # The tests above only compare the two rules as expressions. They would still pass if
    # conftest.py computed _coremltools_is_declared and then never used it, which is the
    # one failure that reaches contributors as a ModuleNotFoundError at collection.
    #
    # Membership, not equality: adding a second correct glob is a legitimate change, and
    # test_the_ignore_globs_cover_every_core_ml_test_module below already catches a glob
    # that stops matching.
    assert "backends/apple/coreml/**" in _coreml_ignore_globs()


@pytest.mark.parametrize("system,machine,sys_platform", PLATFORMS)
@pytest.mark.parametrize("python", PYTHONS)
def test_export_support_helper_never_claims_more_than_the_marker(
    system, machine, sys_platform, python
):
    # export/utils.py is the third place this rule is written down. Callers use it as an
    # import guard (export/target_recipes.py raises ValueError when it says no), so it must
    # never claim support where coremltools is not declared, or the guard is bypassed and
    # the import raises ModuleNotFoundError instead.
    #
    # The reverse is allowed: the helper is deliberately narrower than the marker, because
    # coremltools installs on Darwin x86_64 but lowering is only supported on Apple silicon.
    declared = _marker_says_installed(
        _base_dependency("coremltools"), system, machine, sys_platform, python
    )
    supported = _support_helper_says_supported(system, machine, python)
    assert not (supported and not declared), (
        f"on {system}/{machine} with Python {python[0]}.{python[1]} "
        "is_supported_platform_for_coreml_lowering() reports support while setup.py does "
        "not declare coremltools, so the import guard in export/target_recipes.py is "
        "bypassed and importing coremltools raises ModuleNotFoundError"
    )


def test_the_support_helper_still_says_yes_where_core_ml_is_used():
    # A guard on the one-directional test above, which a helper that always returned False
    # would satisfy trivially.
    assert _support_helper_says_supported("Darwin", "arm64", (3, 12))
    assert _support_helper_says_supported("Linux", "x86_64", (3, 12))
    assert not _support_helper_says_supported("Windows", "AMD64", (3, 12))


def test_the_support_helper_tracks_the_marker_on_python_version():
    # The one-directional test above is deliberately loose on architecture, because
    # coremltools installs on Darwin x86_64 while lowering needs Apple silicon. It must not
    # be loose on the Python version too: if the marker starts declaring coremltools on a
    # version the helper still refuses, every caller keeps taking the "not supported" branch
    # on a platform where Core ML now works, and nothing else in this file notices.
    #
    # So pin the helper to the marker's own bound rather than to a literal, and read the
    # bound out of setup.py so widening the marker fails here until the helper follows.
    marker = _base_dependency("coremltools")
    for python in PYTHONS:
        declared = _marker_says_installed(marker, "Darwin", "arm64", "darwin", python)
        supported = _support_helper_says_supported("Darwin", "arm64", python)
        assert supported == declared, (
            f"on Darwin/arm64 with Python {python[0]}.{python[1]} setup.py "
            f"{'declares' if declared else 'does not declare'} coremltools but "
            f"is_supported_platform_for_coreml_lowering() reports "
            f"{'support' if supported else 'no support'}; the version bound in "
            "export/utils.py has drifted from the marker in setup.py"
        )


def test_the_ignore_globs_cover_every_core_ml_test_module():
    # A glob that no longer matches would silence nothing. Checked against the files on
    # disk so that moving the Core ML tests without updating conftest.py fails here.
    coreml_tests = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "backends/apple/coreml").rglob("test_*.py")
    )
    assert coreml_tests, "no Core ML test modules found, so this guard proves nothing"
    globs = _coreml_ignore_globs()
    for test in coreml_tests:
        assert any(
            fnmatch.fnmatch(test, glob) for glob in globs
        ), f"{test} is not covered by conftest.py's ignore globs {globs}"
