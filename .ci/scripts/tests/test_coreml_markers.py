# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests that the two Core ML availability rules agree.

Where coremltools can be installed is written down twice: as a dependency marker in
setup.py, and as a condition in conftest.py that drops the Core ML tests from collection.
They are written in different languages, PEP 508 and Python, so nothing but a comment
keeps them together. When they drift, the wheel declares a package on a platform the test
run assumes is absent, or the other way round, and pytest fails at collection with a bare
ModuleNotFoundError.

The rules are read out of the two files rather than restated here, so a change to either
one is exercised by this test instead of being duplicated a third time.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

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


def _base_dependency(name: str) -> str:
    """Return the requirement string for `name` from setup.py's _base_dependencies()."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_base_dependencies"
        ):
            continue
        for constant in ast.walk(node):
            if (
                isinstance(constant, ast.Constant)
                and isinstance(constant.value, str)
                and constant.value.startswith(name)
            ):
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
    # coremltools is. Nothing else in the wheel imports it.
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
