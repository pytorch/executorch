# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the top level names the wheel publishes.

Here rather than in the wheel checks, because the behaviour under test is how setuptools
turns ext_modules into metadata, which is a pure function of the names on those entries.
Asserting it against a built wheel would need a full build to exercise one string filter,
and would not run on a machine that had not built one.

The defect this covers shipped. A prebuilt file the wheel only copies is declared as an
Extension, because a non-empty ext_modules is how setuptools decides a wheel is platform
specific, and those entries carry a synthetic name rather than a module path. setuptools
derives top_level.txt from every entry's name, so twenty recipe strings such as
"@EXECUTORCH_BuiltFile_%CMAKE_CACHE_DIR%/libexecutorch" were published as top level
import names on every row, leaving one correct line among them.

setup.py is read rather than imported. It calls setup() at module scope, so importing it
under a test runner hands setup() the runner's own arguments and the session dies on an
invalid command name.
"""

import ast
from pathlib import Path
from typing import Dict

from setuptools import Distribution, Extension

SETUP_PY = Path(__file__).resolve().parents[3] / "setup.py"


def _load_from_setup_py() -> Dict[str, object]:
    """The distribution class and the name prefix, taken from setup.py's own source.

    Only the two class definitions are executed, so nothing in setup.py's module level
    build logic runs.
    """
    module = ast.parse(SETUP_PY.read_text())
    wanted = ("_BaseExtension", "_ExecuTorchDistribution")
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    assert {node.name for node in classes} == set(
        wanted
    ), f"setup.py no longer defines {wanted}, so this test is checking nothing"

    # _BaseExtension's body refers to names setup.py imports, and only the prefix
    # constant matters here, so keep the assignments and drop everything else.
    for node in classes:
        if node.name == "_BaseExtension":
            node.body = [
                statement
                for statement in node.body
                if isinstance(statement, (ast.Assign, ast.AnnAssign))
            ]

    namespace: Dict[str, object] = {
        "Distribution": Distribution,
        "Extension": Extension,
    }
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(SETUP_PY), "exec"),
        namespace,
    )
    return namespace


_NAMESPACE = _load_from_setup_py()
_ExecuTorchDistribution = _NAMESPACE["_ExecuTorchDistribution"]
PREFIX = _NAMESPACE["_BaseExtension"].SYNTHETIC_NAME_PREFIX

# The shapes the real ext_modules list produces, including the two that made the
# published output confusing to read: setuptools splits a name at its first dot, so a
# library file name loses its suffix, and a copy target keeps its ":dst" half when the
# recipe has no dot at all.
SYNTHETIC_NAMES = [
    f"{PREFIX}%CMAKE_CACHE_DIR%/libexecutorch",
    f"{PREFIX}%CMAKE_CACHE_DIR%/backends/cuda/%BUILD_TYPE%/libaoti_cuda_shims",
    f"{PREFIX}%CMAKE_CACHE_DIR%/backends/mlx/mlx/mlx/backend/metal/kernels/mlx",
    f"{PREFIX}%CMAKE_CACHE_DIR%/third-party/flatc_ep/bin/flatc:executorch/data/bin/",
    f"{PREFIX}tools/wheel/pip_data_bin_init.py.in:executorch/data/bin/__init__.py",
]

# A real Python extension, which must survive the filter.
REAL_NAME = "executorch.extension.pybindings._portable_lib"


def _top_level_names(distribution_class):
    """The top level names setuptools would write for a distribution.

    Mirrors setuptools.command.egg_info.write_toplevel_names, which is what turns the
    names into the published file.
    """
    distribution = distribution_class(
        {
            "name": "executorch",
            "packages": ["executorch"],
            "ext_modules": [
                Extension(name=name, sources=[])
                for name in [*SYNTHETIC_NAMES, REAL_NAME]
            ],
        }
    )
    return sorted(
        dict.fromkeys(
            name.split(".", 1)[0] for name in distribution.iter_distribution_names()
        )
    )


def test_publishes_only_the_package_name():
    assert _top_level_names(_ExecuTorchDistribution) == ["executorch"]


def test_setup_is_told_to_use_the_distribution():
    """The filter only takes effect if setup() is actually given the class.

    Without this, deleting the `distclass=` argument leaves the test above green while the
    published metadata goes back to listing every recipe, which is the whole defect.
    """
    module = ast.parse(SETUP_PY.read_text())
    setup_calls = [
        node
        for statement in module.body
        if isinstance(statement, ast.Expr)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(setup_calls) == 1, "expected exactly one module level setup() call"
    distclass = [
        keyword.value
        for keyword in setup_calls[0].keywords
        if keyword.arg == "distclass"
    ]
    assert distclass, "setup() is not given a distclass, so the filter never runs"
    assert isinstance(distclass[0], ast.Name)
    assert distclass[0].id == "_ExecuTorchDistribution"


def test_plain_distribution_publishes_the_recipes():
    """Guards the test above, which passes trivially if the filter is unnecessary.

    Without this, removing the filter and the synthetic names together would leave a
    green test that no longer checks anything.
    """
    published = _top_level_names(Distribution)
    assert [name for name in published if name.startswith(PREFIX)]
    assert published != ["executorch"]
