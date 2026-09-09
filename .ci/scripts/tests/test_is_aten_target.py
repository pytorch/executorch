# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the ATen detection in the Buck macro layer.

Here rather than as a build test, because the open source Buck build cannot query the
Vulkan backend at all, so the targets this decision matters most for are never built by
CI. The decision itself is a pure function of a target's keyword arguments, so it can be
exercised directly.

The detection had a real defect that this covers. A target can name a third-party
dependency either by its short name, which lands in ``external_deps``, or through
``external_dep_location``, which hands back the resolved label and lands in an ordinary
``deps`` list. Only the first was checked, so the Vulkan operator tests, which use the
second, compiled at the older standard against headers that need the newer one.
"""

import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MACROS = (
    REPO_ROOT / "shim_et" / "xplat" / "executorch" / "build" / "runtime_wrapper.bzl"
)

# What the open source dependency map resolves these names to. Kept here rather than
# imported so the test states the mapping it depends on.
RESOLVED = {
    "c10": ["//third-party:libtorch"],
    "libtorch": ["//third-party:libtorch"],
    "libtorch_python": ["//third-party:libtorch_python"],
    "torch-core-cpp": ["//third-party:libtorch"],
    "gtest_aten": ["//third-party:gtest_aten"],
    "gmock_aten": ["//third-party:gmock_aten"],
}
FALLTHROUGH = "@fallthrough@"


def _load_is_aten_target():
    """Execute the real macro text, with the little of Starlark it uses shimmed."""
    text = MACROS.read_text()
    start = text.index("def _has_pytorch_dep")
    end = text.index("def _cxx_library_common", start)

    env = types.SimpleNamespace(
        EXTERNAL_DEP_FALLTHROUGH=FALLTHROUGH,
        resolve_external_dep=lambda name: RESOLVED.get(name, FALLTHROUGH),
    )

    def _apply(obj, function):
        """Stand-in for selects.apply: run over each list the object holds."""
        if isinstance(obj, dict):
            return {key: function(value) for key, value in obj.items()}
        return function(obj)

    namespace = {
        # Starlark's type() returns a name, and the macros compare against "string".
        "type": lambda value: "string" if isinstance(value, str) else "other",
        "env": env,
        "selects": types.SimpleNamespace(apply=_apply),
    }
    exec(compile(text[start:end], str(MACROS), "exec"), namespace)
    return namespace["_is_aten_target"]


class TestIsAtenTarget(unittest.TestCase):
    def setUp(self) -> None:
        self.is_aten_target = _load_is_aten_target()

    def test_resolved_label_in_deps(self) -> None:
        """The Vulkan operator tests name libtorch this way."""
        self.assertTrue(
            self.is_aten_target(
                {
                    "name": "compute_graph_op_tests_bin",
                    "deps": [
                        "//third-party/googletest:gtest_main",
                        "//executorch/backends/vulkan:vulkan_graph_runtime",
                        "//third-party:libtorch",
                    ],
                }
            )
        )

    def test_resolved_label_in_exported_deps(self) -> None:
        self.assertTrue(
            self.is_aten_target(
                {"name": "some_lib", "exported_deps": ["//third-party:libtorch"]}
            )
        )

    def test_short_name_in_external_deps(self) -> None:
        for name in RESOLVED:
            with self.subTest(name=name):
                self.assertTrue(
                    self.is_aten_target({"name": "some_test", "external_deps": [name]})
                )

    def test_plain_target_is_not_aten(self) -> None:
        """The embedded builds rely on these staying at the older standard."""
        self.assertFalse(
            self.is_aten_target(
                {
                    "name": "op_add_test",
                    "deps": [
                        "//executorch/runtime/core:core",
                        "//third-party/googletest:gtest_main",
                    ],
                }
            )
        )

    def test_executorch_label_alone_is_not_aten(self) -> None:
        """Every label under the project contains the word torch."""
        self.assertFalse(
            self.is_aten_target(
                {"name": "evalue_test", "deps": ["//executorch/test/utils:utils"]}
            )
        )

    def test_resolved_label_inside_a_select(self) -> None:
        """A dep list can be a select, which cannot be walked like a list."""
        self.assertTrue(
            self.is_aten_target(
                {
                    "name": "some_test",
                    "deps": {
                        "DEFAULT": ["//third-party:libtorch"],
                        "ovr_config//os:windows": [],
                    },
                }
            )
        )

    def test_select_without_aten_is_not_aten(self) -> None:
        self.assertFalse(
            self.is_aten_target(
                {
                    "name": "some_test",
                    "deps": {"DEFAULT": ["//executorch/runtime/core:core"]},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
