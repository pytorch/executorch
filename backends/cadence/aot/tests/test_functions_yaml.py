# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import glob
import os
import unittest

import yaml

# Kernel namespaces that each backend's operator library compiles.
BACKEND_NAMESPACES: dict[str, set[str]] = {
    "functions.yaml": {"impl::generic", "torch::executor"},
    "functions_hifi.yaml": {"impl::HiFi", "impl::generic", "torch::executor"},
    "functions_vision.yaml": {"impl::vision", "torch::executor"},
    "functions_fusion_g3.yaml": {"impl::G3", "torch::executor"},
}


def load_kernels() -> list[tuple[str, str, str]]:
    """Returns (yaml name, op name, kernel name) for every functions*.yaml kernel."""
    aot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernels = []
    for path in sorted(glob.glob(os.path.join(aot_dir, "functions*.yaml"))):
        with open(path) as f:
            for entry in yaml.safe_load(f):
                op = entry["op"] if "op" in entry else entry["func"].partition("(")[0]
                kernels.extend(
                    (os.path.basename(path), op, kernel["kernel_name"])
                    for kernel in entry["kernels"]
                )
    return kernels


class TestFunctionsYaml(unittest.TestCase):
    def test_all_yamls_loaded(self) -> None:
        self.assertEqual(
            {yaml_name for yaml_name, _, _ in load_kernels()},
            set(BACKEND_NAMESPACES),
        )

    def test_kernel_names_omit_native(self) -> None:
        # Codegen appends ::native to the kernel namespace itself.
        self.assertEqual(
            [
                (yaml_name, op, kernel_name)
                for yaml_name, op, kernel_name in load_kernels()
                if "::native::" in kernel_name
            ],
            [],
        )

    def test_kernel_namespace_belongs_to_backend(self) -> None:
        self.assertEqual(
            [
                (yaml_name, op, kernel_name)
                for yaml_name, op, kernel_name in load_kernels()
                if kernel_name.rpartition("::")[0] not in BACKEND_NAMESPACES[yaml_name]
            ],
            [],
        )

    def test_custom_op_kernel_name_matches_op(self) -> None:
        mismatches = []
        for yaml_name, op, kernel_name in load_kernels():
            if not op.startswith("cadence::"):
                continue
            name, _, overload = op.removeprefix("cadence::").partition(".")
            if kernel_name.rpartition("::")[2] not in {
                f"{name}_out",
                f"{name}_{overload}",
                f"{name}_{overload}_out",
            }:
                mismatches.append((yaml_name, op, kernel_name))
        self.assertEqual(mismatches, [])
