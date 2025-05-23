# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchgen.selective_build.selector import SelectiveBuilder


class TestExecuTorchSelectiveBuild(unittest.TestCase):
    def test_et_kernel_selected(self) -> None:
        yaml_config = """
et_kernel_metadata:
  aten::add.out:
   - "v1/6;0,1|6;0,1|6;0,1|6;0,1"
  aten::sub.out:
   - "v1/6;0,1|6;0,1|6;0,1|6;0,1"
"""
        selector = SelectiveBuilder.from_yaml_str(yaml_config)
        self.assertListEqual(
            ["v1/6;0,1|6;0,1|6;0,1|6;0,1"],
            selector.et_get_selected_kernels(
                "aten::add.out",
                [
                    "v1/6;0,1|6;0,1|6;0,1|6;0,1",
                    "v1/3;0,1|3;0,1|3;0,1|3;0,1",
                    "v1/6;1,0|6;0,1|6;0,1|6;0,1",
                ],
            ),
        )
        self.assertListEqual(
            ["v1/6;0,1|6;0,1|6;0,1|6;0,1"],
            selector.et_get_selected_kernels(
                "aten::sub.out", ["v1/6;0,1|6;0,1|6;0,1|6;0,1"]
            ),
        )
        self.assertListEqual(
            [],
            selector.et_get_selected_kernels(
                "aten::mul.out", ["v1/6;0,1|6;0,1|6;0,1|6;0,1"]
            ),
        )
        # We don't use version for now.
        self.assertListEqual(
            ["v2/6;0,1|6;0,1|6;0,1|6;0,1"],
            selector.et_get_selected_kernels(
                "aten::add.out", ["v2/6;0,1|6;0,1|6;0,1|6;0,1"]
            ),
        )
