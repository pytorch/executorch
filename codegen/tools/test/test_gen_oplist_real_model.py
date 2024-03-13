#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, Final, List

from executorch.codegen.tools.gen_oplist import (
    _get_kernel_metadata_for_model,
    _get_operators,
)

from libfb.py import parutil

MODEL_PATH: Final[str] = parutil.get_file_path("ModuleLinear.pte", pkg=__package__)


class TestGenOplistRealModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_gen_oplist_for_linear_success(self) -> None:
        operators = _get_operators(MODEL_PATH)
        self.assertEqual(len(operators), 2)
        self.assertIn("aten::add.out", operators)
        self.assertIn("aten::mul.out", operators)

    def test_get_kernel_metadata_for_model(self) -> None:
        metadata: Dict[str, List[str]] = _get_kernel_metadata_for_model(MODEL_PATH)

        self.assertEqual(len(metadata), 2)

        self.assertIn("aten::add.out", metadata)
        # We only have one dtype/dim-order combo for add (float/0,1)
        self.assertEqual(len(metadata["aten::add.out"]), 1)
        # We have 5 args
        self.assertEqual(
            metadata["aten::add.out"][0],
            "v1/6;0,1|6;0,1|6;0,1|6;0,1",
        )

        self.assertIn("aten::mul.out", metadata)
        self.assertEqual(len(metadata["aten::mul.out"]), 1)
        self.assertEqual(
            metadata["aten::mul.out"][0],
            "v1/6;0,1|6;0,1|6;0,1|6;0,1",
        )
