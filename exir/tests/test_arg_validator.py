#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.verification.arg_validator import EdgeOpArgValidator
from torch.export import export


class TestArgValidator(unittest.TestCase):
    """Test for EdgeOpArgValidator"""

    def setUp(self) -> None:
        super().setUp()

    def test_edge_dialect_passes(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        m = TestModel()
        inputs = (torch.randn(1, 3, 100, 100).to(dtype=torch.int),)
        egm = to_edge(export(m, inputs)).exported_program().graph_module
        validator = EdgeOpArgValidator(egm)
        validator.run(*inputs)
        self.assertEqual(len(validator.violating_ops), 0)

    def test_edge_dialect_fails(self) -> None:
        # torch.bfloat16 is not supported by edge::aten::_log_softmax
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, x):
                return self.m(x)

        inputs = (torch.randn(1, 3, 100, 100).to(dtype=torch.bfloat16),)
        egm = (
            to_edge(
                export(M(), inputs),
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            .exported_program()
            .graph_module
        )
        validator = EdgeOpArgValidator(egm)
        validator.run(*inputs)
        self.assertEqual(len(validator.violating_ops), 1)
        key: EdgeOpOverload = next(iter(validator.violating_ops))
        self.assertEqual(
            key.name(),
            ops.edge.aten._log_softmax.default.name(),
        )
        self.assertDictEqual(
            validator.violating_ops[key],
            {
                "self": torch.bfloat16,
                "__ret_0": torch.bfloat16,
            },
        )
