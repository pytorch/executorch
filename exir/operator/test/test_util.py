# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir.operator.util import gen_out_variant_schema


class TestUtil(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_gen_out_variant_schema_from_functional(self) -> None:
        func_schema = str(torch.ops.aten.mul.Scalar._schema)

        out_schema = gen_out_variant_schema(func_schema)
        self.assertEqual(out_schema, str(torch.ops.aten.mul.Scalar_out._schema))

    def test_gen_out_variant_schema_from_inplace(self) -> None:
        func_schema = str(torch.ops.aten.add_.Scalar._schema)

        out_schema = gen_out_variant_schema(func_schema)
        self.assertEqual(out_schema, str(torch.ops.aten.add.Scalar_out._schema))

    def test_gen_out_variant_schema_for_custom_ops(self) -> None:
        func_schema = "custom::foo(Tensor a, Tensor b) -> (Tensor c, Tensor d)"

        out_schema = gen_out_variant_schema(func_schema)
        self.assertEqual(
            out_schema,
            "custom::foo.out(Tensor a, Tensor b, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))",
        )
