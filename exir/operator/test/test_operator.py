# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir.operator.convert import _get_overload_schema, to_out_variant
from executorch.exir.operator.util import gen_out_variant_schema
from torch.library import _scoped_library, impl, impl_abstract


class TestOperator(unittest.TestCase):
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

    def test_to_out_variant_mutable(self) -> None:

        with _scoped_library("DO_NOT_USE_TEST_ONLY", "DEF") as lib:

            lib.define("custom_mutator(Tensor x, Tensor(a!) y) -> Tensor")
            lib.define(
                "custom_mutator.out(Tensor x, Tensor(a!) y, *, Tensor(b!) out) -> Tensor(b!)"
            )

            @impl(lib, "custom_mutator", "Meta")
            def custom_mutator_meta(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                return torch.empty_like(x)

            @impl(lib, "custom_mutator", "CompositeExplicitAutograd")
            def custom_mutator(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                return x + y.add_(1)

            @impl_abstract("DO_NOT_USE_TEST_ONLY::custom_mutator.out")
            def custom_mutator_out(
                x: torch.Tensor,
                y: torch.Tensor,
                out: torch.Tensor,
            ) -> torch.Tensor:
                out = custom_mutator_meta(
                    x,
                    y,
                )
                return out

            out, _ = to_out_variant(
                torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator.default
            )
            schema = _get_overload_schema(out)
            self.assertEqual(
                schema.__str__(),
                "DO_NOT_USE_TEST_ONLY::custom_mutator.out(Tensor x, Tensor(a!) y, *, Tensor(b!) out) -> Tensor(b!)",
            )
