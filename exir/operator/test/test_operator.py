# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir.operator.convert import (
    _get_overload_schema,
    output_to_aliased_input_map,
    to_out_variant,
    unwrap_op_overload,
)
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


class TestUnwrapOpOverload(unittest.TestCase):
    def test_aten_overload_returned_as_is(self) -> None:
        op = torch.ops.aten.add.Tensor
        self.assertIs(unwrap_op_overload(op), op)

    def test_wrapper_with_op_attr_peeled_to_aten(self) -> None:
        # Mimic the structural shape of `EdgeOpOverload` /
        # `BackendOpOverload`: a non-OpOverload wrapper that exposes
        # the underlying aten op via `_op`.
        class _FakeWrapper:  # noqa: B903
            def __init__(self, op: torch._ops.OpOverload) -> None:
                self._op = op

        aten_op = torch.ops.aten.add.Tensor
        wrapper = _FakeWrapper(aten_op)
        self.assertIs(unwrap_op_overload(wrapper), aten_op)

    def test_non_op_raises(self) -> None:
        with self.assertRaises(TypeError):
            unwrap_op_overload("not an op")
        with self.assertRaises(TypeError):
            unwrap_op_overload(None)
        with self.assertRaises(TypeError):
            unwrap_op_overload(42)

    def test_wrapper_with_non_op_underlying_raises(self) -> None:
        class _BadWrapper:
            _op = "not an op overload"

        with self.assertRaises(TypeError):
            unwrap_op_overload(_BadWrapper())


class TestOutputToAliasedInputMap(unittest.TestCase):
    def test_functional_op_returns_empty(self) -> None:
        # `aten::add.Tensor` is purely functional — no Tensor(a!) on
        # any return.
        schema = torch.ops.aten.add.Tensor._schema
        self.assertEqual(output_to_aliased_input_map(schema), {})

    def test_single_output_inplace_op(self) -> None:
        # `aten::index_put_` mutates `self` (arg 0) and returns it
        # (return 0).
        schema = torch.ops.aten.index_put_.default._schema
        self.assertEqual(output_to_aliased_input_map(schema), {0: 0})

    def test_single_output_inplace_via_pybind_parse(self) -> None:
        # Synthetic single-mutation schema parsed via the pybind
        # FunctionSchema parser; mutates `self` at position 0 and
        # returns it.
        schema = torch._C.parse_schema(
            "test::single(Tensor(a!) self, int n) -> Tensor(a!)"
        )
        self.assertEqual(output_to_aliased_input_map(schema), {0: 0})

    def test_multi_output_inplace_via_pybind_parse(self) -> None:
        # Synthetic multi-mutation schema: two write-aliased inputs
        # `a` and `b`, each returned as its own aliased output.
        schema = torch._C.parse_schema(
            "test::multi(Tensor(x!) a, Tensor(y!) b, int n) "
            "-> (Tensor(x!), Tensor(y!))"
        )
        # Output 0 (alias set {x}) → input 0 (a).
        # Output 1 (alias set {y}) → input 1 (b).
        self.assertEqual(output_to_aliased_input_map(schema), {0: 0, 1: 1})

    def test_partial_aliasing_returns_only_matched(self) -> None:
        # Two returns, only the first carries write-alias info.
        schema = torch._C.parse_schema(
            "test::partial(Tensor(z!) self, Tensor other) " "-> (Tensor(z!), Tensor)"
        )
        self.assertEqual(output_to_aliased_input_map(schema), {0: 0})

    def test_tied_inputs_first_match_wins(self) -> None:
        # Two inputs share the same write-alias set; per the docstring
        # contract ("the first matching input wins"), the helper must
        # map the single output back to input index 0, not 1.
        schema = torch._C.parse_schema(
            "test::tied(Tensor(a!) x, Tensor(a!) y) -> Tensor(a!)"
        )
        self.assertEqual(output_to_aliased_input_map(schema), {0: 0})
