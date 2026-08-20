# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir.operator import convert as op_convert
from executorch.exir.operator.convert import to_out_variant
from torch._ops import OpOverload


class TestToOutVariant(unittest.TestCase):
    def test_already_out_var(self) -> None:
        self.assertTrue(
            op_convert.is_out_variant(
                "aten::topk",
                "values",
            )
        )

    def test_to_out_variant_already_out(self) -> None:
        op_overload = torch.ops.aten.topk.values
        out_var_op = op_convert.to_out_variant(op_overload)[0]
        self.assertTrue(op_overload is out_var_op)

    def test_to_out_variant_success(self) -> None:
        op_overload = torch.ops.aten.topk.default
        out_var_op, out_args = op_convert.to_out_variant(op_overload)

        input_tensor = torch.randn(100, 200)
        k = 10

        self.assertTrue(out_var_op is not torch.ops.aten.topk.default)
        self.assertTrue(out_var_op is torch.ops.aten.topk.values)

        expect_values, expect_indices = op_overload(input_tensor, k)

        kwargs = {}

        out_arg_dtypes = [val.dtype for val in (expect_values, expect_indices)]
        for name, dtype in zip(out_args, out_arg_dtypes):
            kwargs[name] = torch.Tensor().to(dtype=dtype)

        actual_values, actual_indices = out_var_op(input_tensor, k, **kwargs)

        self.assertTrue(torch.equal(expect_values, actual_values))
        self.assertTrue(torch.equal(expect_indices, actual_indices))

    # These checks are copied from the unsafe_replace_to_out_variant method
    # (https://www.fburl.com/code/ukwq31xz)
    # which are patch rules for the functional ops that can not be
    # handled generically before. Add unit tests to showoff that we can handle
    # the custom ops generically now!
    def test_to_out_variant_batch(self) -> None:
        aten = torch.ops.aten
        checklist = {
            aten.topk.default: (aten.topk.values, ("values", "indices")),
            aten.view_copy.default: aten.view_copy.out,
            aten.log_softmax.int: aten.log_softmax.int_out,
            aten.softmax.int: aten.softmax.int_out,
            aten.relu.default: aten.relu.out,
            torch.ops.my_awesome_3rdparty_ns.my_awesome_op.func: torch.ops.my_awesome_3rdparty_ns.my_awesome_op.out,
        }
        for func_op, expected_any in checklist.items():
            if isinstance(expected_any, OpOverload):
                # the default case where the out args are ("out",)
                expected_out_var = expected_any
                expected_out_args = ("out",)
            else:
                expected_out_var, expected_out_args = expected_any
            actual_out_var, actual_out_args = op_convert.to_out_variant(func_op)
            self.assertEqual(expected_out_var, actual_out_var)
            self.assertEqual(expected_out_args, actual_out_args)

    def test_to_out_variant_schema_mismatch(self) -> None:
        func_var_op: OpOverload = (
            torch.ops.my_awesome_3rdparty_ns.schema_mismatch_op.default
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Found an out variant for operator name .* but its schema mismatched with functional op.",
        ):
            to_out_variant(func_var_op)
