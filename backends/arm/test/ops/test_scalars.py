# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import common
import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineBI,
    TosaPipelineMI,
    TransformAnnotationPassPipeline,
)

"""
Summary of non-working cases.
MI:
    Op(scalar, tensor):
        One issue is that lift_constant_tensor_pass looks for a fake_tensor in the meta of the first
        node which does not work the first node is a scalar.
        Fixing that, the lowering fails since edge_program.graph_signatures.inputs_to_buffers is changed from
        {"_lifted_tensor_constant0":"_lifted_tensor_constant0"} to {"x":"_lifted_tensor_constant0"}
        somewhere in _transform in the to_edge step. This makes ArmPartitioner miss tagging the
        data in tag_constant_data.
        # MLETORCH-408
    Sub or inplace-sub with an integer input.
"""
input_t1 = Tuple[torch.Tensor, torch.scalar_tensor]  # Input x, Input y


class TestScalars(unittest.TestCase):
    """Tests various scalar cases"""

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class Sub(torch.nn.Module):
        def forward(self, x, y):
            return x - y

    class Div(torch.nn.Module):
        def forward(self, x, y):
            return x / y

    class Mul(torch.nn.Module):
        def forward(self, x, y):
            return x * y

    class MulScalar(torch.nn.Module):
        def forward(self, x, y):
            return torch.ops.aten.mul.Scalar(x, y)

    class DivScalar(torch.nn.Module):
        def forward(self, x, y):
            return torch.ops.aten.div.Scalar(x, y)

    class AddScalar(torch.nn.Module):
        def forward(self, x, y):
            return torch.ops.aten.add.Scalar(x, y)

    class SubScalar(torch.nn.Module):
        def forward(self, x, y):
            return torch.ops.aten.sub.Scalar(x, y)

    class AddInplace(torch.nn.Module):
        def forward(self, x, y):
            x += y
            return x

    class SubInplace(torch.nn.Module):
        def forward(self, x, y):
            x -= y
            return x

    class DivInplace(torch.nn.Module):
        def forward(self, x, y):
            x /= y
            return x

    class MulInplace(torch.nn.Module):
        def forward(self, x, y):
            x *= y
            return x

    class AddConst(torch.nn.Module):
        def forward(self, x):
            x = 1.0 + x
            return x

    class ShiftInplaceSub(torch.nn.Module):
        def forward(self, x):
            x = x >> 4
            x -= 10
            return x


# Inplace ops end with '_' (from aten naming)
ops = [
    ("Add", TestScalars.Add()),
    ("Sub", TestScalars.Sub()),
    ("Mul", TestScalars.Mul()),
    ("Div", TestScalars.Div()),
    ("Add_", TestScalars.AddInplace()),
    ("Sub_", TestScalars.SubInplace()),
    ("Mul_", TestScalars.MulInplace()),
    ("Div_", TestScalars.DivInplace()),
    ("MulScalar", TestScalars.MulScalar()),
    ("DivScalar", TestScalars.DivScalar()),
    ("AddScalar", TestScalars.AddScalar()),
    ("SubScalar", TestScalars.SubScalar()),
]

const_ops = [("Add", TestScalars.AddConst())]

dtypes = [("int", 3), ("float", 3.0)]
sizes = [("r1", (1)), ("r4", (2, 4, 5, 3))]

# Create combinations of tests
tensor_scalar_tests = {}
for op in ops:
    for dtype in dtypes:
        for size in sizes:
            test_name = f"{op[0]}_{dtype[0]}_{size[0]}"
            tensor = torch.rand(size[1])
            scalar = dtype[1]
            tensor_scalar_tests[test_name + "_ts"] = (op[1], tensor, scalar)
            # Don't add (scalar, tensor) test case for .Scalar ops.
            if op[0][-6:] == "Scalar":
                continue

            tensor_scalar_tests[test_name + "_st"] = (op[1], scalar, tensor)

tensor_const_tests = {}
for op in const_ops:
    for size in sizes:
        test_name = f"{op[0]}_{size[0]}"
        tensor = torch.rand(size[1])
        tensor_const_tests[test_name] = (op[1], tensor)


def _test_add_tosa_MI_pipeline(module: torch.nn.Module, test_data: tuple):
    pipeline = TosaPipelineMI[input_t1](module, test_data, aten_op=[], exir_op=[])
    pipeline.run()


def _test_add_tosa_BI_pipeline(
    module: torch.nn.Module, test_data: tuple, check_quant_nodes=True
):
    pipeline = TosaPipelineBI[input_t1](module, test_data, aten_op=[], exir_op=[])
    if not check_quant_nodes:
        pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


fail_str = "MLETORCH-408: Arithmetic ops can't handle scalars first for MI"
MI_xfails = {
    "Add_int_r1_st": fail_str,
    "Add_int_r4_st": fail_str,
    "Add_float_r1_st": fail_str,
    "Add_float_r4_st": fail_str,
    "Sub_int_r1_ts": fail_str,
    "Sub_int_r1_st": fail_str,
    "Sub_int_r4_ts": fail_str,
    "Sub_int_r4_st": fail_str,
    "Sub_float_r1_st": fail_str,
    "Sub_float_r4_st": fail_str,
    "Mul_int_r1_st": fail_str,
    "Mul_int_r4_st": fail_str,
    "Mul_float_r1_st": fail_str,
    "Mul_float_r4_st": fail_str,
    "Div_int_r1_st": fail_str,
    "Div_int_r4_st": fail_str,
    "Div_float_r1_st": fail_str,
    "Div_float_r4_st": fail_str,
    "Add__int_r1_st": fail_str,
    "Add__float_r1_st": fail_str,
    "Add__float_r4_st": fail_str,
    "Add__int_r4_st": fail_str,
    "Sub__int_r1_ts": fail_str,
    "Sub__int_r1_st": fail_str,
    "Sub__int_r4_ts": fail_str,
    "Sub__int_r4_st": fail_str,
    "Sub__float_r1_st": fail_str,
    "Sub__float_r4_st": fail_str,
    "Mul__int_r1_st": fail_str,
    "Mul__int_r4_st": fail_str,
    "Mul__float_r1_st": fail_str,
    "Mul__float_r4_st": fail_str,
    "Div__int_r1_st": fail_str,
    "Div__int_r4_st": fail_str,
    "Div__float_r1_st": fail_str,
    "Div__float_r4_st": fail_str,
}


@common.parametrize("tensor_scalar_tests", tensor_scalar_tests, MI_xfails)
def test_MI(tensor_scalar_tests: list):
    op, x, y = tensor_scalar_tests
    _test_add_tosa_MI_pipeline(op, (x, y))


def _test_passes_tosa_BI_pipeline(module: torch.nn.Module, test_data: tuple):
    pipeline = TransformAnnotationPassPipeline[input_t1](
        module, test_data, tosa_version="TOSA-0.80+BI"
    )
    pipeline.run()


fail_str = "MLETORCH-770: Numerical issues on Div Scalar."
passes_xfails = {
    "Div__int_r1_ts": fail_str,
    "Div__int_r4_ts": fail_str,
    "Div__float_r1_ts": fail_str,
    "Div__float_r4_ts": fail_str,
}


@common.parametrize("tensor_scalar_tests", tensor_scalar_tests, passes_xfails)
def test_passes_BI(tensor_scalar_tests: list):
    op, x, y = tensor_scalar_tests
    _test_passes_tosa_BI_pipeline(op, (x, y))


# op(Scalar float, tensor) works if the scalar is constant.
@common.parametrize("tensor_const_tests", tensor_const_tests)
def test_MI_const(tensor_const_tests: list):
    op, x = tensor_const_tests
    _test_add_tosa_MI_pipeline(op, (x,))


@common.parametrize("tensor_scalar_tests", tensor_scalar_tests)
def test_BI(tensor_scalar_tests: list):
    op, x, y = tensor_scalar_tests
    _test_add_tosa_BI_pipeline(op, (x, y))


# op(Scalar float, tensor) works if the scalar is constant.
@common.parametrize("tensor_const_tests", tensor_const_tests)
def test_BI_const(tensor_const_tests: list):
    op, x = tensor_const_tests
    _test_add_tosa_BI_pipeline(op, (x,))


def test_shift_sub_inplace_tosa_MI():
    _test_add_tosa_MI_pipeline(TestScalars.ShiftInplaceSub(), (torch.IntTensor(5),))


# Do not check for quant nodes in the graph for rshift.
def test_shift_sub_inplace_tosa_BI():
    _test_add_tosa_BI_pipeline(
        TestScalars.ShiftInplaceSub(), (torch.IntTensor(5),), check_quant_nodes=False
    )
