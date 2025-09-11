# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

"""
Summary of non-working cases.
FP:
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


"""Tests various scalar cases"""


class Add(torch.nn.Module):
    aten_op = "torch.ops.aten.add.Tensor"

    def forward(self, x, y):
        return x + y


class Sub(torch.nn.Module):
    aten_op = "torch.ops.aten.sub.Tensor"

    def forward(self, x, y):
        return x - y


class Div(torch.nn.Module):
    aten_op = "torch.ops.aten.div.Tensor"

    def forward(self, x, y):
        return x / y


class Mul(torch.nn.Module):
    aten_op = "torch.ops.aten.mul.Tensor"

    def forward(self, x, y):
        return x * y


class MulScalar(torch.nn.Module):
    aten_op = "torch.ops.aten.mul.Scalar"

    def forward(self, x, y):
        return torch.ops.aten.mul.Scalar(x, y)


class DivScalar(torch.nn.Module):
    aten_op = "torch.ops.aten.div.Scalar"

    def forward(self, x, y):
        return torch.ops.aten.div.Scalar(x, y)


class AddScalar(torch.nn.Module):
    aten_op = "torch.ops.aten.add.Scalar"

    def forward(self, x, y):
        return torch.ops.aten.add.Scalar(x, y)


class SubScalar(torch.nn.Module):
    aten_op = "torch.ops.aten.sub.Scalar"

    def forward(self, x, y):
        return torch.ops.aten.sub.Scalar(x, y)


class AddInplace(torch.nn.Module):
    aten_op = "torch.ops.aten.add_.Tensor"

    def forward(self, x, y):
        x += y
        return x


class SubInplace(torch.nn.Module):
    aten_op = "torch.ops.aten.sub_.Tensor"

    def forward(self, x, y):
        x -= y
        return x


class DivInplace(torch.nn.Module):
    aten_op = "torch.ops.aten.div_.Tensor"

    def forward(self, x, y):
        x /= y
        return x


class MulInplace(torch.nn.Module):
    aten_op = "torch.ops.aten.mul_.Tensor"

    def forward(self, x, y):
        x *= y
        return x


class AddConst(torch.nn.Module):
    aten_op = "torch.ops.aten.add.Tensor"

    def forward(self, x):
        x = 1.0 + x
        return x


class ShiftInplaceSub(torch.nn.Module):

    def forward(self, x):
        x = x >> 4
        x -= 10
        return x


dtypes = [("int", 3), ("float", 3.0)]
sizes = [("r1", (1)), ("r4", (2, 4, 5, 3))]

# Create combinations of tests
tensor_scalar_tests = {}
for dtype in dtypes:
    for size in sizes:
        test_name = f"{dtype[0]}_{size[0]}"
        tensor = torch.rand(size[1])
        scalar = dtype[1]
        tensor_scalar_tests[test_name + "_ts"] = (tensor, scalar)
        # # Don't add (scalar, tensor) test case for .Scalar ops.
        # if op[0][-6:] == "Scalar":
        # continue

        tensor_scalar_tests[test_name + "_st"] = (scalar, tensor)

tensor_const_tests = {}
for size in sizes:
    test_name = f"{size[0]}"
    tensor = torch.rand(size[1])
    tensor_const_tests[test_name] = (tensor,)

xfails = {
    "int_r1_st": "MLETORCH-408: Arithmetic ops can't handle scalars first",
    "int_r4_st": "MLETORCH-408: Arithmetic ops can't handle scalars first",
    "float_r1_st": "MLETORCH-408: Arithmetic ops can't handle scalars first",
    "float_r4_st": "MLETORCH-408: Arithmetic ops can't handle scalars first",
}


# ADD FP ------------------------------------------------------
@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_add_tensor_tosa_FP_scalar(test_data):
    """Tests regular add with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](Add(), test_data, aten_op=Add.aten_op)
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_add_tensor_tosa_FP_inplace(test_data):
    """Tests inplace add with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](AddInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_const_tests, xfails=xfails)
def test_add_tensor_tosa_FP_const(test_data):
    """Tests regular add with one scalar input, with one of inputs constant."""
    pipeline = TosaPipelineFP[input_t1](AddConst(), test_data, aten_op=AddConst.aten_op)
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_add_scalar_tosa_FP(test_data):
    """Tests a scalar add with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](
        AddScalar(), test_data, aten_op=AddScalar.aten_op
    )
    pipeline.run()


# ADD INT ------------------------------------------------------
@common.parametrize("test_data", tensor_scalar_tests)
def test_add_tensor_tosa_INT_scalar(test_data):
    """Tests regular add with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](Add(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests)
def test_add_tensor_tosa_INT_inplace(test_data):
    """Tests inplace add with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](AddInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_const_tests)
def test_add_tensor_tosa_INT_const(test_data):
    """Tests regular add with one scalar input, with one of inputs constant."""
    pipeline = TosaPipelineINT[input_t1](
        AddConst(), test_data, aten_op=AddConst.aten_op
    )
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_add_scalar_tosa_INT(test_data):
    """Tests a scalar add with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](AddScalar(), test_data, aten_op=Add.aten_op)
    pipeline.run()


# ADD ETHOS-U ------------------------------------------------------
@pytest.mark.skip(reason="This is tested in test_add_scalar_tosa_INT")
def test_add_scalar_u55_INT():
    pass


@pytest.mark.skip(reason="This is tested in test_add_scalar_tosa_INT")
def test_add_scalar_u85_INT():
    pass


# SUB FP ------------------------------------------------------


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_sub_tensor_tosa_FP_scalar(test_data):
    """Tests regular sub with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](Sub(), test_data, aten_op=Sub.aten_op)
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_sub_tensor_tosa_FP_inplace(test_data):
    """Tests inplace sub with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](SubInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_sub_scalar_tosa_FP(test_data):
    """Tests a scalar sub with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](
        SubScalar(), test_data, aten_op=SubScalar.aten_op
    )
    pipeline.run()


# SUB INT ------------------------------------------------------
@common.parametrize("test_data", tensor_scalar_tests)
def test_sub_tensor_tosa_INT_scalar(test_data):
    """Tests regular sub with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](Sub(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests)
def test_sub_tensor_tosa_INT_inplace(test_data):
    """Tests inplace sub with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](SubInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_sub_scalar_tosa_INT(test_data):
    """Tests a scalar sub with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](SubScalar(), test_data, aten_op=Sub.aten_op)
    pipeline.run()


# SUB ETHOS-U ------------------------------------------------------
@pytest.mark.skip(reason="This is tested in test_sub_scalar_tosa_INT")
def test_sub_scalar_u55_INT():
    pass


@pytest.mark.skip(reason="This is tested in test_sub_scalar_tosa_INT")
def test_sub_scalar_u85_INT():
    pass


# MUL FP ------------------------------------------------------
@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_mul_tensor_tosa_FP_scalar(test_data):
    """Tests regular mul with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](Mul(), test_data, aten_op=Mul.aten_op)
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_mul_tensor_tosa_FP_inplace(test_data):
    """Tests inplace mul with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](MulInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_mul_scalar_tosa_FP(test_data):
    """Tests a scalar mul with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](
        MulScalar(), test_data, aten_op=MulScalar.aten_op
    )
    pipeline.run()


# MUL INT ------------------------------------------------------
@common.parametrize("test_data", tensor_scalar_tests)
def test_mul_tensor_tosa_INT_scalar(test_data):
    """Tests regular mul with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](Mul(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests)
def test_mul_tensor_tosa_INT_inplace(test_data):
    """Tests inplace mul with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](MulInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_mul_scalar_tosa_INT(test_data):
    """Tests a scalar mul with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](MulScalar(), test_data, aten_op=Mul.aten_op)
    pipeline.run()


# MUL ETHOS-U ------------------------------------------------------
@pytest.mark.skip(reason="This is tested in test_mul_scalar_tosa_INT")
def test_mul_scalar_u55_INT():
    pass


@pytest.mark.skip(reason="This is tested in test_mul_scalar_tosa_INT")
def test_mul_scalar_u85_INT():
    pass


# DIV FP ------------------------------------------------------
@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_div_tensor_tosa_FP_scalar(test_data):
    """Tests regular div with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](Div(), test_data, aten_op=Div.aten_op)
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_div_tensor_tosa_FP_inplace(test_data):
    """Tests inplace div with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](DivInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_div_scalar_tosa_FP(test_data):
    """Tests a scalar div with one scalar input."""
    pipeline = TosaPipelineFP[input_t1](
        DivScalar(), test_data, aten_op=DivScalar.aten_op
    )
    pipeline.run()


# DIV INT ------------------------------------------------------
@common.parametrize("test_data", tensor_scalar_tests)
def test_div_tensor_tosa_INT_scalar(test_data):
    """Tests regular div with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](Div(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests)
def test_div_tensor_tosa_INT_inplace(test_data):
    """Tests inplace div with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](DivInplace(), test_data, aten_op=[])
    pipeline.run()


@common.parametrize("test_data", tensor_scalar_tests, xfails=xfails)
def test_div_scalar_tosa_INT(test_data):
    """Tests a scalar div with one scalar input."""
    pipeline = TosaPipelineINT[input_t1](DivScalar(), test_data, aten_op=[])
    pipeline.run()


# DIV ETHOS-U ------------------------------------------------------
@pytest.mark.skip(reason="This is tested in test_div_scalar_tosa_INT")
def test_div_scalar_u55_INT():
    pass


@pytest.mark.skip(reason="This is tested in test_div_scalar_tosa_INT")
def test_div_scalar_u85_INT():
    pass


# SHIFT ETHOS-U ------------------------------------------------------
@pytest.mark.skip(
    reason="integer operations (shift and sub) are not supported on FP profile"
)
def test_bitwise_right_shift_tensor_tosa_FP_inplace():
    pipeline = TosaPipelineFP[input_t1](
        ShiftInplaceSub(),
        (torch.IntTensor(5),),
        aten_op="torch.ops.aten.__rshift__.Scalar",
    )
    pipeline.run()


def test_bitwise_right_shift_tensor_tosa_INT_inplace():
    pipeline = TosaPipelineINT[input_t1](
        ShiftInplaceSub(),
        (torch.IntTensor(5),),
        aten_op="torch.ops.aten.bitwise_right_shift.Tensor",
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
