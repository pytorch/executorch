# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import pytest

import torch
from executorch.backends.arm._passes import (
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t1 = Tuple[torch.Tensor]  # Input x
input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


#####################################################
## Test arange(dtype=int64) -> arange(dtype=int32) ##
#####################################################


class ArangeDefaultIncrementViewLessThan(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return (torch.arange(10, dtype=torch.int64) + 1).view(-1, 1) < x

    test_data = {
        "randint": (
            torch.randint(
                0,
                10,
                (1,),
                dtype=torch.int32,
            ),
        ),
    }


@common.parametrize("test_data", ArangeDefaultIncrementViewLessThan.test_data)
def test_convert_arange_default_int64_dtype_to_int32_pass_tosa_FP(test_data: input_t1):
    module = ArangeDefaultIncrementViewLessThan()
    aten_ops_checks = [
        "torch.ops.aten.lt.Tensor",
        "torch.ops.aten.view.default",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
    ]
    pipeline = TosaPipelineFP[input_t1](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
        transform_passes=[ConvertInt64ConstOpsToInt32Pass()],
    )
    pipeline.run()


@common.parametrize("test_data", ArangeDefaultIncrementViewLessThan.test_data)
def test_convert_arange_default_int64_dtype_to_int32_pass_tosa_INT(test_data: input_t1):
    module = ArangeDefaultIncrementViewLessThan()
    aten_ops_checks = [
        "torch.ops.aten.lt.Tensor",
        "torch.ops.aten.view.default",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
    ]
    pipeline = TosaPipelineINT[input_t1](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


class ArangeStartIncrementViewLessThan(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return (torch.arange(0, 10, dtype=torch.int64) + 1).view(-1, 1) < x

    test_data = {
        "randint": (
            torch.randint(
                0,
                10,
                (1,),
                dtype=torch.int32,
            ),
        ),
    }


@common.parametrize("test_data", ArangeStartIncrementViewLessThan.test_data)
def test_convert_arange_start_int64_dtype_to_int32_pass_tosa_FP(test_data: input_t1):
    module = ArangeStartIncrementViewLessThan()
    aten_ops_checks = [
        "torch.ops.aten.lt.Tensor",
        "torch.ops.aten.view.default",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
    ]
    pipeline = TosaPipelineFP[input_t1](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
        transform_passes=[ConvertInt64ConstOpsToInt32Pass()],
    )
    pipeline.run()


@common.parametrize("test_data", ArangeStartIncrementViewLessThan.test_data)
def test_convert_arange_start_int64_dtype_to_int32_pass_tosa_INT(test_data: input_t1):
    module = ArangeStartIncrementViewLessThan()
    aten_ops_checks = [
        "torch.ops.aten.lt.Tensor",
        "torch.ops.aten.view.default",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
    ]
    pipeline = TosaPipelineINT[input_t1](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


class ArangeStartStepIncrementViewLessThan(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return (torch.arange(0, 10, 2, dtype=torch.int64) + 1).view(-1, 1) < x

    test_data = {
        "randint": (
            torch.randint(
                0,
                10,
                (1,),
                dtype=torch.int32,
            ),
        ),
    }


@common.parametrize("test_data", ArangeStartStepIncrementViewLessThan.test_data)
def test_convert_arange_start_step_int64_dtype_to_int32_pass_tosa_FP(
    test_data: input_t1,
):
    module = ArangeStartStepIncrementViewLessThan()
    aten_ops_checks = [
        "torch.ops.aten.lt.Tensor",
        "torch.ops.aten.view.default",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
    ]
    pipeline = TosaPipelineFP[input_t1](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
        transform_passes=[ConvertInt64ConstOpsToInt32Pass()],
    )
    pipeline.run()


@common.parametrize("test_data", ArangeStartStepIncrementViewLessThan.test_data)
def test_convert_arange_start_step_int64_dtype_to_int32_pass_tosa_INT(
    test_data: input_t1,
):
    module = ArangeStartStepIncrementViewLessThan()
    aten_ops_checks = [
        "torch.ops.aten.lt.Tensor",
        "torch.ops.aten.view.default",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
    ]
    pipeline = TosaPipelineINT[input_t1](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


#########################################################
## Test arange(dtype=None) -> arange(dtype=None/int32) ##
#########################################################


class ArangeAddDtypeNone(torch.nn.Module):
    aten_op: str = "torch.ops.aten.arange.start_step"
    exir_op: str = "executorch_exir_dialects_edge__ops_aten_arange_start_step"

    def __init__(self, start: float, stop: float, step: float):
        super().__init__()
        self.args = (start, stop, step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arange(*self.args) + x

    test_data = {
        "int64": (lambda: (torch.randn(10, 1),), (0, 10, 1)),
        "float32_start": (lambda: (torch.randn(10, 1),), (0.0, 10, 1)),
        "float32_stop": (lambda: (torch.randn(10, 1),), (0, 10.0, 1)),
        "float32_step": (lambda: (torch.randn(10, 1),), (0, 10, 1.0)),
        "int64_bool_0": (lambda: (torch.randn(10, 1),), (False, True, True)),
        "int64_bool_1": (lambda: (torch.randn(10, 1),), (False, True, True * 10)),
        "float32_bool_0": (lambda: (torch.randn(10, 1),), (0.0, True, True)),
        "float32_bool_1": (lambda: (torch.randn(10, 1),), (False, True, True * 10.0)),
    }


@common.parametrize("test_data", ArangeAddDtypeNone.test_data)
def test_arange_dtype_none_tosa_FP(test_data):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t1](
        ArangeAddDtypeNone(*init_data),
        input_data(),
        ArangeAddDtypeNone.aten_op,
        ArangeAddDtypeNone.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAddDtypeNone.test_data)
def test_arange_dtype_none_tosa_INT(test_data):
    input_data, init_data = test_data
    pipeline = TosaPipelineINT[input_t1](
        ArangeAddDtypeNone(*init_data),
        input_data(),
        ArangeAddDtypeNone.aten_op,
        ArangeAddDtypeNone.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


#################################################
## Test full(dtype=int64) -> full(dtype=int32) ##
#################################################


class FullIncrementViewMulXLessThanY(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (
            (
                torch.full(
                    (
                        1,
                        3,
                        5,
                    ),
                    10,
                    dtype=torch.int64,
                )
                + 1
            ).view(-1, 1)
            * x
        ) < y

    test_data = {
        "randint": (
            torch.randint(
                0,
                10,
                (1,),
                dtype=torch.int32,
            ),
            torch.randint(
                0,
                10,
                (1,),
                dtype=torch.int32,
            ),
        ),
    }


@common.parametrize("test_data", FullIncrementViewMulXLessThanY.test_data)
def test_convert_full_int64_dtype_to_int32_pass_tosa_FP(test_data: input_t1):
    """
    There are four int64 placeholders in the original graph:
    1. _lifted_tensor_constant0: 1
    2. x
    3. y
    Ideally, after applying ConvertInt64ConstOpsToInt32Pass to convert the aten.full from int64 to int32,
    the int32 type should propagate throughout the graph, and no int64 values should remain.
    However, due to unexpected retracing behavior, a cast from int32 → int64 for x was reintroducedh.

    Applying ConvertInt64OutputOpsToInt32Pass afterward resolves this issue,
    removing the int64 cast and producing a fully delegated int32 graph.
    """
    module = FullIncrementViewMulXLessThanY()
    aten_ops_checks = [
        "torch.ops.aten.full.default",
        "torch.ops.aten.add.Tensor",
        "torch.ops.aten.view.default",
        "torch.ops.aten.mul.Tensor",
        "torch.ops.aten.lt.Tensor",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_full_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
    ]
    pipeline = TosaPipelineFP[input_t2](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
        transform_passes=[
            ConvertInt64ConstOpsToInt32Pass(),
            ConvertInt64OutputOpsToInt32Pass(),
        ],
    )
    pipeline.run()


@common.parametrize("test_data", FullIncrementViewMulXLessThanY.test_data)
def test_convert_full_int64_dtype_to_int32_pass_tosa_INT(test_data: input_t1):
    """
    For INT profile, _lifted_tensor_constant0 is still int64 after applying ConvertInt64ConstOpsToInt32Pass().
    And an int64->int32 cast is inserted at the beginning of the graph.
    TODO: Explore why _lifted_tensor_constant0 is handled in different ways in FP and INT profile.
        Find a way to optimize out the int64->int32 cast.
    """
    module = FullIncrementViewMulXLessThanY()
    aten_ops_checks = [
        "torch.ops.aten.full.default",
        "torch.ops.aten.add.Tensor",
        "torch.ops.aten.view.default",
        "torch.ops.aten.mul.Tensor",
        "torch.ops.aten.lt.Tensor",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_full_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
    ]
    pipeline = TosaPipelineINT[input_t2](
        module,
        test_data,
        aten_ops_checks,
        exir_ops_checks,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


class RejectFullIncrementViewMulXLessThanY(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (
            (
                torch.full(
                    (
                        1,
                        3,
                        5,
                    ),
                    torch.iinfo(torch.int32).max + 1,
                    dtype=torch.int64,
                )
                + 1
            ).view(-1, 1)
            * x
        ) < y

    test_data = {
        "randint": (
            torch.randint(
                0,
                10,
                (1,),
                dtype=torch.int32,
            ),
            torch.randint(
                0,
                10,
                (1,),
                dtype=torch.int32,
            ),
        ),
    }


@common.parametrize("test_data", RejectFullIncrementViewMulXLessThanY.test_data)
@pytest.mark.xfail(
    reason="MLETORCH-1254: Add operator support check for aten.arange and aten.full"
)
def test_reject_convert_full_int64_dtype_to_int32_pass_tosa_FP(test_data: input_t1):
    module = RejectFullIncrementViewMulXLessThanY()
    aten_ops_checks = [
        "torch.ops.aten.full.default",
        "torch.ops.aten.add.Tensor",
        "torch.ops.aten.view.default",
        "torch.ops.aten.mul.Tensor",
        "torch.ops.aten.lt.Tensor",
    ]
    pipeline = TosaPipelineFP[input_t2](
        module,
        test_data,
        aten_ops_checks,
        exir_op=[],
        transform_passes=[
            ConvertInt64ConstOpsToInt32Pass(),
            ConvertInt64OutputOpsToInt32Pass(),
        ],
    )
    pipeline.run()


#####################################################
## Test full(dtype=None) -> full(dtype=None/int32) ##
#####################################################


class AddConstFullDtypeNone(torch.nn.Module):
    # Input + a full with constant value.
    exir_op = "executorch_exir_dialects_edge__ops_aten_full_default"

    def __init__(self, size: tuple, fill_value: Union[bool, float, int]):
        super().__init__()
        self.args = (size, fill_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(*self.args) + x

    test_data = {
        "int64": (lambda: (torch.randn(1),), ((1, 2, 3), 10)),
        "float32": (lambda: (torch.randn(1),), ((1, 2, 3), 10.0)),
    }

    test_data_bool = {
        "bool": (lambda: (torch.randn(1),), ((1, 2, 3), True)),
    }


@common.parametrize("test_data", AddConstFullDtypeNone.test_data)
def test_full_dtype_none_tosa_FP(test_data):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t1](
        AddConstFullDtypeNone(*init_data),
        input_data(),
        aten_op=[],
        exir_op=AddConstFullDtypeNone.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", AddConstFullDtypeNone.test_data_bool)
def test_full_dtype_none_tosa_FP_bool(test_data):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t1](
        AddConstFullDtypeNone(*init_data),
        input_data(),
        aten_op=[],
        exir_op=AddConstFullDtypeNone.exir_op,
    )
    pipeline.change_args(
        "check_count.exir",
        {"torch.ops.higher_order.executorch_call_delegate": 2},
    )
    pipeline.run()


@common.parametrize(
    "test_data", AddConstFullDtypeNone.test_data | AddConstFullDtypeNone.test_data_bool
)
def test_full_dtype_none_tosa_INT(test_data):
    input_data, init_data = test_data
    pipeline = TosaPipelineINT[input_t1](
        AddConstFullDtypeNone(*init_data),
        input_data(),
        aten_op=[],
        exir_op=AddConstFullDtypeNone.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
