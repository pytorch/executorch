# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from copy import copy
from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineINT,
    VgfPipeline,
)

input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class BitwiseBinary(torch.nn.Module):
    test_data: dict[input_t2] = {
        "zeros": lambda: (
            torch.zeros(1, 10, 10, 10, dtype=torch.int32),
            torch.zeros(1, 10, 10, 10, dtype=torch.int32),
        ),
        "ones": lambda: (
            torch.ones(10, 10, 10, dtype=torch.int8),
            torch.ones(10, 10, 10, dtype=torch.int8),
        ),
        "pattern_int8": lambda: (
            0xAA * torch.ones(1, 2, 2, 2, dtype=torch.int8),
            0xCC * torch.ones(1, 2, 2, 2, dtype=torch.int8),
        ),
        "pattern_int16": lambda: (
            0xAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int16),
            0xCCCC * torch.ones(1, 2, 2, 2, dtype=torch.int16),
        ),
        "pattern_int32": lambda: (
            0xAAAAAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int32),
            0xCCCCCCCC * torch.ones(1, 2, 2, 2, dtype=torch.int32),
        ),
        "pattern_bool": lambda: (
            torch.tensor([True, False, True], dtype=torch.bool),
            torch.tensor([True, True, False], dtype=torch.bool),
        ),
        "rand_rank2": lambda: (
            torch.randint(-128, 127, (10, 10), dtype=torch.int8),
            torch.randint(-128, 127, (10, 10), dtype=torch.int8),
        ),
        "rand_rank4": lambda: (
            torch.randint(-128, -127, (1, 10, 10, 10), dtype=torch.int8),
            torch.randint(-128, 127, (1, 10, 10, 10), dtype=torch.int8),
        ),
    }

    test_data_u85 = copy(test_data)
    del test_data_u85["zeros"]


class BitwiseBinaryScalar(torch.nn.Module):
    test_data = {
        "zeros": lambda: (torch.zeros(1, 10, 10, 10, dtype=torch.int32), 0),
        "ones_int8": lambda: (torch.ones(10, 10, 10, dtype=torch.int8), 1),
        "pattern_int8": lambda: (0xAA * torch.ones(1, 2, 2, 2, dtype=torch.int8), 0x77),
        "pattern_int16": lambda: (
            0xAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int16),
            0x7777,
        ),
        "pattern_int32": lambda: (
            0xAAAAAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int32),
            0x77777777,
        ),
        "rand_rank2": lambda: (torch.randint(-128, 127, (10, 10), dtype=torch.int8), 5),
        "rand_rank4": lambda: (
            torch.randint(-128, 127, (1, 10, 10, 10), dtype=torch.int8),
            -7,
        ),
    }

    test_data_u85 = copy(test_data)
    del test_data_u85["zeros"]


class And(BitwiseBinary):
    aten_op = "torch.ops.aten.bitwise_and.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_and_Tensor"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.bitwise_and(tensor2)


class Xor(BitwiseBinary):
    aten_op = "torch.ops.aten.bitwise_xor.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_xor_Tensor"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.bitwise_xor(tensor2)


class Or(BitwiseBinary):
    aten_op = "torch.ops.aten.bitwise_or.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_or_Tensor"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.bitwise_or(tensor2)


class AndScalar(BitwiseBinaryScalar):
    # Tensor because it gets converted from Scalar -> Tensor in lowering
    aten_op = "torch.ops.aten.bitwise_and.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_and_Tensor"
    exir_op_scalar = "executorch_exir_dialects_edge__ops_aten_bitwise_and_Scalar"

    def forward(self, tensor: torch.Tensor, scalar: int):
        return tensor.bitwise_and(scalar)


class XorScalar(BitwiseBinaryScalar):
    # Tensor because it gets converted from Scalar -> Tensor in lowering
    aten_op = "torch.ops.aten.bitwise_xor.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_xor_Tensor"
    exir_op_scalar = "executorch_exir_dialects_edge__ops_aten_bitwise_xor_Scalar"

    def forward(self, tensor: torch.Tensor, scalar: int):
        return tensor.bitwise_xor(scalar)


class OrScalar(BitwiseBinaryScalar):
    # Tensor because it gets converted from Scalar -> Tensor in lowering
    aten_op = "torch.ops.aten.bitwise_or.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_or_Tensor"
    exir_op_scalar = "executorch_exir_dialects_edge__ops_aten_bitwise_or_Scalar"

    def forward(self, tensor: torch.Tensor, scalar: int):
        return tensor.bitwise_or(scalar)


#########
## AND ##
#########


@common.parametrize("test_data", And().test_data)
def test_bitwise_and_tensor_tosa_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        And(),
        test_data(),
        {And.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", AndScalar.test_data)
def test_bitwise_and_scalar_tosa_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        AndScalar(),
        test_data(),
        {AndScalar.exir_op_scalar: 1},
    )
    pipeline.run()


@common.parametrize("test_data", And().test_data)
def test_bitwise_and_tensor_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        And(),
        test_data(),
        And().aten_op,
        And().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", AndScalar.test_data)
def test_bitwise_and_scalar_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        AndScalar(),
        test_data(),
        AndScalar.aten_op,
        AndScalar.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", And().test_data)
def test_bitwise_and_tensor_u55_INT(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t2](
        And(),
        test_data(),
        {And.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", AndScalar.test_data)
def test_bitwise_and_scalar_u55_INT(test_data: input_t2):
    # There will be one full op which will be delegated.
    num_delegates = 1
    num_exir = 0
    pipeline = OpNotSupportedPipeline[input_t2](
        AndScalar(),
        test_data(),
        {
            AndScalar.exir_op: 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": num_exir,
        },
        num_delegates,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", AndScalar.test_data_u85)
@common.XfailIfNoCorstone320
def test_bitwise_and_scalar_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        AndScalar(),
        test_data(),
        AndScalar.aten_op,
        AndScalar.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", And().test_data_u85)
@common.XfailIfNoCorstone320
def test_bitwise_and_tensor_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        And(),
        test_data(),
        And().aten_op,
        And().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", And().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_and_tensor_vgf_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        And(),
        test_data(),
        {And.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", AndScalar().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_and_scalar_vgf_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        AndScalar(),
        test_data(),
        {AndScalar.exir_op_scalar: 1},
    )
    pipeline.run()


@common.parametrize("test_data", And().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_and_tensor_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        And(),
        test_data(),
        And().aten_op,
        And().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", AndScalar().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_and_scalar_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        AndScalar(),
        test_data(),
        AndScalar().aten_op,
        AndScalar().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


#########
## XOR ##
#########


@common.parametrize("test_data", Xor().test_data)
def test_bitwise_xor_tensor_tosa_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        Xor(),
        test_data(),
        {Xor.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", XorScalar.test_data)
def test_bitwise_xor_scalar_tosa_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        XorScalar(),
        test_data(),
        {XorScalar.exir_op_scalar: 1},
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
def test_bitwise_xor_tensor_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", XorScalar.test_data)
def test_bitwise_xor_scalar_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        XorScalar(),
        test_data(),
        XorScalar.aten_op,
        XorScalar.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
def test_bitwise_xor_tensor_u55_INT(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t2](
        Xor(),
        test_data(),
        {Xor().exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", XorScalar.test_data)
def test_bitwise_xor_scalar_u55_INT(test_data: input_t2):
    # There will be one full op which will be delegated.
    num_delegates = 1
    num_exir = 0
    pipeline = OpNotSupportedPipeline[input_t2](
        XorScalar(),
        test_data(),
        {
            XorScalar.exir_op: 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": num_exir,
        },
        num_delegates,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data_u85)
@common.XfailIfNoCorstone320
def test_bitwise_xor_tensor_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", XorScalar.test_data_u85)
@common.XfailIfNoCorstone320
def test_bitwise_xor_scalar_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        XorScalar(),
        test_data(),
        XorScalar.aten_op,
        XorScalar.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_xor_tensor_vgf_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        Xor(),
        test_data(),
        {Xor.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", XorScalar().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_xor_scalar_vgf_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        XorScalar(),
        test_data(),
        {XorScalar.exir_op_scalar: 1},
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_xor_tensor_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", XorScalar().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_xor_scalar_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        XorScalar(),
        test_data(),
        XorScalar().aten_op,
        XorScalar().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


########
## OR ##
########


@common.parametrize("test_data", Or().test_data)
def test_bitwise_or_tensor_tosa_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        Or(),
        test_data(),
        {Or.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", OrScalar.test_data)
def test_bitwise_or_scalar_tosa_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        OrScalar(),
        test_data(),
        {OrScalar.exir_op_scalar: 1},
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
def test_bitwise_or_tensor_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", OrScalar.test_data)
def test_bitwise_or_scalar_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        OrScalar(),
        test_data(),
        OrScalar.aten_op,
        OrScalar.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
def test_bitwise_or_tensor_u55_INT(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t2](
        Or(),
        test_data(),
        {Or().exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", OrScalar.test_data)
def test_bitwise_or_scalar_u55_INT(test_data: input_t2):
    # There will be one full op which will be delegated.
    num_delegates = 1
    num_exir = 0
    pipeline = OpNotSupportedPipeline[input_t2](
        OrScalar(),
        test_data(),
        {
            OrScalar.exir_op: 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": num_exir,
        },
        num_delegates,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data_u85)
@common.XfailIfNoCorstone320
def test_bitwise_or_tensor_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", OrScalar.test_data_u85)
@common.XfailIfNoCorstone320
def test_bitwise_or_scalar_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        OrScalar(),
        test_data(),
        OrScalar.aten_op,
        OrScalar.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_or_tensor_vgf_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        Or(),
        test_data(),
        {Or.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", OrScalar().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_or_scalar_vgf_FP(test_data: input_t2):
    pipeline = OpNotSupportedPipeline[input_t2](
        OrScalar(),
        test_data(),
        {OrScalar.exir_op_scalar: 1},
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_or_tensor_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", OrScalar().test_data)
@common.SkipIfNoModelConverter
def test_bitwise_or_scalar_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        OrScalar(),
        test_data(),
        OrScalar().aten_op,
        OrScalar().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@pytest.mark.xfail(
    reason="MLBEDSW-11029: Fatal Python floating point error in Vela for rank 4 bitwse ops with int32 dtype."
)
def test_bitwise_or_tensor_u85_INT_zeros():
    raise RuntimeError(
        "Dummy test to xfail mark u85 zeros test case since running the actual test causes a fatal crash."
    )


@pytest.mark.xfail(
    reason="MLBEDSW-11029: Fatal Python floating point error in Vela for rank 4 bitwse ops with int32 dtype."
)
def test_bitwise_and_tensor_u85_INT_zeros():
    raise RuntimeError(
        "Dummy test to xfail mark u85 zeros test case since running the actual test causes a fatal crash."
    )


@pytest.mark.xfail(
    reason="MLBEDSW-11029: Fatal Python floating point error in Vela for rank 4 bitwse ops with int32 dtype."
)
def test_bitwise_xor_tensor_u85_INT_zeros():
    raise RuntimeError(
        "Dummy test to xfail mark u85 zeros test case since running the actual test causes a fatal crash."
    )
