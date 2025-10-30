# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y
input_t1 = Tuple[torch.Tensor]  # Input x


class LogicalBinary(torch.nn.Module):
    test_data: dict[input_t2] = {
        "rank1": lambda: (
            torch.tensor([True, True, False, False], dtype=torch.bool),
            torch.tensor([True, False, True, False], dtype=torch.bool),
        ),
        "rand_rank2": lambda: (
            torch.randint(0, 2, (10, 10), dtype=torch.bool),
            torch.randint(0, 2, (10, 10), dtype=torch.bool),
        ),
        "rand_rank3": lambda: (
            torch.randint(0, 2, (10, 10, 10), dtype=torch.bool),
            torch.randint(0, 2, (10, 10, 10), dtype=torch.bool),
        ),
        "rand_rank4": lambda: (
            torch.randint(0, 2, (1, 10, 10, 10), dtype=torch.bool),
            torch.randint(0, 2, (1, 10, 10, 10), dtype=torch.bool),
        ),
    }


class And(LogicalBinary):
    aten_op = "torch.ops.aten.logical_and.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_logical_and_default"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.logical_and(tensor2)


class Xor(LogicalBinary):
    aten_op = "torch.ops.aten.logical_xor.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_logical_xor_default"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.logical_xor(tensor2)


class Or(LogicalBinary):
    aten_op = "torch.ops.aten.logical_or.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_logical_or_default"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.logical_or(tensor2)


class Not(torch.nn.Module):
    test_data: dict[input_t1] = {
        "rank1": lambda: (torch.tensor([True, True, False, False], dtype=torch.bool),),
        "rand_rank2": lambda: (torch.randint(0, 2, (10, 10), dtype=torch.bool),),
        "rand_rank3": lambda: (torch.randint(0, 2, (10, 10, 10), dtype=torch.bool),),
        "rand_rank4": lambda: (torch.randint(0, 2, (1, 10, 10, 10), dtype=torch.bool),),
    }

    aten_op = "torch.ops.aten.logical_not.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_logical_not_default"

    def forward(self, tensor: torch.Tensor):
        return torch.logical_not(tensor)


#################
## logical_and ##
#################


@common.parametrize("test_data", And().test_data)
def test_logical_and_tosa_FP(test_data: input_t2):
    pipeline = TosaPipelineFP[input_t2](
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
def test_logical_and_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        And(),
        test_data(),
        And().aten_op,
        And().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", And().test_data)
def test_logical_and_u55_INT_not_delegated(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t2](
        And(),
        test_data(),
        {And().exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", And().test_data)
@common.XfailIfNoCorstone320
def test_logical_and_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        And(),
        test_data(),
        And().aten_op,
        And().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", And().test_data)
@common.SkipIfNoModelConverter
def test_logical_and_vgf_FP(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        And(),
        test_data(),
        And().aten_op,
        And().exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", And().test_data)
@common.SkipIfNoModelConverter
def test_logical_and_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        And(),
        test_data(),
        And().aten_op,
        And().exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


#################
## logical_xor ##
#################


@common.parametrize("test_data", Xor().test_data)
def test_logical_xor_tosa_FP(test_data: input_t2):
    pipeline = TosaPipelineFP[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
def test_logical_xor_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
def test_logical_xor_u55_INT_not_delegated(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t2](
        Xor(),
        test_data(),
        {Xor().exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
@common.XfailIfNoCorstone320
def test_logical_xor_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
@common.SkipIfNoModelConverter
def test_logical_xor_vgf_FP(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Xor().test_data)
@common.SkipIfNoModelConverter
def test_logical_xor_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Xor(),
        test_data(),
        Xor().aten_op,
        Xor().exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


################
## logical_or ##
################


@common.parametrize("test_data", Or().test_data)
def test_logical_or_tosa_FP(test_data: input_t2):
    pipeline = TosaPipelineFP[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
def test_logical_or_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
def test_logical_or_u55_INT_not_delegated(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t2](
        Or(),
        test_data(),
        {Or().exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
@common.XfailIfNoCorstone320
def test_logical_or_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
@common.SkipIfNoModelConverter
def test_logical_or_vgf_FP(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Or().test_data)
@common.SkipIfNoModelConverter
def test_logical_or_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Or(),
        test_data(),
        Or().aten_op,
        Or().exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


#################
## logical_not ##
#################


@common.parametrize("test_data", Not().test_data)
def test_logical_not_tosa_FP(test_data: input_t2):
    pipeline = TosaPipelineFP[input_t2](
        Not(),
        test_data(),
        Not().aten_op,
        Not().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", Not().test_data)
def test_logical_not_tosa_INT(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](
        Not(),
        test_data(),
        Not().aten_op,
        Not().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", Not().test_data)
def test_logical_not_u55_INT_not_delegated(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t2](
        Not(),
        test_data(),
        {Not().exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", Not().test_data)
@common.XfailIfNoCorstone320
def test_logical_not_u85_INT(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        Not(),
        test_data(),
        Not().aten_op,
        Not().exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", Not().test_data)
@common.SkipIfNoModelConverter
def test_logical_not_vgf_FP(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Not(),
        test_data(),
        Not().aten_op,
        Not().exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Not().test_data)
@common.SkipIfNoModelConverter
def test_logical_not_vgf_INT(test_data: input_t2):
    pipeline = VgfPipeline[input_t2](
        Not(),
        test_data(),
        Not().aten_op,
        Not().exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
