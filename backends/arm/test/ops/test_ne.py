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


input_t = Tuple[torch.Tensor]


class NotEqual(torch.nn.Module):
    aten_op_Tensor = "torch.ops.aten.ne.Tensor"
    aten_op_Scalar = "torch.ops.aten.ne.Scalar"
    decomposed_ops = ["torch.ops.aten.eq.Tensor", "torch.ops.aten.logical_not.default"]
    decomposed_exir_ops = [
        "executorch_exir_dialects_edge__ops_aten_eq_Tensor",
        "executorch_exir_dialects_edge__ops_aten_logical_not_default",
    ]
    exir_op = "executorch_exir_dialects_edge__ops_aten_ne_Tensor"

    def __init__(self, input, other):
        super().__init__()
        self.input_ = input
        self.other_ = other

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor | int | float,
    ):
        return input_ != other_

    def get_inputs(self):
        return (self.input_, self.other_)


op_ne_tensor_rank1_ones = NotEqual(
    torch.ones(5),
    torch.ones(5),
)
op_ne_tensor_rank2_rand = NotEqual(
    torch.rand(4, 5),
    torch.rand(1, 5),
)
op_ne_tensor_rank3_randn = NotEqual(
    torch.randn(10, 5, 2),
    torch.randn(10, 5, 2),
)
op_ne_tensor_rank4_randn = NotEqual(
    torch.randn(3, 2, 2, 2),
    torch.randn(3, 2, 2, 2),
)

op_ne_scalar_rank1_ones = NotEqual(torch.ones(5), 1.0)
op_ne_scalar_rank2_rand = NotEqual(torch.rand(4, 5), 0.2)
op_ne_scalar_rank3_randn = NotEqual(torch.randn(10, 5, 2), -0.1)
op_ne_scalar_rank4_randn = NotEqual(torch.randn(3, 2, 2, 2), 0.3)
op_ne_scalar_rank4_randn_1batch = NotEqual(torch.randn(1, 2, 2, 2), 0.3)

test_data_tensor = {
    "ne_tensor_rank1_ones": op_ne_tensor_rank1_ones,
    "ne_tensor_rank2_rand": op_ne_tensor_rank2_rand,
    "ne_tensor_rank3_randn": op_ne_tensor_rank3_randn,
    "ne_tensor_rank4_randn": op_ne_tensor_rank4_randn,
}

test_data_scalar = {
    "ne_scalar_rank1_ones": op_ne_scalar_rank1_ones,
    "ne_scalar_rank2_rand": op_ne_scalar_rank2_rand,
    "ne_scalar_rank3_randn": op_ne_scalar_rank3_randn,
    "ne_scalar_rank4_randn": op_ne_scalar_rank4_randn,
    "ne_scalar_rank4_randn_1batch": op_ne_scalar_rank4_randn_1batch,
}


@common.parametrize("test_module", test_data_tensor)
def test_ne_tensor_tosa_FP(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module, test_module.get_inputs(), NotEqual.aten_op_Tensor, NotEqual.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_ne_scalar_tosa_FP(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module,
        test_module.get_inputs(),
        NotEqual.aten_op_Scalar,
        NotEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
def test_ne_tensor_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module, test_module.get_inputs(), NotEqual.decomposed_ops, NotEqual.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_ne_scalar_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module, test_module.get_inputs(), NotEqual.decomposed_ops, NotEqual.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone300
def test_ne_tensor_u55_INT(test_module):
    # EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        {
            NotEqual.decomposed_exir_ops[0]: 1,
            NotEqual.decomposed_exir_ops[1]: 1,
        },
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone300
def test_ne_scalar_u55_INT(test_module):
    # Not equal (ne) is decomposed into the TOSA ops EQUAL and LOGICAL_NOT, both of
    # which are unsupported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        {
            NotEqual.decomposed_exir_ops[0]: 1,
            NotEqual.decomposed_exir_ops[1]: 1,
        },
        quantize=True,
        u55_subset=True,
        n_expected_delegates=1,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_tensor,
    xfails={
        "ne_tensor_rank4_randn": "MLETORCH-517: Batch size > 1 not fully supported",
    },
    strict=False,
)
@common.XfailIfNoCorstone320
def test_ne_tensor_u85_INT(test_module):
    pipeline = EthosU85PipelineINT[input_t](
        test_module,
        test_module.get_inputs(),
        NotEqual.decomposed_ops,
        NotEqual.decomposed_exir_ops,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_scalar,
    xfails={
        "ne_scalar_rank4_randn": "MLETORCH-517: Batch size > 1 not fully supported",
        "ne_scalar_rank4_randn_1batch": "MLETORCH-847: Boolean ne result unstable on U85",
    },
    strict=False,
)
@common.XfailIfNoCorstone320
def test_ne_scalar_u85_INT(test_module):
    pipeline = EthosU85PipelineINT[input_t](
        test_module,
        test_module.get_inputs(),
        NotEqual.decomposed_ops,
        NotEqual.decomposed_exir_ops,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.SkipIfNoModelConverter
def test_ne_tensor_vgf_FP(test_module):
    pipeline = VgfPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        NotEqual.aten_op_Tensor,
        NotEqual.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.SkipIfNoModelConverter
def test_ne_tensor_vgf_INT(test_module):
    pipeline = VgfPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        NotEqual.decomposed_ops,
        NotEqual.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.SkipIfNoModelConverter
def test_ne_scalar_vgf_FP(test_module):
    pipeline = VgfPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        NotEqual.aten_op_Scalar,
        NotEqual.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.SkipIfNoModelConverter
def test_ne_scalar_vgf_INT(test_module):
    pipeline = VgfPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        NotEqual.decomposed_ops,
        NotEqual.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
