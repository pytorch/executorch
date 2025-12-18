# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

Tensor1 = Tuple[torch.Tensor]


class NegAdd(torch.nn.Module):
    # neg(x) + 1
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_neg_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    ]

    def get_inputs(self) -> Tensor1:
        return (torch.rand(10, 10, 10),)

    def forward(self, x):
        return torch.neg(x) + 1.0


class MinAddZero(torch.nn.Module):
    # min(x, 0) + 1
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_full_like_default",
        "executorch_exir_dialects_edge__ops_aten_minimum_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    ]

    # range [-1, 1]
    def get_inputs(self) -> Tensor1:
        return (torch.rand(10, 10, 10) * 2 - 1,)

    def forward(self, x):
        # We want Tensor-Tensor minimum
        z = torch.full_like(x, 0.0)
        return torch.minimum(x, z) + 1.0


class MaxAddZero(torch.nn.Module):
    # max(x, 0) + 1.0
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_full_like_default",
        "executorch_exir_dialects_edge__ops_aten_maximum_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    ]

    # range [-1, 1]
    def get_inputs(self) -> Tensor1:
        return (torch.rand(10, 10, 10) * 2 - 1,)

    def forward(self, x):
        z = torch.full_like(x, 0.0)
        return torch.maximum(x, z) + 1.0


class AbsAdd(torch.nn.Module):
    # abs(x) + 1.0
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_abs_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    ]

    def get_inputs(self) -> Tensor1:
        return (torch.rand(10, 10, 10),)

    def forward(self, x):
        return torch.abs(x) + 1.0


MODELS = [NegAdd, AbsAdd, MaxAddZero, MinAddZero]


def _build(model_cls):
    m = model_cls()
    return m, m.get_inputs(), model_cls.edge_op_list


@pytest.mark.parametrize("model_cls", MODELS, ids=lambda c: c.__name__)
def test_unary_combos_tosa_FP(model_cls):
    m, inputs, exir = _build(model_cls)
    p = TosaPipelineFP[Tensor1](m, inputs, aten_op=[], exir_op=exir)
    p.run()


@pytest.mark.parametrize("model_cls", MODELS, ids=lambda c: c.__name__)
def test_unary_combos_tosa_INT(model_cls):
    m, inputs, exir = _build(model_cls)
    p = TosaPipelineINT[Tensor1](m, inputs, aten_op=[], exir_op=exir, qtol=1)
    p.run()


@common.XfailIfNoCorstone300
@pytest.mark.parametrize("model_cls", MODELS, ids=lambda c: c.__name__)
def test_unary_combos_u55_INT(model_cls):
    m, inputs, exir = _build(model_cls)
    p = EthosU55PipelineINT[Tensor1](
        m,
        inputs,
        aten_ops=[],
        exir_ops=exir,
    )
    p.run()


@common.XfailIfNoCorstone320
@pytest.mark.parametrize("model_cls", MODELS, ids=lambda c: c.__name__)
def test_unary_combos_u85_INT(model_cls):
    m, inputs, exir = _build(model_cls)
    p = EthosU85PipelineINT[Tensor1](
        m,
        inputs,
        aten_ops=[],
        exir_ops=exir,
    )
    p.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize("model_cls", MODELS, ids=lambda c: c.__name__)
def test_unary_combos_vgf_quant(model_cls):
    m, inputs, exir = _build(model_cls)
    p = VgfPipeline[Tensor1](
        m,
        inputs,
        aten_op=[],
        exir_op=exir,
        quantize=True,
    )
    p.run()
