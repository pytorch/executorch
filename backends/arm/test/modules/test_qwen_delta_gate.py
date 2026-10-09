# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineINT,
    VgfPipeline,
)


class QwenDeltaGate(torch.nn.Module):
    a_log: torch.Tensor
    dt_bias: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("a_log", torch.log(torch.tensor([0.5, 1.0, 1.5, 2.0])))
        self.register_buffer("dt_bias", torch.tensor([-2.0, -0.5, 0.5, 2.0]))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        g = -self.a_log.exp() * F.softplus(a + self.dt_bias)
        return g.exp()


@common.XfailIfNoCorstone300
def test_qwen_gate_u55_int():
    values = torch.linspace(-4.0, 4.0, 32).reshape(1, 8, 4)
    pipeline = EthosU55PipelineINT(
        QwenDeltaGate(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
        run_on_fvp=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_qwen_gate_u85_int8():
    values = torch.linspace(-4.0, 4.0, 32).reshape(1, 8, 4)
    pipeline = EthosU85PipelineINT(
        QwenDeltaGate(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
        run_on_fvp=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_qwen_gate_u85_int16():
    values = torch.linspace(-4.0, 4.0, 32).reshape(1, 8, 4)
    pipeline = EthosU85PipelineINT(
        QwenDeltaGate(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
        run_on_fvp=True,
        a16w8_quantization=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_qwen_gate_vgf_quant():
    values = torch.linspace(-4.0, 4.0, 32).reshape(1, 8, 4)
    pipeline = VgfPipeline(
        QwenDeltaGate(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
        run_on_vulkan_runtime=True,
    )
    pipeline.run()


def test_qwen_gate_tosa_int():
    values = torch.linspace(-4.0, 4.0, 32).reshape(1, 8, 4)
    pipeline = TosaPipelineINT(
        QwenDeltaGate(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
    )
    pipeline.count_tosa_ops({"TABLE": 2})
    pipeline.run()


def test_qwen_gate_tosa_a16w8():
    values = torch.linspace(-4.0, 4.0, 32).reshape(1, 8, 4)
    pipeline = TosaPipelineINT(
        QwenDeltaGate(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
        tosa_extensions=["int16"],
    )
    pipeline.count_tosa_ops({"TABLE": 2})
    pipeline.run()
