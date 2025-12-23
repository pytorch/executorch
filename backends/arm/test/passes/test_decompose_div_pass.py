# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.decompose_div_pass import DecomposeDivPass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class Div(torch.nn.Module):
    """
    Basic div model using torch.div
    """

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(x, 2)


class DivTensor(torch.nn.Module):
    """
    Basic div model using torch.Tensor.div
    """

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.div(2)


modules: Dict[str, ModuleWithInputs] = {"div_basic": Div(), "div_tensor": DivTensor()}


@common.parametrize("module", modules)
def test_decompose_div_tosa_FP(module: ModuleWithInputs) -> None:
    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_reciprocal_default",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_reciprocal_default": 1,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_aten_div_Tensor"],
        pass_list=[DecomposeDivPass],
    )
    pipeline.run()
