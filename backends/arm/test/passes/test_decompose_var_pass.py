# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.decompose_var_pass import DecomposeVarPass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class VarDim(torch.nn.Module):
    """
    Basic variance model using torch.Tensor.var function.
    """

    def __init__(self, keepdim):
        super(VarDim, self).__init__()
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.var(dim=-1, keepdim=self.keepdim)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


class VarCorrection(torch.nn.Module):
    """
    Basic variance model using torch.var function.
    """

    def __init__(self, keepdim):
        super(VarCorrection, self).__init__()
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.var(x, -1, keepdim=self.keepdim)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


modules: Dict[str, ModuleWithInputs] = {
    "vardim_keepdim": VarDim(True),
    "vardim_no_keepdim": VarDim(False),
    "varcorrection_keepdim": VarCorrection(True),
    "varcorrection_no_keepdim": VarCorrection(False),
}


@common.parametrize("module", modules)
def test_decompose_var_tosa_FP(module: ModuleWithInputs) -> None:
    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_var_correction": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_full_default",
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList",
            "executorch_exir_dialects_edge__ops_aten_mean_dim",
            "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 2,
            "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
            "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": 1,
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 1,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_aten_var_correction"],
        pass_list=[DecomposeVarPass],
    )
    pipeline.run()
