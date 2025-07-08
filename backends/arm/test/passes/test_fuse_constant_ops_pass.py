# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Tuple

import torch
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOT,
    FuseConstantArgsPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class FuseParameter(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_full_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_addmm_default": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
    }
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_add_Tensor": 1}
    ops_not_after_pass = [
        "executorch_exir_dialects_edge__ops_aten_full_default",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
    ]

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.fc(torch.ones(1)) + x


class FuseBuffer(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
    }
    ops_not_after_pass = [
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
    ]

    def forward(self, x: torch.Tensor):
        return (x + 1) * 2


class FuseLiftedTensor(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_select_copy_int": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
    }
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_add_Tensor": 1}
    ops_not_after_pass = ["executorch_exir_dialects_edge__ops_aten_select_copy_int"]

    def __init__(
        self,
    ):
        super().__init__()
        self.lifted_tensor = torch.rand(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sliced = self.lifted_tensor[0]
        return operator.add(sliced, x)


modules = {
    "fuse_parameter": FuseParameter(),
    "fuse_buffer": FuseBuffer(),
    "fuse_const_tensor": FuseLiftedTensor(),
}


@common.parametrize("module", modules)
def test_fuse_const_ops_tosa_MI(module: torch.nn.Module):
    pipeline = PassPipeline[input_t](
        module=module,
        test_data=(torch.rand(1),),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        passes_with_exported_program=[ComputeConstantOpsAOT, FuseConstantArgsPass],
    )
    pipeline.run()


@common.parametrize("module", modules)
def test_fuse_const_ops_tosa_BI(module: torch.nn.Module):
    pipeline = PassPipeline[input_t](
        module,
        (torch.rand(10, 10),),
        quantize=True,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[ComputeConstantOpsAOT, FuseConstantArgsPass],
    )
    pipeline.run()
