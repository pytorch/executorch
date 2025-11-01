# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import cast, ClassVar, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOT,
    FuseConstantArgsPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor]  # Input x
input_t2 = Tuple[torch.Tensor, torch.Tensor]


class ModuleWithFuseAttrs(Protocol):
    ops_before_pass: Dict[str, int]
    ops_after_pass: Dict[str, int]
    ops_not_after_pass: list[str]

    def get_inputs(self) -> input_t: ...


class FuseParameter(torch.nn.Module):
    ops_before_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_full_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_addmm_default": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
    }
    ops_after_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1
    }
    ops_not_after_pass: ClassVar[list[str]] = [
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.ones(1)) + x


class FuseBuffer(torch.nn.Module):
    ops_before_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
    }
    ops_after_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
    }
    ops_not_after_pass: ClassVar[list[str]] = [
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
    ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1) * 2


class FuseLiftedTensor(torch.nn.Module):
    ops_before_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_select_copy_int": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
    }
    ops_after_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1
    }
    ops_not_after_pass: ClassVar[list[str]] = [
        "executorch_exir_dialects_edge__ops_aten_select_copy_int"
    ]

    def __init__(
        self,
    ):
        super().__init__()
        self.lifted_tensor = torch.rand(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sliced = self.lifted_tensor[0]
        return operator.add(sliced, x)


class CatConst(torch.nn.Module):
    ops_before_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    }
    ops_after_pass: ClassVar[Dict[str, int]] = {
        "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    }
    ops_not_after_pass: ClassVar[list[str]] = []

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat((a, b), dim=0)


class LinearConst(torch.nn.Module):
    """A linear layer that can be computed AOT"""

    def __init__(self, in_out_features: int = 3, bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(in_out_features, in_out_features, bias=bias)
        self.example_input = torch.rand(in_out_features, in_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.full_like(x, 1.0)
        return self.linear(y) + x

    def get_example_input(self) -> torch.Tensor:
        return self.example_input


modules: Dict[str, ModuleWithFuseAttrs] = {
    "fuse_parameter": cast(ModuleWithFuseAttrs, FuseParameter()),
    "fuse_buffer": cast(ModuleWithFuseAttrs, FuseBuffer()),
    "fuse_const_tensor": cast(ModuleWithFuseAttrs, FuseLiftedTensor()),
}

cat_module: Dict[str, ModuleWithFuseAttrs] = {
    "fuse_cat": cast(ModuleWithFuseAttrs, CatConst()),
}


@common.parametrize("module", modules)
def test_fuse_const_ops_tosa_FP(module: ModuleWithFuseAttrs) -> None:
    pipeline = PassPipeline[input_t](
        module=cast(torch.nn.Module, module),
        test_data=(torch.rand(1),),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        passes_with_exported_program=[ComputeConstantOpsAOT, FuseConstantArgsPass],
    )
    pipeline.run()


@common.parametrize("module", modules)
def test_fuse_const_ops_tosa_INT(module: ModuleWithFuseAttrs) -> None:
    pipeline = PassPipeline[input_t](
        cast(torch.nn.Module, module),
        (torch.rand(10, 10),),
        quantize=True,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[ComputeConstantOpsAOT, FuseConstantArgsPass],
    )
    pipeline.run()


@common.parametrize("module", cat_module)
def test_fuse_const_ops_tosa_BI_cat(module: ModuleWithFuseAttrs) -> None:
    pipeline = PassPipeline[input_t2](
        cast(torch.nn.Module, module),
        (torch.rand(3), torch.rand(2)),
        quantize=True,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[ComputeConstantOpsAOT, FuseConstantArgsPass],
    )
    pipeline.run()


def test_linear_const_tosa_FP():
    model = LinearConst()
    example_input = model.get_example_input()
    pipeline = TosaPipelineFP[input_t](
        model,
        (example_input,),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


def test_linear_const_tosa_INT():
    model = LinearConst()
    example_input = model.get_example_input()
    pipeline = TosaPipelineINT[input_t](
        model,
        (example_input,),
        aten_op=[],
        exir_op=[],
        per_channel_quantization=False,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
