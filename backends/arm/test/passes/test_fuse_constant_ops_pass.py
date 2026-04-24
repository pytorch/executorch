# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import cast, ClassVar, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
    FuseConstantArgsPass,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.test.harness.stages import StageType

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


class QuantizedCatConstantBuffers(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "horizontal_ramp",
            torch.tensor(
                [
                    [
                        [
                            [-95, -32, 32, 95, 0],
                            [-95, -32, 32, 95, 0],
                            [-95, -32, 32, 95, 0],
                            [-95, -32, 32, 95, 0],
                        ]
                    ]
                ],
                dtype=torch.int8,
            ),
        )
        self.register_buffer(
            "vertical_ramp",
            torch.tensor(
                [
                    [
                        [
                            [-95, -95, -95, -95, -95],
                            [-32, -32, -32, -32, -32],
                            [32, 32, 32, 32, 32],
                            [95, 95, 95, 95, 95],
                        ]
                    ]
                ],
                dtype=torch.int8,
            ),
        )

    def forward(self) -> torch.Tensor:
        return torch.cat(
            (
                cast(torch.Tensor, self.horizontal_ramp),
                cast(torch.Tensor, self.vertical_ramp),
            ),
            dim=1,
        )


modules: Dict[str, ModuleWithFuseAttrs] = {
    "fuse_parameter": cast(ModuleWithFuseAttrs, FuseParameter()),
    "fuse_buffer": cast(ModuleWithFuseAttrs, FuseBuffer()),
    "fuse_const_tensor": cast(ModuleWithFuseAttrs, FuseLiftedTensor()),
}

cat_module: Dict[str, ModuleWithFuseAttrs] = {
    "fuse_cat": cast(ModuleWithFuseAttrs, CatConst()),
}


@common.parametrize("module", modules)
def test_fuse_constant_args_tosa_FP(module: ModuleWithFuseAttrs) -> None:
    pipeline = PassPipeline[input_t](
        module=cast(torch.nn.Module, module),
        test_data=(torch.rand(1),),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        passes_with_exported_program=[
            ComputeConstantOpsAOTPass,
            FuseConstantArgsPass,
        ],
    )
    pipeline.run()


@common.parametrize("module", modules)
def test_fuse_constant_args_tosa_INT(module: ModuleWithFuseAttrs) -> None:
    pipeline = PassPipeline[input_t](
        cast(torch.nn.Module, module),
        (torch.rand(10, 10),),
        quantize=True,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[
            ComputeConstantOpsAOTPass,
            FuseConstantArgsPass,
        ],
    )
    pipeline.run()


@common.parametrize("module", cat_module)
def test_fuse_constant_args_tosa_INT_cat(module: ModuleWithFuseAttrs) -> None:
    pipeline = PassPipeline[input_t2](
        cast(torch.nn.Module, module),
        (torch.rand(3), torch.rand(2)),
        quantize=True,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[
            ComputeConstantOpsAOTPass,
            FuseConstantArgsPass,
        ],
    )
    pipeline.run()


def test_fuse_constant_args_tosa_INT_cat_uses_top_level_arg_qparams() -> None:
    qargs = QuantArgs(
        scale=1.0 / 127.0,
        zp=0,
        qmin=-127,
        qmax=127,
        dtype=torch.int8,
    )
    module = QuantizedCatConstantBuffers()
    compile_spec = common.get_tosa_compile_spec(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    )
    tester = ArmTester(module, example_inputs=(), compile_spec=compile_spec)
    tester.export().to_edge()
    exported_program = tester.get_artifact(StageType.TO_EDGE).exported_program()

    cat_node = next(
        node
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    )
    cat_node.meta["input_qparams"] = {0: qargs}
    cat_node.meta["output_qparams"] = {0: qargs}

    pass_result = FuseConstantArgsPass(exported_program).call(
        exported_program.graph_module
    )

    assert list(exported_program.state_dict) == ["aten_cat_default_fused_const"]
    torch.testing.assert_close(
        exported_program.state_dict["aten_cat_default_fused_const"],
        torch.cat(
            (
                cast(torch.Tensor, module.horizontal_ramp),
                cast(torch.Tensor, module.vertical_ramp),
            ),
            dim=1,
        ),
    )
    assert [
        node.name
        for node in pass_result.graph_module.graph.nodes
        if node.op == "placeholder"
    ] == ["aten_cat_default_fused_const"]
