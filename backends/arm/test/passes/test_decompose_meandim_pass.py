# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch

from executorch.backends.arm._passes.decompose_meandim_pass import DecomposeMeanDimPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    TosaPipelineINT,
)

from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export

input_t = Tuple[torch.Tensor]  # Input x


class ModuleWithMeanAttrs(Protocol):
    ops_after_pass: Dict[str, int]
    ops_not_after_pass: list[str]
    u55_ops_after_pass: Dict[str, int]
    u55_ops_not_after_pass: list[str]

    def get_inputs(self) -> input_t: ...


class MeanDim(torch.nn.Module):
    """Basic mean model using torch.mean with keepdim = True."""

    ops_after_pass = u55_ops_after_pass = {
        "torch.ops.aten.sum.dim_IntList": 2,
        "torch.ops.aten.mul.Tensor": 1,
    }

    ops_not_after_pass = u55_ops_not_after_pass = [
        "torch.ops.aten.reshape.default",
        "torch.ops.aten.avg_pool2d.default",
        "torch.ops.aten.mean.dim",
    ]

    def __init__(self):
        super(MeanDim, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, (0, 1), True)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4, 4, 4),)


class MeanDimTensor(torch.nn.Module):
    """Basic mean model using torch.Tensor.mean with keepdim = False."""

    ops_after_pass = {
        "torch.ops.aten.sum.dim_IntList": 2,
        "torch.ops.aten.mul.Tensor": 1,
        "torch.ops.aten.reshape.default": 1,
    }

    ops_not_after_pass = [
        "torch.ops.aten.mean.dim",
        "torch.ops.aten.avg_pool2d.default",
    ]

    u55_ops_after_pass = {
        "torch.ops.aten.sum.dim_IntList": 2,
        "torch.ops.aten.mul.Tensor": 1,
        "torch.ops.aten.reshape.default": 1,
    }

    u55_ops_not_after_pass = [
        "torch.ops.aten.avg_pool2d.default",
        "torch.ops.aten.mean.dim",
    ]

    def __init__(self):
        super(MeanDimTensor, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean((0, 2), False)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4, 4, 4),)


modules: Dict[str, ModuleWithMeanAttrs] = {
    "meandim_basic": MeanDim(),
    "meandim_tensor": MeanDimTensor(),
}


@common.parametrize("module", modules)
def test_decompose_mean_dim_tosa_INT(module: ModuleWithMeanAttrs) -> None:
    # Decompose meandim_pass requires initiating the pas with args, which is not supported
    # by RunPasses in the arm_tester -> PassPipeline cannot be used.
    nn_module = cast(torch.nn.Module, module)
    pipeline = TosaPipelineINT[input_t](nn_module, module.get_inputs(), [])
    pipeline.pop_stage("check_not.exir")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("to_executorch")
    pipeline.pop_stage("check.aten")
    pipeline.add_stage_after("quantize", pipeline.tester.check, module.ops_after_pass)
    pipeline.add_stage_after(
        "quantize", pipeline.tester.check_not, module.ops_not_after_pass
    )

    pipeline.dump_artifact("quantize")
    pipeline.run()


@common.parametrize("module", modules)
def test_decompose_mean_dim_u55_INT(module: ModuleWithMeanAttrs) -> None:
    # Decompose meandim_pass requires initiating the pas with args, which is not supported
    # by RunPasses in the arm_tester -> PassPipeline cannot be used.
    nn_module = cast(torch.nn.Module, module)
    pipeline = EthosU55PipelineINT[input_t](
        nn_module, module.get_inputs(), [], run_on_fvp=False
    )
    pipeline.pop_stage("check_not.exir")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("to_executorch")
    pipeline.pop_stage("check.aten")
    pipeline.add_stage_after(
        "quantize", pipeline.tester.check, module.u55_ops_after_pass
    )
    pipeline.add_stage_after(
        "quantize", pipeline.tester.check_not, module.u55_ops_not_after_pass
    )

    pipeline.dump_artifact("quantize")
    pipeline.run()


def test_decompose_mean_dim_materializes_intermediate_dynamic_shapes() -> None:
    class MeanRank3(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean((0,), False)

    class MeanRank5(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean((2,), False)

    cases = (
        (
            MeanRank3(),
            (torch.rand(2, 3, 4),),
            {"x": {1: Dim("height", min=1, max=8)}},
        ),
        (
            MeanRank5(),
            (torch.rand(2, 3, 4, 5, 6),),
            {"x": {1: Dim("height", min=1, max=8)}},
        ),
    )
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+shape")

    for module, inputs, dynamic_shapes in cases:
        edge_model = to_edge(export(module, inputs, dynamic_shapes=dynamic_shapes))
        graph_module = edge_model.exported_program().graph_module
        result = DecomposeMeanDimPass(graph_module, tosa_spec)(graph_module)
        assert result is not None

        view_nodes = [
            node
            for node in result.graph_module.graph.nodes
            if node.target == exir_ops.edge.aten.view_copy.default
        ]
        assert view_nodes
        assert all(
            not any(isinstance(dim, torch.SymInt) for dim in node.args[1])
            for node in view_nodes
        )
