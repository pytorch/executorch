# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.rewrite_avg_pool2d_pass import RewriteAvgPool2dPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops

input_t = Tuple[torch.Tensor]


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class AvgPool2dWithStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


class AvgPool2dWithoutStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, kernel_size=3)


class AvgPool2dListKernel(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, kernel_size=[2, 3])


class AvgPool2dScalarPadding(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1)


class AvgPool2dWithEmptyStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, kernel_size=[2, 3], stride=[])


modules: Dict[str, ModuleWithInputs] = {
    "avg_pool2d_with_stride": AvgPool2dWithStride(),
    "avg_pool2d_without_stride": AvgPool2dWithoutStride(),
    "avg_pool2d_list_kernel": AvgPool2dListKernel(),
}


@common.parametrize("module", modules)
def test_rewrite_avg_pool2d_tosa(module: ModuleWithInputs) -> None:
    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_default": 1,
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
        },
        pass_list=[RewriteAvgPool2dPass],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # Cannot run aten graph with tosa dialect ops
    pipeline.run()


def _get_tosa_avg_pool2d_node(
    pipeline: PassPipeline[input_t],
) -> torch.fx.Node:
    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    graph_module = exported_program.graph_module

    tosa_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.AVG_POOL2D.default
    ]
    assert len(tosa_nodes) == 1
    return tosa_nodes[0]


def test_rewrite_avg_pool2d_tosa_empty_stride_uses_kernel_size() -> None:
    module = AvgPool2dWithEmptyStride()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_default": 1,
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
        },
        pass_list=[RewriteAvgPool2dPass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    tosa_node = _get_tosa_avg_pool2d_node(pipeline)
    assert tosa_node.args[4] == [2, 3]
