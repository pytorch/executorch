# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.remove_getitem_pass import RemoveGetItemPass
from executorch.backends.arm._passes.rewrite_max_pool2d_pass import RewriteMaxPool2dPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.harness.stages import StageType
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.utils import _get_shape_env_from_gm
from torch.export import Dim, export

input_t = Tuple[torch.Tensor]


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class MaxPool2dWithStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)


class MaxPool2dWithoutStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 9, 9),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=3)


class MaxPool2dListKernel(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 9),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=[2, 3])


class MaxPool2dWithEmptyStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=[2, 2], stride=[])


class MaxPool2dDynamic(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x, kernel_size=3, stride=2, padding=1, ceil_mode=True
        )


class MaxPool2dDynamicAdaptive(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x, kernel_size=3, stride=2, padding=1, ceil_mode=False
        )


modules: Dict[str, ModuleWithInputs] = {
    "max_pool2d_with_stride": MaxPool2dWithStride(),
    "max_pool2d_without_stride": MaxPool2dWithoutStride(),
    "max_pool2d_list_kernel": MaxPool2dListKernel(),
}


@common.parametrize("module", modules)
def test_rewrite_max_pool2d_tosa(module: ModuleWithInputs) -> None:
    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_MAX_POOL2D_default": 1,
        },
        pass_list=[RemoveGetItemPass, RewriteMaxPool2dPass],
    )
    pipeline.run()


def _get_tosa_max_pool2d_node(
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
        and node.target == exir_ops.backend.tosa.MAX_POOL2D.default
    ]
    assert len(tosa_nodes) == 1
    return tosa_nodes[0]


def test_rewrite_max_pool2d_tosa_empty_stride_uses_kernel_size() -> None:
    module = MaxPool2dWithEmptyStride()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_MAX_POOL2D_default": 1,
        },
        pass_list=[RemoveGetItemPass, RewriteMaxPool2dPass],
    )
    pipeline.run()

    tosa_node = _get_tosa_max_pool2d_node(pipeline)
    assert tosa_node.args[2] == [2, 2]


def test_rewrite_max_pool2d_tosa_dynamic_shape() -> None:
    module = MaxPool2dDynamic()
    example_inputs = (torch.rand(1, 3, 8, 8),)
    ep = export(
        module,
        example_inputs,
        dynamic_shapes={
            "x": {
                2: Dim("height", min=2, max=8) * 2,
                3: Dim("width", min=2, max=8) * 2,
            }
        },
    )
    edge_model = to_edge(ep)
    shape_env = _get_shape_env_from_gm(edge_model.exported_program().graph_module)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        edge_model = edge_model.transform([RemoveGetItemPass(), RewriteMaxPool2dPass()])

    nodes = list(edge_model.exported_program().graph.nodes)
    assert not any(n.target == exir_ops.edge.aten.max_pool2d.default for n in nodes)
    assert any(n.target == exir_ops.backend.tosa.MAX_POOL2D.default for n in nodes)


def test_rewrite_max_pool2d_tosa_dynamic_shape_adjusts_adaptive_trailing_pad() -> None:
    module = MaxPool2dDynamicAdaptive()
    example_inputs = (torch.rand(1, 3, 8, 8),)
    ep = export(
        module,
        example_inputs,
        dynamic_shapes={
            "x": {
                2: Dim("height", min=2, max=8) * 2,
                3: Dim("width", min=2, max=8) * 2,
            }
        },
    )
    edge_model = to_edge(ep)
    shape_env = _get_shape_env_from_gm(edge_model.exported_program().graph_module)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        edge_model = edge_model.transform([RemoveGetItemPass(), RewriteMaxPool2dPass()])

    nodes = list(edge_model.exported_program().graph.nodes)
    adaptive_nodes = [
        n
        for n in nodes
        if n.target == exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default
    ]
    assert len(adaptive_nodes) == 1
    assert not any(n.target == exir_ops.backend.tosa.MAX_POOL2D.default for n in nodes)

    pad_node = adaptive_nodes[0].args[3]
    if isinstance(pad_node, torch.fx.Node):
        assert pad_node.target == exir_ops.backend.tosa.CONST_SHAPE.default
        assert pad_node.args == ([1, 0, 1, 0],)
    else:
        assert list(pad_node) == [1, 0, 1, 0]
