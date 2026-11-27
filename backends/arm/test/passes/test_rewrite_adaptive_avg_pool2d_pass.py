# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes.rewrite_adaptive_avg_pool2d import (
    RewriteAdaptiveAvgPool2dPass,
)
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.utils import _get_shape_env_from_gm
from torch.export import Dim, export

input_t = Tuple[torch.Tensor]


class AdaptiveAvgPoolUniform(torch.nn.Module):
    def __init__(self, output_size=(4, 4)):
        super().__init__()
        self.output_size = output_size

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)


class AdaptiveAvgPoolLargeStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 32, 32),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))


class AdaptiveAvgPoolIrregular(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 7, 7),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))


class AdaptiveAvgPoolDynamic(torch.nn.Module):
    def __init__(self, output_size=(4, 4)):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)


def _run_dynamic_rewrite(
    dynamic_shapes,
    spec_str: str = "TOSA-1.1+FP+shape",
    output_size=(4, 4),
    example_inputs: input_t | None = None,
):
    module = AdaptiveAvgPoolDynamic(output_size)
    if example_inputs is None:
        example_inputs = (torch.rand(1, 3, 8, 8),)
    ep = export(module, example_inputs, dynamic_shapes=dynamic_shapes)
    edge_model = to_edge(ep)

    shape_env = _get_shape_env_from_gm(edge_model.exported_program().graph_module)
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec_str), shape_env=shape_env
    ):
        result = RewriteAdaptiveAvgPool2dPass().call(
            edge_model.exported_program().graph_module
        )
    return list(result.graph_module.graph.nodes)


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_static_uniform_no_rewrite():
    module = AdaptiveAvgPoolUniform()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[RewriteAdaptiveAvgPool2dPass],
        tosa_version="1.1",
    )
    pipeline.run()


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_static_large_stride_no_rewrite():
    module = AdaptiveAvgPoolLargeStride()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[RewriteAdaptiveAvgPool2dPass],
        tosa_version="1.1",
    )
    pipeline.run()


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_irregular_falls_back():
    module = AdaptiveAvgPoolIrregular()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[RewriteAdaptiveAvgPool2dPass],
        tosa_version="1.1",
    )
    pipeline.run()


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_dynamic_uniform():
    nodes = _run_dynamic_rewrite(
        {
            "x": {
                2: Dim("height", min=1, max=4) * 4,
                3: Dim("width", min=1, max=4) * 4,
            }
        }
    )

    adaptive_node = next(
        n
        for n in nodes
        if n.target == exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default
    )
    permute_nodes = [
        n for n in nodes if n.target == exir_ops.edge.aten.permute_copy.default
    ]
    kernel, stride, pad = adaptive_node.args[3:6]

    assert adaptive_node is not None
    assert len(permute_nodes) == 2
    assert permute_nodes[0].args[1] == list(NHWC_ORDER)
    assert permute_nodes[1].args[1] == list(NHWC_INVERSE_ORDER)
    assert adaptive_node.args[0] is permute_nodes[0]
    assert permute_nodes[1].args[0] is adaptive_node
    assert any(isinstance(v, torch.SymInt) for v in kernel)
    assert any(isinstance(v, torch.SymInt) for v in stride)
    assert pad.name == "tosa_const_shape_default"
    assert pad.target == exir_ops.backend.tosa.CONST_SHAPE.default
    assert pad.args == ([0, 0, 0, 0],)
    assert not any(
        n.target == exir_ops.edge.aten._adaptive_avg_pool2d.default for n in nodes
    )


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_dynamic_asymmetric_uniform():
    nodes = _run_dynamic_rewrite(
        {
            "x": {
                2: Dim("height", min=1, max=4) * 2,
                3: Dim("width", min=1, max=4) * 3,
            }
        },
        output_size=(2, 3),
        example_inputs=(torch.rand(1, 3, 8, 9),),
    )

    adaptive_node = next(
        n
        for n in nodes
        if n.target == exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default
    )
    permute_nodes = [
        n for n in nodes if n.target == exir_ops.edge.aten.permute_copy.default
    ]
    kernel, stride, pad = adaptive_node.args[3:6]

    assert len(permute_nodes) == 2
    assert permute_nodes[0].args[1] == list(NHWC_ORDER)
    assert permute_nodes[1].args[1] == list(NHWC_INVERSE_ORDER)
    assert adaptive_node.args[0] is permute_nodes[0]
    assert permute_nodes[1].args[0] is adaptive_node
    assert all(isinstance(v, torch.SymInt) for v in kernel)
    assert all(isinstance(v, torch.SymInt) for v in stride)
    assert pad.name == "tosa_const_shape_default"
    assert pad.target == exir_ops.backend.tosa.CONST_SHAPE.default
    assert pad.args == ([0, 0, 0, 0],)
    assert not any(
        n.target == exir_ops.edge.aten._adaptive_avg_pool2d.default for n in nodes
    )


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_mixed_dynamic_uniform():
    nodes = _run_dynamic_rewrite(
        {
            "x": {
                2: Dim("height", min=1, max=4) * 4,
            }
        }
    )

    adaptive_node = next(
        n
        for n in nodes
        if n.target == exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default
    )
    permute_nodes = [
        n for n in nodes if n.target == exir_ops.edge.aten.permute_copy.default
    ]
    kernel, stride, pad = adaptive_node.args[3:6]

    assert len(permute_nodes) == 2
    assert permute_nodes[0].args[1] == list(NHWC_ORDER)
    assert permute_nodes[1].args[1] == list(NHWC_INVERSE_ORDER)
    assert adaptive_node.args[0] is permute_nodes[0]
    assert permute_nodes[1].args[0] is adaptive_node
    assert isinstance(kernel[0], torch.SymInt)
    assert kernel[1] == 2
    assert isinstance(stride[0], torch.SymInt)
    assert stride[1] == 2
    assert pad.name == "tosa_const_shape_default"
    assert pad.target == exir_ops.backend.tosa.CONST_SHAPE.default
    assert pad.args == ([0, 0, 0, 0],)
    assert not any(
        n.target == exir_ops.edge.aten._adaptive_avg_pool2d.default for n in nodes
    )


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_dynamic_irregular_falls_back():
    nodes = _run_dynamic_rewrite(
        {
            "x": {
                2: Dim("height", min=4, max=10),
                3: Dim("width", min=4, max=10),
            }
        }
    )

    assert any(
        n.target == exir_ops.edge.aten._adaptive_avg_pool2d.default for n in nodes
    )
    assert not any(
        n.target == exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default for n in nodes
    )


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_none_output_falls_back():
    nodes = _run_dynamic_rewrite(
        {
            "x": {
                2: Dim("height", min=4, max=10),
                3: Dim("width", min=4, max=10),
            }
        },
        output_size=(2, None),
    )

    assert any(
        n.target == exir_ops.edge.aten._adaptive_avg_pool2d.default for n in nodes
    )
    assert not any(
        n.target == exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default for n in nodes
    )


def test_rewrite_adaptive_avg_pool2d_tosa_1_1_without_shape_extension_errors():
    with pytest.raises(
        RuntimeError,
        match=(
            "Dynamic adaptive_avg_pool2d rewrite requires TOSA-1.1 with the shape "
            "extension."
        ),
    ):
        _run_dynamic_rewrite(
            {
                "x": {
                    2: Dim("height", min=1, max=4) * 4,
                    3: Dim("width", min=1, max=4) * 4,
                }
            },
            spec_str="TOSA-1.1+FP",
        )


def test_rewrite_adaptive_avg_pool2d_tosa_1_0_no_rewrite():
    module = AdaptiveAvgPoolUniform()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[RewriteAdaptiveAvgPool2dPass],
        tosa_version="1.0",
    )
    pipeline.run()
