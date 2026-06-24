# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.insert_dynamic_padding import (
    InsertDynamicPaddingPass,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


SPEC = TosaSpecification.create_from_string("TOSA-1.1+FP+shape")


def _build_conv_graph(
    target_op,
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    padding: list[int],
    stride: list[int],
    dilation: list[int],
) -> GraphModule:
    with TosaLoweringContext(SPEC):
        builder = GraphBuilder()
        input_tensor = builder.placeholder("input", torch.randn(input_shape))
        weight = builder.placeholder("weight", torch.randn(weight_shape))
        bias = builder.placeholder("bias", torch.randn(weight_shape[0]))
        padding_shape = builder.call_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, (padding,)
        )
        padding_shape.node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE
        conv = builder.call_operator(
            target_op,
            (input_tensor, weight, bias, stride, padding_shape, dilation),
        )
        builder.output([conv])
        return ExportPass().call(builder.get_graph_module()).graph_module


def _run_insert_dynamic_padding(graph_module: GraphModule) -> GraphModule:
    with TosaLoweringContext(SPEC):
        result = InsertDynamicPaddingPass()(graph_module)
    assert isinstance(result, PassResult)
    return result.graph_module


def _assert_inserted_padding(
    graph_module: GraphModule,
    target_op,
    zero_spatial_padding: list[int],
    expected_full_padding_len: int,
) -> None:
    nodes = graph_module.graph.nodes
    conv_node = next(n for n in nodes if n.target == target_op)
    assert conv_node.args[4] == zero_spatial_padding

    padding_node = next(
        n for n in nodes if n.target == exir_ops.backend.tosa.PAD.default
    )
    padding_shape_node = padding_node.args[1]
    assert padding_shape_node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default

    n_padding, spatial_padding, c_padding = padding_shape_node.args[0]
    assert n_padding.meta["val"] == [0, 0]
    assert spatial_padding.target == exir_ops.backend.tosa.CONST_SHAPE.default
    assert c_padding.meta["val"] == [0, 0]

    pad_list = padding_shape_node.meta["val"]
    spatial_padding_value = spatial_padding.meta["val"]
    assert len(pad_list) == expected_full_padding_len
    assert pad_list[:2] == [0, 0]
    assert pad_list[2:-2] == spatial_padding_value
    assert pad_list[-2:] == [0, 0]


def test_insert_dynamic_padding():
    graph_module = _build_conv_graph(
        exir_ops.backend.tosa.CONV2D.default,
        input_shape=(1, 8, 8, 3),
        weight_shape=(16, 2, 2, 3),
        padding=[2, 2, 2, 2],
        stride=[3, 3],
        dilation=[1, 1],
    )

    graph_module = _run_insert_dynamic_padding(graph_module)

    _assert_inserted_padding(
        graph_module,
        exir_ops.backend.tosa.CONV2D.default,
        zero_spatial_padding=[0, 0, 0, 0],
        expected_full_padding_len=8,
    )


def test_insert_dynamic_padding_conv3d():
    graph_module = _build_conv_graph(
        exir_ops.backend.tosa.CONV3D.default,
        input_shape=(1, 8, 8, 8, 3),
        weight_shape=(16, 2, 2, 2, 3),
        padding=[2, 2, 2, 2, 2, 2],
        stride=[3, 3, 3],
        dilation=[1, 1, 1],
    )

    graph_module = _run_insert_dynamic_padding(graph_module)

    _assert_inserted_padding(
        graph_module,
        exir_ops.backend.tosa.CONV3D.default,
        zero_spatial_padding=[0, 0, 0, 0, 0, 0],
        expected_full_padding_len=10,
    )
