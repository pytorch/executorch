# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.insert_dynamic_padding import (
    InsertDynamicPaddingPass,
)
from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.utils import _get_shape_env_from_gm
from torch.export import Dim, export
from torch.fx import GraphModule


def _assert_inserted_padding(
    graph_module: GraphModule,
    target_op,
    zero_spatial_padding: list[int],
    expected_full_padding_len: int,
) -> None:
    nodes = graph_module.graph.nodes
    conv_node = next(n for n in nodes if n.target == target_op)
    # ``Graph.materialize_symints`` may have lifted SymInt pad entries into
    # FX nodes; accept either plain ints or Nodes with the SymInt in meta['val'].
    assert (
        all(
            (
                (p == exp)
                if isinstance(p, int)
                else (p.meta["val"] == exp) if isinstance(p, torch.fx.Node) else False
            )
            for p, exp in zip(conv_node.args[4], zero_spatial_padding, strict=True)
        )
        or conv_node.args[4] == zero_spatial_padding
    )

    padding_node = next(
        n for n in nodes if n.target == exir_ops.backend.tosa.PAD.default
    )
    padding_shape_node = padding_node.args[1]
    assert padding_shape_node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default

    n_padding, spatial_padding, c_padding = padding_shape_node.args[0]
    assert n_padding.meta["val"] == [0, 0]
    assert c_padding.meta["val"] == [0, 0]

    pad_list = padding_shape_node.meta["val"]
    pad_list_vals = [
        p.meta["val"] if isinstance(p, torch.fx.Node) else p for p in pad_list
    ]
    assert len(pad_list_vals) == expected_full_padding_len
    assert pad_list_vals[:2] == [0, 0]
    assert pad_list_vals[-2:] == [0, 0]
    # For static graphs spatial_padding is a CONST_SHAPE node; for dynamic
    # graphs (RewriteConvPass materialized) it is an immutable_list of Nodes/ints.
    if hasattr(spatial_padding, "target"):
        assert spatial_padding.target == exir_ops.backend.tosa.CONST_SHAPE.default
        spatial_padding_value = spatial_padding.meta["val"]
        if isinstance(spatial_padding_value, (list, tuple)):
            spatial_vals = [
                p.meta["val"] if isinstance(p, torch.fx.Node) else p
                for p in spatial_padding_value
            ]
            assert pad_list_vals[2:-2] == spatial_vals
        else:
            assert pad_list_vals[2:-2] == spatial_padding_value
    else:
        # Dynamic case: spatial_padding is the original pad list (possibly Nodes)
        spatial_vals = [
            p.meta["val"] if isinstance(p, torch.fx.Node) else p
            for p in spatial_padding
        ]
        assert pad_list_vals[2:-2] == spatial_vals


class ConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=2, stride=3, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 16, kernel_size=2, stride=3, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _export_conv_model(conv_cls, example_inputs, dynamic_shapes):
    model = conv_cls()
    ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    edge_model = to_edge(ep)
    shape_env = _get_shape_env_from_gm(edge_model.exported_program().graph_module)
    return edge_model, shape_env


def test_insert_dynamic_padding():
    edge_model, shape_env = _export_conv_model(
        ConvModule,
        (torch.randn(1, 3, 8, 8),),
        dynamic_shapes={
            "x": {2: Dim("height", min=4, max=10), 3: Dim("width", min=4, max=10)}
        },
    )
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ):
        edge_model = edge_model.transform(
            [RewriteConvPass(edge_model.exported_program())]
        )
        # Verify pad was materialized into FX nodes (Graph.materialize_symints).
        nodes = edge_model.exported_program().graph.nodes
        conv_node = next(
            n for n in nodes if n.target == exir_ops.backend.tosa.CONV2D.default
        )
        initial_padding = conv_node.args[4]
        assert any(isinstance(p, torch.fx.Node) for p in initial_padding)
        initial_padding_vals = [
            p.meta["val"] if isinstance(p, torch.fx.Node) else p
            for p in initial_padding
        ]

        edge_model = edge_model.transform([InsertDynamicPaddingPass()])
        graph_module = edge_model.exported_program().graph_module

        conv_node = next(
            n
            for n in graph_module.graph.nodes
            if n.target == exir_ops.backend.tosa.CONV2D.default
        )
        assert conv_node.args[4] == [0, 0, 0, 0]
        padding_node = next(
            n
            for n in graph_module.graph.nodes
            if n.target == exir_ops.backend.tosa.PAD.default
        )
        assert padding_node is not None
        pad_list = padding_node.args[1].meta["val"]
        assert len(pad_list) == 8
        assert pad_list[:2] == [0, 0]
        assert pad_list[2:6] == initial_padding_vals
        assert pad_list[6:] == [0, 0]

        # Cross-check the shared _assert_inserted_padding helper against the same graph.
        _assert_inserted_padding(
            graph_module,
            exir_ops.backend.tosa.CONV2D.default,
            zero_spatial_padding=[0, 0, 0, 0],
            expected_full_padding_len=8,
        )


def test_insert_dynamic_padding_conv3d():
    edge_model, shape_env = _export_conv_model(
        Conv3dModule,
        (torch.randn(1, 3, 8, 8, 8),),
        dynamic_shapes={
            "x": {
                2: Dim("depth", min=4, max=10),
                3: Dim("height", min=4, max=10),
                4: Dim("width", min=4, max=10),
            }
        },
    )
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ):
        edge_model = edge_model.transform(
            [RewriteConvPass(edge_model.exported_program())]
        )
        nodes = edge_model.exported_program().graph.nodes
        conv_node = next(
            n for n in nodes if n.target == exir_ops.backend.tosa.CONV3D.default
        )
        initial_padding = conv_node.args[4]
        assert any(isinstance(p, torch.fx.Node) for p in initial_padding)
        initial_padding_vals = [
            p.meta["val"] if isinstance(p, torch.fx.Node) else p
            for p in initial_padding
        ]

        edge_model = edge_model.transform([InsertDynamicPaddingPass()])
        graph_module = edge_model.exported_program().graph_module

        conv_node = next(
            n
            for n in graph_module.graph.nodes
            if n.target == exir_ops.backend.tosa.CONV3D.default
        )
        assert conv_node.args[4] == [0, 0, 0, 0, 0, 0]
        padding_node = next(
            n
            for n in graph_module.graph.nodes
            if n.target == exir_ops.backend.tosa.PAD.default
        )
        assert padding_node is not None
        pad_list = padding_node.args[1].meta["val"]
        assert len(pad_list) == 10
        assert pad_list[:2] == [0, 0]
        assert pad_list[2:8] == initial_padding_vals
        assert pad_list[8:] == [0, 0]

        _assert_inserted_padding(
            graph_module,
            exir_ops.backend.tosa.CONV3D.default,
            zero_spatial_padding=[0, 0, 0, 0, 0, 0],
            expected_full_padding_len=10,
        )
