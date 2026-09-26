# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.remove_getitem_pass import RemoveGetItemPass
from executorch.backends.arm._passes.size_adjust_input_pass import (
    _greater_than,
    SizeAdjustInputPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.capture._config import EdgeCompileConfig
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export
from torch.export.exported_program import _get_shape_env


class ConvModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TransposeConvModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=2,
            output_padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _needs_truncation(input_length, kernel_size, stride, padding):
    return _greater_than((input_length + 2 * padding - kernel_size) % stride, padding)


def test_size_adjust_input_static_conv2d():
    kernel_size, stride, padding = 3, 3, 1
    model = ConvModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 9, 9),)
    ep = export(model, example_inputs)
    edge_model = to_edge(ep)
    # Verify that input needs to be truncated by checking that the remainder is not 0 before transformation, and is 0 after transformation
    input_shape = example_inputs[0].shape
    assert _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should need truncation before transformation"
    assert _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should need truncation before transformation"
    edge_model = edge_model.transform([SizeAdjustInputPass()])
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.aten.convolution.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    assert not _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should not need truncation after transformation"
    assert not _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should not need truncation after transformation"


def test_size_adjust_input_static_conv_no_adjustment_needed():
    kernel_size, stride, padding = 3, 1, 1
    model = ConvModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 9, 9),)
    ep = export(model, example_inputs)
    edge_model = to_edge(ep)
    input_shape = example_inputs[0].shape
    assert not _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should not need truncation before transformation"
    assert not _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should not need truncation before transformation"
    edge_model = edge_model.transform([SizeAdjustInputPass()])
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.aten.convolution.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    assert not _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should not need truncation after transformation"
    assert not _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should not need truncation after transformation"
    slice_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.edge.aten.slice_copy.Tensor
    ]
    assert (
        len(slice_nodes) == 0
    ), "No slice nodes should be inserted when no adjustment is needed"


def test_size_adjust_input_skips_transpose_conv2d() -> None:
    model = TransposeConvModule()
    example_inputs = (torch.randn(1, 3, 16, 16),)
    edge_model = to_edge(export(model, example_inputs))
    edge_model = edge_model.transform([SizeAdjustInputPass()])
    gm = edge_model.exported_program().graph_module

    conv_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.aten.convolution.default
    )
    input_node = conv_node.args[0]
    assert input_node.meta["val"].shape == example_inputs[0].shape

    slice_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.edge.aten.slice_copy.Tensor
    ]
    assert len(slice_nodes) == 0


def test_size_adjust_input_dynamic_conv2d():
    kernel_size, stride, padding = 3, 3, 1
    model = ConvModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 14, 15),)
    ep = export(
        model,
        example_inputs,
        strict=True,
        dynamic_shapes={
            "x": {2: Dim("height", min=1, max=16), 3: Dim("width", min=1, max=16)}
        },
    )
    edge_model = to_edge(ep)
    # Verify that input needs to be truncated by checking that the remainder is not 0 before transformation, and is 0 after transformation
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.aten.convolution.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    shape_env = _get_shape_env(gm)
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        assert _needs_truncation(
            input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input height should need truncation before transformation"
        assert _needs_truncation(
            input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input width should need truncation before transformation"
        edge_model = edge_model.transform([SizeAdjustInputPass()])
        slice_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.aten.slice_copy.Tensor
        ]
        assert (
            len(slice_nodes) == 2
        ), "Two slice nodes should be inserted when adjustment is needed"


def test_size_adjust_input_dynamic_conv_no_adjustment_needed():
    kernel_size, stride, padding = 3, 1, 1
    model = ConvModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 9, 9),)
    ep = export(
        model,
        example_inputs,
        dynamic_shapes={
            "x": {2: Dim("height", min=2, max=6) * 3, 3: Dim("width", min=2, max=6) * 3}
        },
    )
    edge_model = to_edge(ep)
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.aten.convolution.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    gm = edge_model.exported_program().graph_module
    shape_env = _get_shape_env(gm)
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        assert not _needs_truncation(
            input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input height should not need truncation before transformation"
        assert not _needs_truncation(
            input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input width should not need truncation before transformation"
        edge_model = edge_model.transform([SizeAdjustInputPass()])
        gm = edge_model.exported_program().graph_module
        input_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.aten.convolution.default
        ).args[0]
        input_shape = input_node.meta["val"].shape
        assert not _needs_truncation(
            input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input height should not need truncation after transformation"
        assert not _needs_truncation(
            input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input width should not need truncation after transformation"
        slice_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.aten.slice_copy.Tensor
        ]
        assert (
            len(slice_nodes) == 0
        ), "No slice nodes should be inserted when no adjustment is needed"


class PoolingModule(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def test_size_adjust_input_static_pooling():
    kernel_size, stride, padding = 3, 3, 1
    model = PoolingModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 9, 9),)
    ep = export(model, example_inputs)
    edge_model = to_edge(
        ep,
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default]
        ),
    )
    input_shape = example_inputs[0].shape
    assert _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should need truncation before transformation"
    assert _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should need truncation before transformation"
    edge_model = edge_model.transform([RemoveGetItemPass(), SizeAdjustInputPass()])
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.edge.aten.max_pool2d.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    assert not _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should not need truncation after transformation"
    assert not _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should not need truncation after transformation"


def test_size_adjust_input_static_pooling_no_adjustment_needed():
    kernel_size, stride, padding = 3, 1, 1
    model = PoolingModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 9, 9),)
    ep = export(model, example_inputs)
    edge_model = to_edge(
        ep,
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default]
        ),
    )
    input_shape = example_inputs[0].shape
    assert not _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should not need truncation before transformation"
    assert not _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should not need truncation before transformation"
    edge_model = edge_model.transform([RemoveGetItemPass(), SizeAdjustInputPass()])
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.edge.aten.max_pool2d.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    assert not _needs_truncation(
        input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input height should not need truncation after transformation"
    assert not _needs_truncation(
        input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
    ), "Input width should not need truncation after transformation"
    slice_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.edge.aten.slice_copy.Tensor
    ]
    assert (
        len(slice_nodes) == 0
    ), "No slice nodes should be inserted when no adjustment is needed"


def test_size_adjust_input_dynamic_pooling():
    kernel_size, stride, padding = 3, 3, 1
    model = PoolingModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 18, 18),)
    ep = export(
        model,
        example_inputs,
        dynamic_shapes={
            "x": {
                2: Dim("height", min=3, max=12) * 3,
                3: Dim("width", min=3, max=12) * 3,
            }
        },
    )
    edge_model = to_edge(
        ep,
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default]
        ),
    )
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.aten.max_pool2d_with_indices.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    gm = edge_model.exported_program().graph_module
    shape_env = _get_shape_env(gm)
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        assert _needs_truncation(
            input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input height should need truncation before transformation"
        assert _needs_truncation(
            input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input width should need truncation before transformation"
        edge_model = edge_model.transform([RemoveGetItemPass(), SizeAdjustInputPass()])
        gm = edge_model.exported_program().graph_module
        slice_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.aten.slice_copy.Tensor
        ]
        assert (
            len(slice_nodes) == 2
        ), "Two slice nodes should be inserted when adjustment is needed"


def test_size_adjust_input_dynamic_pooling_no_adjustment_needed():
    kernel_size, stride, padding = 3, 1, 1
    model = PoolingModule(kernel_size=kernel_size, stride=stride, padding=padding)
    example_inputs = (torch.randn(1, 3, 18, 18),)
    ep = export(
        model,
        example_inputs,
        dynamic_shapes={
            "x": {
                2: Dim("height", min=3, max=12) * 3,
                3: Dim("width", min=3, max=12) * 3,
            }
        },
    )
    edge_model = to_edge(
        ep,
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default]
        ),
    )
    gm = edge_model.exported_program().graph_module
    input_node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.edge.aten.max_pool2d_with_indices.default
    ).args[0]
    input_shape = input_node.meta["val"].shape
    gm = edge_model.exported_program().graph_module
    shape_env = _get_shape_env(gm)
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        assert not _needs_truncation(
            input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input height should not need truncation before transformation"
        assert not _needs_truncation(
            input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input width should not need truncation before transformation"
        edge_model = edge_model.transform([RemoveGetItemPass(), SizeAdjustInputPass()])
        gm = edge_model.exported_program().graph_module
        input_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.aten.max_pool2d.default
        ).args[0]
        input_shape = input_node.meta["val"].shape
        assert not _needs_truncation(
            input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input height should not need truncation after transformation"
        assert not _needs_truncation(
            input_shape[3], kernel_size=kernel_size, stride=stride, padding=padding
        ), "Input width should not need truncation after transformation"
        slice_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.aten.slice_copy.Tensor
        ]
        assert (
            len(slice_nodes) == 0
        ), "No slice nodes should be inserted when no adjustment is needed"
