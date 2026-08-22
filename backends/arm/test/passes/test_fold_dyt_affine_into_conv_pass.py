# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict
"""Tests for folding exact quantized DyT affine maps into following convs."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import cast, ClassVar, Dict, Tuple

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch

from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    InsertRescaleInt32Pass,
    MatchArgRanksPass,
)
from executorch.backends.arm._passes.arm_pass_utils import get_param_tensor
from executorch.backends.arm._passes.fold_dyt_affine_into_conv_pass import (
    FoldDyTAffineIntoConvPass,
)
from executorch.backends.arm._passes.fold_dyt_alpha_into_lut_pass import (
    FoldDyTAlphaIntoLUTPass,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.export import export
from torch.fx import Node


_CHANNELS: int = 2


class _PostRescaleAffineFixture(torch.nn.Module):
    # Declared so the checker sees the registered buffers as Tensors rather than
    # the ``Tensor | Module`` that ``nn.Module.__getattr__`` is annotated to give.
    table: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        *,
        table: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("table", table)
        self.register_buffer("gamma", gamma)
        self.register_buffer("beta", beta)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x_code: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return x_code, self.table, self.gamma, self.beta, self.weight, self.bias


def _qargs(scale: float, zp: int, dtype: torch.dtype = torch.int8) -> QuantArgs:
    dtype_range = torch.iinfo(dtype)
    return QuantArgs(
        scale=scale,
        zp=zp,
        qmin=dtype_range.min,
        qmax=dtype_range.max,
        dtype=dtype,
    )


# ``QuantArgs.scale``/``zp`` are typed to also cover the per-channel case, where
# they are lists. Every fixture in this file is per-tensor, so narrow them once
# here instead of casting at each arithmetic site.
def _scale_of(qargs: QuantArgs) -> float:
    return cast(float, qargs.scale)


def _zp_of(qargs: QuantArgs) -> int:
    return cast(int, qargs.zp)


def _channel_codes(value: int | tuple[int, ...]) -> torch.Tensor:
    values = (value,) * _CHANNELS if isinstance(value, int) else value
    return torch.tensor(values, dtype=torch.int8)


def _buffer_nodes(exported_program: ExportedProgram) -> dict[str, Node]:
    graph = exported_program.graph_module.graph
    nodes_by_name = {node.name: node for node in graph.nodes}
    return {
        buffer_name: nodes_by_name[placeholder_name]
        for placeholder_name, buffer_name in exported_program.graph_signature.inputs_to_buffers.items()
    }


def _pass_module() -> ModuleType:
    return importlib.import_module(
        "executorch.backends.arm._passes.fold_dyt_affine_into_conv_pass"
    )


def _fixture_weight_and_bias(
    *, depthwise: bool, channel_slice: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the conv constants matching the graph shape under test."""
    if depthwise:
        weight = torch.tensor(
            [
                [[[1, -2, 1]]],
                [[[2, 1, -1]]],
            ],
            dtype=torch.int8,
        )
    elif channel_slice:
        # A single input channel keeps the graph well formed behind the
        # narrowing slice: the conv really does consume 1 of the 2 affine
        # channels, which is what makes this a genuine mismatch rather than an
        # impossible graph.
        weight = torch.tensor([[[[1]]], [[[3]]]], dtype=torch.int8)
    else:
        weight = torch.tensor(
            [
                [[[1]], [[2]]],
                [[[3]], [[-2]]],
            ],
            dtype=torch.int8,
        )
    return weight, torch.tensor([5, -7], dtype=torch.int32)


def _fixture_rescale_params(
    actual_dyt_identity_qparams: bool,
) -> tuple[QuantArgs, int, float, int, int, float, float, float]:
    """Return the TABLE qparams and the rescale scales/zero points to build.

    The identity variant replays the scales a real DyT block produces, so the
    fixture exercises the same integer arithmetic the pass sees in a model.

    """
    if actual_dyt_identity_qparams:
        table_qargs = _qargs(scale=0.00588326808065176, zp=-6)
        beta_scale = 1.52587890625e-05
        common_scale = (2.0 * _scale_of(table_qargs)) / (1 << 20)
        return (
            table_qargs,
            -128,
            1.0 / 255.0,
            _zp_of(table_qargs),
            -128,
            _scale_of(table_qargs) / common_scale,
            beta_scale / common_scale,
            common_scale / _scale_of(table_qargs),
        )
    table_qargs = _qargs(scale=0.1, zp=0)
    return (table_qargs, 0, 1.0, 0, 0, 1.0, 1.0, 1.0)


def _fixture_conv_input(
    graph: torch.fx.Graph,
    layout_source: Node,
    *,
    layout_permute_count: int,
    slice_passthrough: bool,
    channel_slice: bool,
) -> tuple[Node, Node]:
    """Build the NHWC->NCHW layout chain feeding the conv.

    Returns the layout output (shared users hang off it) and the node the conv
    actually consumes, which may sit behind a slice.

    """
    layout_output = layout_source
    for _ in range(layout_permute_count):
        layout_output = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            (layout_output, [0, 3, 1, 2]),
        )
    conv_input = layout_output
    if slice_passthrough:
        conv_input = graph.call_function(
            exir_ops.edge.aten.slice_copy.Tensor,
            (layout_output, 0, 0, 1, 1),
        )
    if channel_slice:
        conv_input = graph.call_function(
            exir_ops.edge.aten.slice_copy.Tensor,
            (layout_output, 1, 0, 1, 1),
        )
    return layout_output, conv_input


def _build_post_rescale_fixture(
    *,
    table: torch.Tensor,
    gamma_code: int | tuple[int, ...],
    beta_code: int | tuple[int, ...],
    actual_dyt_identity_qparams: bool,
    padded_depthwise: bool,
    unpadded_depthwise: bool = False,
    slice_passthrough: bool = False,
    channel_slice: bool = False,
    shared_layout_user: bool = False,
    input_width: int = 4,
    input_height: int = 1,
    input_channels: int = _CHANNELS,
    affine_view_shape: tuple[int, ...] | None = None,
    activation_view_shape: tuple[int, ...] | None = None,
    layout_permute_count: int = 1,
    conv_input_zp: int | None = None,
) -> tuple[ExportedProgram, torch.Tensor, torch.Tensor]:
    weight, bias = _fixture_weight_and_bias(
        depthwise=padded_depthwise or unpadded_depthwise,
        channel_slice=channel_slice,
    )

    test_input = (
        torch.arange(
            input_height * input_width * input_channels, dtype=torch.int8
        ).reshape(1, input_height, input_width, input_channels),
    )
    exported_program = export(
        _PostRescaleAffineFixture(
            table=table,
            gamma=_channel_codes(gamma_code),
            beta=_channel_codes(beta_code),
            weight=weight,
            bias=bias,
        ),
        test_input,
        strict=True,
    )
    graph = exported_program.graph_module.graph
    buffers = _buffer_nodes(exported_program)
    activation = next(
        node
        for node in graph.nodes
        if node.op == "placeholder"
        and node.name not in exported_program.graph_signature.inputs_to_buffers
    )
    output = next(node for node in graph.nodes if node.op == "output")
    view_shape = list(affine_view_shape or (1, 1, 1, _CHANNELS))

    (
        table_qargs,
        gamma_input_zp,
        gamma_output_scale,
        gamma_output_zp,
        beta_input_zp,
        add_activation_scale,
        add_beta_scale,
        add_output_scale,
    ) = _fixture_rescale_params(actual_dyt_identity_qparams)

    with graph.inserting_before(output):
        table_node = graph.call_function(
            exir_ops.backend.tosa.TABLE.default,
            (activation, buffers["table"]),
        )
        activation_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (table_node, torch.int32, [1.0], table_qargs.zp, 0),
        )
        gamma_activation = activation_rescale
        if activation_view_shape is not None:
            gamma_activation = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                (activation_rescale, list(activation_view_shape)),
            )
        gamma_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (buffers["gamma"], torch.int32, [1.0], gamma_input_zp, 0),
        )
        gamma_view = graph.call_function(
            exir_ops.edge.aten.view_copy.default,
            (gamma_rescale, view_shape),
        )
        mul = graph.call_function(
            exir_ops.edge.aten.mul.Tensor,
            (gamma_activation, gamma_view),
        )
        mul_output_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (mul, torch.int8, [gamma_output_scale], 0, gamma_output_zp),
        )
        add_activation_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (
                mul_output_rescale,
                torch.int32,
                [add_activation_scale],
                gamma_output_zp,
                0,
            ),
        )
        beta_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (buffers["beta"], torch.int32, [add_beta_scale], beta_input_zp, 0),
        )
        beta_view = graph.call_function(
            exir_ops.edge.aten.view_copy.default,
            (beta_rescale, view_shape),
        )
        add = graph.call_function(
            exir_ops.edge.aten.add.Tensor,
            (add_activation_rescale, beta_view),
        )
        add_output_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (add, torch.int8, [add_output_scale], 0, table_qargs.zp),
        )
        layout_output, conv_input = _fixture_conv_input(
            graph,
            add_output_rescale,
            layout_permute_count=layout_permute_count,
            slice_passthrough=slice_passthrough,
            channel_slice=channel_slice,
        )
        conv = graph.call_function(
            exir_ops.edge.aten.convolution.default,
            (
                conv_input,
                buffers["weight"],
                buffers["bias"],
                [1, 1],
                [0, 1] if padded_depthwise else [0, 0],
                [1, 1],
                False,
                [0, 0],
                _CHANNELS if padded_depthwise or unpadded_depthwise else 1,
            ),
        )
        shared_output = None
        if shared_layout_user:
            shared_output = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                (layout_output, [1, _CHANNELS, 1, input_width]),
            )

    table_node.meta["output_qparams"] = {0: table_qargs}
    table_node.meta["val"] = torch.empty(
        (1, input_height, input_width, input_channels),
        dtype=torch.int8,
        device="meta",
    )
    add_output_rescale.meta["val"] = torch.empty(
        (1, input_height, input_width, _CHANNELS),
        dtype=torch.int8,
        device="meta",
    )
    conv.meta["input_qparams"] = {
        0: _qargs(
            scale=_scale_of(table_qargs),
            zp=_zp_of(table_qargs) if conv_input_zp is None else conv_input_zp,
        ),
        1: _qargs(scale=0.02, zp=0),
    }
    output.args = ((conv, shared_output) if shared_output is not None else (conv,),)
    graph.eliminate_dead_code()
    graph.lint()
    exported_program.graph_module.recompile()
    return exported_program, weight, bias


def _add_second_shared_weight_branch(
    exported_program: ExportedProgram,
) -> None:
    graph = exported_program.graph_module.graph
    buffers = _buffer_nodes(exported_program)
    output = next(node for node in graph.nodes if node.op == "output")
    table = next(
        node
        for node in graph.nodes
        if node.target == exir_ops.backend.tosa.TABLE.default
    )
    conv = next(
        node
        for node in graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    )
    table_qargs = cast(dict[int, QuantArgs], table.meta["output_qparams"])[0]

    with graph.inserting_before(output):
        activation_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (table, torch.int32, [1.0], table_qargs.zp, 0),
        )
        gamma_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (buffers["gamma"], torch.int32, [1.0], -1, 0),
        )
        gamma_view = graph.call_function(
            exir_ops.edge.aten.view_copy.default,
            (gamma_rescale, [1, 1, 1, _CHANNELS]),
        )
        mul = graph.call_function(
            exir_ops.edge.aten.mul.Tensor,
            (activation_rescale, gamma_view),
        )
        gamma_output = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (mul, torch.int8, [1.0], 0, table_qargs.zp),
        )
        add_activation = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (gamma_output, torch.int32, [1.0], table_qargs.zp, 0),
        )
        beta_rescale = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (buffers["beta"], torch.int32, [1.0], 1, 0),
        )
        beta_view = graph.call_function(
            exir_ops.edge.aten.view_copy.default,
            (beta_rescale, [1, 1, 1, _CHANNELS]),
        )
        add = graph.call_function(
            exir_ops.edge.aten.add.Tensor,
            (add_activation, beta_view),
        )
        add_output = graph.call_function(
            exir_ops.backend.tosa.RESCALE.default,
            (add, torch.int8, [1.0], 0, table_qargs.zp),
        )
        nchw = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            (add_output, [0, 3, 1, 2]),
        )
        second_conv = graph.call_function(
            exir_ops.edge.aten.convolution.default,
            (nchw, *conv.args[1:]),
        )

    add_output.meta["val"] = table.meta["val"]
    second_conv.meta["input_qparams"] = dict(conv.meta["input_qparams"])
    output.args = ((conv, second_conv),)
    graph.eliminate_dead_code()
    graph.lint()
    exported_program.graph_module.recompile()


def _call_pass(exported_program: ExportedProgram) -> PassResult:
    pass_class = _pass_module().FoldDyTAffineIntoConvPass
    return pass_class(exported_program).call(exported_program.graph_module)


def _call_targets(exported_program: ExportedProgram) -> list[str]:
    return [
        str(node.target)
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _conv_constants(
    exported_program: ExportedProgram,
) -> tuple[torch.Tensor, torch.Tensor]:
    conv = next(
        node
        for node in exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    )
    weight_node = cast(Node, conv.args[1])
    bias_node = cast(Node, conv.args[2])
    weight = get_param_tensor(exported_program, weight_node)
    bias = get_param_tensor(exported_program, bias_node)
    assert weight is not None
    assert bias is not None
    return weight, bias


def test_unpadded_conv_folds_exact_integer_affine_into_weight_and_bias() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
    )
    original_constants = _buffer_nodes(exported_program)

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    expected_weight = weight.to(torch.int16).mul(2).to(torch.int8)
    expected_bias = bias + weight.to(torch.int32).sum(dim=(1, 2, 3)).mul(3)
    placeholder_names = {
        node.name
        for node in exported_program.graph_module.graph.nodes
        if node.op == "placeholder"
    }

    assert result.modified
    assert torch.equal(folded_weight, expected_weight)
    assert torch.equal(folded_bias, expected_bias)
    assert not any("aten.mul" in target for target in _call_targets(exported_program))
    assert not any("aten.add" in target for target in _call_targets(exported_program))
    assert original_constants["weight"].name not in placeholder_names
    assert original_constants["bias"].name not in placeholder_names
    assert original_constants["gamma"].name not in placeholder_names
    assert original_constants["beta"].name not in placeholder_names


def test_unpadded_depthwise_folds_nonuniform_gamma_and_beta() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=(2, 3),
        beta_code=(3, -2),
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        unpadded_depthwise=True,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    expected_weight = (
        weight.to(torch.int16)
        .mul(torch.tensor([2, 3], dtype=torch.int16).reshape(2, 1, 1, 1))
        .to(torch.int8)
    )
    expected_bias = bias + weight.to(torch.int32).sum(dim=(1, 2, 3)).mul(
        torch.tensor([3, -2], dtype=torch.int32)
    )

    assert result.modified
    assert torch.equal(folded_weight, expected_weight)
    assert torch.equal(folded_bias, expected_bias)
    assert not any("aten.mul" in target for target in _call_targets(exported_program))
    assert not any("aten.add" in target for target in _call_targets(exported_program))


def test_mismatched_conv_input_zero_point_is_rejected() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        conv_input_zp=1,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.add" in target for target in targets)


def test_shared_conv_constants_get_distinct_folded_values() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
    )
    _add_second_shared_weight_branch(exported_program)

    result = _call_pass(exported_program)
    convs = [
        node
        for node in exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]
    folded_constants = []
    for conv in convs:
        folded_weight = get_param_tensor(exported_program, cast(Node, conv.args[1]))
        folded_bias = get_param_tensor(exported_program, cast(Node, conv.args[2]))
        assert folded_weight is not None
        assert folded_bias is not None
        folded_constants.append((folded_weight, folded_bias))

    weight_sum = weight.to(torch.int32).sum(dim=(1, 2, 3))
    assert result.modified
    assert len(folded_constants) == 2
    assert torch.equal(
        folded_constants[0][0], weight.to(torch.int16).mul(2).to(torch.int8)
    )
    assert torch.equal(folded_constants[0][1], bias + weight_sum.mul(3))
    assert torch.equal(
        folded_constants[1][0], weight.to(torch.int16).mul(3).to(torch.int8)
    )
    assert torch.equal(folded_constants[1][1], bias + weight_sum.mul(2))


def test_unpadded_identity_affine_removes_ops_without_changing_constants() -> None:
    table = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=127,
        beta_code=-128,
        actual_dyt_identity_qparams=True,
        padded_depthwise=False,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)

    assert result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert not any("aten.mul" in target for target in _call_targets(exported_program))
    assert not any("aten.add" in target for target in _call_targets(exported_program))


def test_padded_depthwise_removes_identity_gamma_but_keeps_beta_add() -> None:
    table = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=127,
        beta_code=127,
        actual_dyt_identity_qparams=True,
        padded_depthwise=True,
        shared_layout_user=True,
    )
    original_constants = _buffer_nodes(exported_program)

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert not any("aten.mul" in target for target in targets)
    assert sum("aten.add" in target for target in targets) == 1
    placeholder_names = {
        node.name
        for node in exported_program.graph_module.graph.nodes
        if node.op == "placeholder"
    }
    assert original_constants["gamma"].name not in placeholder_names
    assert original_constants["beta"].name in placeholder_names


def test_zero_layout_permute_is_rejected() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        input_height=_CHANNELS,
        layout_permute_count=0,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)


def test_repeated_layout_permute_is_rejected() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        input_width=_CHANNELS,
        layout_permute_count=2,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)


def test_wrong_axis_affine_views_are_rejected() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        input_width=_CHANNELS,
        affine_view_shape=(1, 1, _CHANNELS, 1),
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.add" in target for target in targets)


def test_activation_side_view_is_rejected() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        activation_view_shape=(1, 1, 4, _CHANNELS),
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.add" in target for target in targets)


def test_singleton_table_channel_broadcast_is_rejected() -> None:
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        input_channels=1,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.add" in target for target in targets)


def test_unpadded_shared_layout_removes_only_identity_gamma() -> None:
    table = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=127,
        beta_code=127,
        actual_dyt_identity_qparams=True,
        padded_depthwise=False,
        shared_layout_user=True,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert not any("aten.mul" in target for target in targets)
    assert sum("aten.add" in target for target in targets) == 1


def test_unpadded_conv_folds_through_slice_passthrough() -> None:
    table = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=127,
        beta_code=-128,
        actual_dyt_identity_qparams=True,
        padded_depthwise=False,
        slice_passthrough=True,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert not any("aten.mul" in target for target in targets)
    assert not any("aten.add" in target for target in targets)
    assert any("aten.slice_copy" in target for target in targets)


def test_channel_narrowing_slice_is_rejected() -> None:
    """A slice that changes which channels the conv consumes must not fold.

    ``_trace_layout_source`` deliberately does not inspect ``slice_copy``
    arguments; safety comes from the channel-count guard in
    ``_fold_conv_constants``, which compares the affine site's per-channel
    slope/offset count against the conv weight's input channels. Here the
    affine site produces two channels but the slice leaves the conv consuming
    one, so the counts disagree and the fold must decline. This is the
    fail-closed path that keeps the unvalidated ``slice_copy`` passthrough
    sound, so it is pinned here rather than left implicit.

    """
    table = (torch.arange(256, dtype=torch.int16).remainder(21) - 10).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
        channel_slice=True,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.add" in target for target in targets)
    assert any("aten.slice_copy" in target for target in targets)


def test_identity_affine_behind_channel_slice_leaves_conv_constants() -> None:
    """An exact-identity gamma/beta may be dropped even behind a channel slice.

    Identity is established per channel over the whole affine site, so removing
    the Mul/Add is a no-op on every channel and stays sound no matter which
    channels the conv goes on to consume. The conv constants must be left
    untouched: nothing is folded into them, the redundant ops are just deleted.
    Contrast ``test_channel_narrowing_slice_is_rejected``, where a real
    (non-identity) affine behind the same slice is refused outright.

    """
    table = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=127,
        beta_code=-128,
        actual_dyt_identity_qparams=True,
        padded_depthwise=False,
        channel_slice=True,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)

    assert result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    targets = _call_targets(exported_program)
    assert not any("aten.mul" in target for target in targets)
    assert not any("aten.add" in target for target in targets)
    assert any("aten.slice_copy" in target for target in targets)


def test_non_affine_integer_mapping_is_rejected() -> None:
    table = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    exported_program, weight, bias = _build_post_rescale_fixture(
        table=table,
        gamma_code=2,
        beta_code=3,
        actual_dyt_identity_qparams=False,
        padded_depthwise=False,
    )

    result = _call_pass(exported_program)
    folded_weight, folded_bias = _conv_constants(exported_program)
    targets = _call_targets(exported_program)

    assert not result.modified
    assert torch.equal(folded_weight, weight)
    assert torch.equal(folded_bias, bias)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.add" in target for target in targets)


class DyTAffineModule(torch.nn.Module):
    """A full DyT site between two convs, mirroring the real module.

    conv -> NHWC permute -> tanh(alpha * x) -> x * gamma + beta -> back to NCHW
    -> conv. The trailing conv is what the affine folds into.

    """

    test_data: ClassVar[Dict[str, Tuple[torch.Tensor]]] = {
        "rand": (torch.rand(1, 3, 8, 8),),
    }

    def __init__(self, channels: int = 3, alpha: float = 0.5) -> None:
        super().__init__()
        self.conv_in = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))
        self.conv_out = torch.nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.permute(self.conv_in(x), (0, 2, 3, 1))
        y = torch.tanh(self.alpha * y)
        y = y * self.gamma + self.beta
        y = torch.permute(y, (0, 3, 1, 2))
        return self.conv_out(y)


@common.parametrize("test_data", DyTAffineModule.test_data)
def test_fold_dyt_affine_into_conv_tosa_INT(test_data: Tuple[torch.Tensor]) -> None:
    """Pipeline-level counterpart to the IR-level regressions above.

    ``MatchArgRanksPass`` is required, not incidental: this pass only matches
    gamma/beta operands that carry an explicit ``(1, 1, 1, C)`` view. A bare
    ``(C,)`` constant broadcasts against the NHWC activation without one, and the
    pass then declines to fold. ``MatchArgRanksPass`` is what materialises that
    view, and it sits between ``InsertRescaleInt32Pass`` and
    ``InsertTableOpsPass`` in ``ArmPassManager`` for exactly this reason.

    """
    pipeline = PassPipeline[Tuple[torch.Tensor]](
        DyTAffineModule(),
        test_data,
        quantize=True,
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_TABLE_default": 1,
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 2,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
            "executorch_exir_dialects_edge__ops_aten_tanh_default",
        ],
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
        passes_with_exported_program=[
            MatchArgRanksPass,
            FoldDyTAlphaIntoLUTPass,
            FoldDyTAffineIntoConvPass,
        ],
    )
    # The partial ``pass_list`` above stops short of a full TOSA lowering, so no
    # runnable program is left for the comparison stage to execute. Dropped for
    # the same reason as in ``test_insert_rescale_i32_pass.py``, which drives
    # the same two passes. Skipping it does not leave the rewritten weights and
    # biases unchecked: the IR-level regressions above assert the folded
    # constants exactly, and the fold is only ever applied when the per-channel
    # mapping is provably integer-affine, so it is exact by construction rather
    # than approximate.
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
