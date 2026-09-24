# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from typing import cast

import executorch.backends.fused_quant.ops  # noqa: F401
from executorch.backends.fused_quant.colorer import is_colored
from executorch.backends.fused_quant.graph_utils import (
    compute_meta_val,
    get_qparams_flat,
    get_qparams_from_node,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from torch import fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult


def _copy_meta_and_compute(node: fx.Node, source: fx.Node) -> None:
    node.meta = source.meta.copy()
    node.meta["val"] = compute_meta_val(node)


def _permute(
    graph: fx.Graph, tensor: fx.Node, dims: list[int], source: fx.Node
) -> fx.Node:
    permute = graph.call_function(
        exir_ops.edge.aten.permute_copy.default,
        args=(tensor, dims),
    )
    _copy_meta_and_compute(permute, source)
    return permute


def _permute_qparams(
    node: fx.Node,
    prefix: str,
    dims: list[int],
) -> list[fx.node.Argument]:
    qparams = get_qparams_from_node(node, prefix)
    if qparams is None or qparams.is_per_tensor():
        return get_qparams_flat(node, prefix)

    graph = node.graph
    scale = _permute(graph, qparams.scale, dims, qparams.scale)
    zero_point = _permute(graph, qparams.zero_point, dims, qparams.zero_point)
    return [
        scale,
        zero_point,
        qparams.dtype,
        qparams.quant_min,
        qparams.quant_max,
    ]


def _convert_convolution_to_channels_first(node: fx.Node) -> None:
    graph = node.graph
    inp = get_arg(node, "inp", fx.Node)
    weight = get_arg(node, "weight", fx.Node)
    bias = get_arg(node, "bias", fx.Node | None)
    ndim = inp.meta["val"].ndim
    to_channels_first = [0, ndim - 1, *range(1, ndim - 1)]
    to_channels_last = [0, *range(2, ndim), 1]

    with graph.inserting_before(node):
        inp_channels_first = _permute(graph, inp, to_channels_first, inp)
        weight_channels_first = _permute(graph, weight, to_channels_first, weight)
        args: tuple[fx.node.Argument, ...] = (
            inp_channels_first,
            weight_channels_first,
            bias,
            *_permute_qparams(node, "inp", to_channels_first),
            *_permute_qparams(node, "weight", to_channels_first),
            *get_qparams_flat(node, "bias"),
            *_permute_qparams(node, "out", to_channels_first),
            get_arg(node, "stride"),
            get_arg(node, "padding"),
            get_arg(node, "dilation"),
            get_arg(node, "transposed"),
            get_arg(node, "output_padding"),
            get_arg(node, "groups"),
        )
        convolution = graph.call_function(
            exir_ops.edge.fused_quant.convolution.default,
            args=args,
        )
        _copy_meta_and_compute(convolution, node)
        output = _permute(graph, convolution, to_channels_last, node)

    node.replace_all_uses_with(output)
    graph.erase_node(node)


def _convert_avg_pool2d_to_channels_first(node: fx.Node) -> None:
    graph = node.graph
    inp = get_arg(node, "inp", fx.Node)
    to_channels_first = [0, 3, 1, 2]
    to_channels_last = [0, 2, 3, 1]

    with graph.inserting_before(node):
        inp_channels_first = _permute(graph, inp, to_channels_first, inp)
        args: tuple[fx.node.Argument, ...] = (
            inp_channels_first,
            *_permute_qparams(node, "inp", to_channels_first),
            *_permute_qparams(node, "out", to_channels_first),
            get_arg(node, "kernel_size"),
            get_arg(node, "stride"),
            get_arg(node, "padding"),
            get_arg(node, "ceil_mode"),
            get_arg(node, "count_include_pad"),
            get_arg(node, "divisor_override"),
        )
        pool = graph.call_function(
            exir_ops.edge.fused_quant.avg_pool2d.default,
            args=args,
        )
        _copy_meta_and_compute(pool, node)
        output = _permute(graph, pool, to_channels_last, node)

    node.replace_all_uses_with(output)
    graph.erase_node(node)


def _convert_max_pool2d_with_indices_to_channels_first(node: fx.Node) -> None:
    users = list(node.users)
    if any(
        user.target is not operator.getitem or user.args[1] not in (0, 1)
        for user in users
    ):
        raise ValueError(
            "fused_quant.max_pool2d_with_indices_channels_last must be consumed "
            "through getitem(0) or getitem(1)"
        )

    graph = node.graph
    inp = get_arg(node, "inp", fx.Node)
    to_channels_first = [0, 3, 1, 2]
    to_channels_last = [0, 2, 3, 1]
    with graph.inserting_before(node):
        inp_channels_first = _permute(graph, inp, to_channels_first, inp)
        args: tuple[fx.node.Argument, ...] = (
            inp_channels_first,
            *_permute_qparams(node, "inp", to_channels_first),
            *_permute_qparams(node, "out", to_channels_first),
            get_arg(node, "kernel_size"),
            get_arg(node, "stride"),
            get_arg(node, "padding"),
            get_arg(node, "dilation"),
            get_arg(node, "ceil_mode"),
        )
        pool = graph.call_function(
            exir_ops.edge.fused_quant.max_pool2d_with_indices.default,
            args=args,
        )
        _copy_meta_and_compute(pool, node)

        for user in users:
            output_index = cast(int, user.args[1])
            output_channels_first = graph.call_function(
                operator.getitem,
                args=(pool, output_index),
            )
            _copy_meta_and_compute(output_channels_first, user)
            output = _permute(
                graph,
                output_channels_first,
                to_channels_last,
                user,
            )
            user.replace_all_uses_with(output)
            graph.erase_node(user)

    graph.erase_node(node)


class ToChannelsFirst(PassBase):
    """Canonicalize channels-last fused ops to channels-first fused ops.

    Colored nodes are left alone: a backend that claimed a channels-last op
    wants it in that layout, and relayouting would take it away. See
    :class:`executorch.backends.fused_quant.colorer.ColorerBase`.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        conversions = (
            (
                exir_ops.edge.fused_quant.convolution_channels_last.default,
                _convert_convolution_to_channels_first,
            ),
            (
                exir_ops.edge.fused_quant.max_pool2d_with_indices_channels_last.default,
                _convert_max_pool2d_with_indices_to_channels_first,
            ),
            (
                exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
                _convert_avg_pool2d_to_channels_first,
            ),
        )

        modified = False
        for target, convert in conversions:
            for node in list(graph.find_nodes(op="call_function", target=target)):
                if is_colored(node):
                    continue
                convert(node)
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
