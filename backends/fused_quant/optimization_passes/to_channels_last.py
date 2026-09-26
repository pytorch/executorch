# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator
from typing import cast, Optional

import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.colorer import is_colored
from executorch.backends.fused_quant.graph_utils import (
    get_qparams_flat,
    get_qparams_from_node,
)
from executorch.backends.fused_quant.ops_utils import compute_conv_out_shape_nhwc
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch import fx
from torch._export.utils import _detect_fake_mode_from_gm


def _permute_qparams(
    node: fx.Node,
    prefix: str,
    dims: list[int],
) -> list[fx.node.Argument]:
    qparams = get_qparams_from_node(node, prefix)
    if qparams is None or qparams.is_per_tensor():
        return get_qparams_flat(node, prefix)

    graph = node.graph
    permuted: list[fx.Node] = []
    for value in (qparams.scale, qparams.zero_point):
        value_permuted = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            args=(value, dims),
        )
        value_permuted.meta["val"] = value.meta["val"].permute(*dims)
        permuted.append(value_permuted)

    return [
        *permuted,
        qparams.dtype,
        qparams.quant_min,
        qparams.quant_max,
    ]


def _convert_convolution_to_channels_last(node: fx.Node) -> None:
    """Convert a fused_quant.convolution node to fused_quant.convolution_channels_last.

    This function:
    1. Inserts permute nodes to convert inputs from NCHW -> NHWC
    2. Inserts permute nodes to convert weights from OIHW -> OHWI
    3. Replaces the convolution with convolution_channels_last
    4. Inserts permute node to convert output from NHWC -> NCHW
    """
    graph = node.graph

    inp = get_arg(node, "inp", fx.Node)
    weight = get_arg(node, "weight", fx.Node)

    inp_shape = inp.meta["val"].shape
    num_spatial_dims = len(inp_shape) - 2  # Subtract batch and channel dims

    # NCHW -> NHWC: (0, 2, 3, ..., n+1, 1)
    nchw_to_nhwc = [0] + list(range(2, num_spatial_dims + 2)) + [1]
    # NHWC -> NCHW: (0, n+1, 1, 2, ..., n)
    nhwc_to_nchw = [0, num_spatial_dims + 1] + list(range(1, num_spatial_dims + 1))
    # OIHW -> OHWI: (0, 2, 3, ..., n+1, 1)
    oihw_to_ohwi = [0] + list(range(2, num_spatial_dims + 2)) + [1]

    with graph.inserting_before(node):
        # Transpose input from NCHW to NHWC
        inp_nhwc = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            args=(inp, nchw_to_nhwc),
        )
        inp_nhwc.meta["val"] = inp.meta["val"].permute(*nchw_to_nhwc)

        # Transpose weight from OIHW to OHWI
        weight_ohwi = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            args=(weight, oihw_to_ohwi),
        )
        weight_ohwi.meta["val"] = weight.meta["val"].permute(*oihw_to_ohwi)

        bias = get_arg(node, "bias", Optional[fx.Node])
        stride = get_arg(node, "stride", list[int])
        padding = get_arg(node, "padding", list[int])
        dilation = get_arg(node, "dilation", list[int])
        transposed = get_arg(node, "transposed", bool)
        output_padding = get_arg(node, "output_padding", list[int])
        groups = get_arg(node, "groups", int)

        # pyre-ignore[60]: Pyre doesn't support multiple variadic tuple splats
        new_args: tuple[fx.node.Argument, ...] = (
            inp_nhwc,
            weight_ohwi,
            bias,
            *_permute_qparams(node, "inp", nchw_to_nhwc),
            *_permute_qparams(node, "weight", oihw_to_ohwi),
            *get_qparams_flat(node, "bias"),
            *_permute_qparams(node, "out", nchw_to_nhwc),
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        conv_channels_last = graph.call_function(
            exir_ops.edge.fused_quant.convolution_channels_last.default,
            args=new_args,
        )

        out_shape_nhwc = compute_conv_out_shape_nhwc(
            inp_nhwc.meta["val"],
            weight_ohwi.meta["val"],
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        assert graph.owning_module is not None
        fake_mode = _detect_fake_mode_from_gm(graph.owning_module)
        assert fake_mode
        with fake_mode:
            conv_channels_last.meta["val"] = torch.empty(
                out_shape_nhwc, dtype=node.meta["val"].dtype
            )

        # Transpose output from NHWC back to NCHW
        out_nchw = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            args=(conv_channels_last, nhwc_to_nchw),
        )
        out_nchw.meta["val"] = conv_channels_last.meta["val"].permute(*nhwc_to_nchw)

    node.replace_all_uses_with(out_nchw)
    graph.erase_node(node)


def _convert_max_pool2d_with_indices_to_channels_last(node: fx.Node) -> None:
    """Convert both max_pool2d_with_indices tuple outputs to channels-last."""
    graph = node.graph
    inp = get_arg(node, "inp", fx.Node)
    nchw_to_nhwc = [0, 2, 3, 1]
    nhwc_to_nchw = [0, 3, 1, 2]
    users = list(node.users)
    if any(
        user.target != operator.getitem or user.args[1] not in (0, 1) for user in users
    ):
        raise ValueError(
            "fused_quant.max_pool2d_with_indices must be consumed through "
            "getitem(0) or getitem(1)"
        )

    with graph.inserting_before(node):
        inp_nhwc = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            args=(inp, nchw_to_nhwc),
        )
        inp_nhwc.meta["val"] = inp.meta["val"].permute(*nchw_to_nhwc)

        # pyre-ignore[60]: Pyre doesn't support multiple variadic tuple splats
        new_args: tuple[fx.node.Argument, ...] = (
            inp_nhwc,
            *_permute_qparams(node, "inp", nchw_to_nhwc),
            *_permute_qparams(node, "out", nchw_to_nhwc),
            get_arg(node, "kernel_size", list[int]),
            get_arg(node, "stride", list[int]),
            get_arg(node, "padding", list[int]),
            get_arg(node, "dilation", list[int]),
            get_arg(node, "ceil_mode", bool),
        )
        pool_channels_last = graph.call_function(
            exir_ops.edge.fused_quant.max_pool2d_with_indices_channels_last.default,
            args=new_args,
        )
        node_values = cast(tuple[torch.Tensor, torch.Tensor], node.meta["val"])
        pool_channels_last.meta["val"] = tuple(
            value.permute(*nchw_to_nhwc) for value in node_values
        )

        for user in users:
            output_index = cast(int, user.args[1])
            output_nhwc = graph.call_function(
                operator.getitem,
                args=(pool_channels_last, output_index),
            )
            output_nhwc.meta = user.meta.copy()
            output_nhwc.meta["val"] = pool_channels_last.meta["val"][output_index]

            output_nchw = graph.call_function(
                exir_ops.edge.aten.permute_copy.default,
                args=(output_nhwc, nhwc_to_nchw),
            )
            output_nchw.meta = user.meta.copy()
            output_nchw.meta["val"] = node_values[output_index]
            user.replace_all_uses_with(output_nchw)
            graph.erase_node(user)

    graph.erase_node(node)


def _convert_avg_pool2d_to_channels_last(node: fx.Node) -> None:
    graph = node.graph
    inp = get_arg(node, "inp", fx.Node)
    nchw_to_nhwc = [0, 2, 3, 1]
    nhwc_to_nchw = [0, 3, 1, 2]

    with graph.inserting_before(node):
        inp_nhwc = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            args=(inp, nchw_to_nhwc),
        )
        inp_nhwc.meta["val"] = inp.meta["val"].permute(*nchw_to_nhwc)

        # pyre-ignore[60]: Pyre doesn't support multiple variadic tuple splats
        new_args: tuple[fx.node.Argument, ...] = (
            inp_nhwc,
            *_permute_qparams(node, "inp", nchw_to_nhwc),
            *_permute_qparams(node, "out", nchw_to_nhwc),
            get_arg(node, "kernel_size", list[int]),
            get_arg(node, "stride", list[int]),
            get_arg(node, "padding", list[int]),
            get_arg(node, "ceil_mode", bool),
            get_arg(node, "count_include_pad", bool),
            get_arg(node, "divisor_override", Optional[int]),
        )
        pool_channels_last = graph.call_function(
            exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
            args=new_args,
        )
        pool_channels_last.meta["val"] = node.meta["val"].permute(*nchw_to_nhwc)

        output_nchw = graph.call_function(
            exir_ops.edge.aten.permute_copy.default,
            args=(pool_channels_last, nhwc_to_nchw),
        )
        output_nchw.meta = node.meta.copy()

    node.replace_all_uses_with(output_nchw)
    graph.erase_node(node)


class ToChannelsLast(ExportPass):
    """
    Convert fused quant convolution and pool ops to channels-last layout.

    This pass identifies:
        fused_quant.convolution (NCHW layout)
        fused_quant.max_pool2d_with_indices (NCHW layout)
        fused_quant.avg_pool2d (NCHW layout)

    and replaces it with:
        permute (NCHW->NHWC) -> fused_quant.convolution_channels_last -> permute (NHWC->NCHW)
        permute (NCHW->NHWC) -> fused_quant.max_pool2d_with_indices_channels_last
            -> getitem per live output -> permute (NHWC->NCHW)
        permute (NCHW->NHWC) -> fused_quant.avg_pool2d_channels_last
            -> permute (NHWC->NCHW)

    For most backends, NHWC layout is more efficient than NCHW layout, so this optimization
    is applied in the fused_quant IR.

    Colored nodes are left alone: a backend that claimed an op is going to lower
    it itself, and relayouting would take it away. See
    :class:`executorch.backends.fused_quant.colorer.ColorerBase`.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        """
        Execute the channels-last optimization pass on the graph module.

        Args:
            graph_module: The graph module to transform.

        Returns:
            PassResult with the transformed graph module and modification status.
        """
        modified = False
        graph = graph_module.graph

        for node in graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution.default,
        ):
            if is_colored(node):
                continue
            _convert_convolution_to_channels_last(node)
            modified = True

        for node in graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.max_pool2d_with_indices.default,
        ):
            if is_colored(node):
                continue
            _convert_max_pool2d_with_indices_to_channels_last(node)
            modified = True

        for node in graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.avg_pool2d.default,
        ):
            if is_colored(node):
                continue
            _convert_avg_pool2d_to_channels_last(node)
            modified = True

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
