# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.cortex_m.utils import to_physical_order
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def _is_nchw_to_nhwc(node: torch.fx.Node) -> bool:
    if node.target == exir_ops.edge.cortex_m.transpose.default:
        return list(node.args[1]) == [0, 2, 3, 1]
    if node.target != exir_ops.edge.aten.view_copy.default:
        return False
    shape = get_first_fake_tensor(node.args[0]).shape
    return (
        len(shape) == 4
        and (shape[1] == 1 or shape[2] == shape[3] == 1)
        and list(node.args[1]) == [shape[0], shape[2], shape[3], shape[1]]
    )


def _get_fused_padding(conv, pad, spatial_dims, offset_index):
    source, before, after, value = pad.args
    tensor = get_first_fake_tensor(source)
    height_dim, width_dim = spatial_dims
    physical_order = to_physical_order([0, 1, 2, 3], tensor)
    height_index = physical_order.index(height_dim)
    width_index = physical_order.index(width_dim)
    if (
        any(
            before[i] or after[i]
            for i in range(4)
            if i not in (height_index, width_index)
        )
        or any(p < 0 for p in (*before, *after))
        or value != -conv.args[offset_index]
    ):
        return None
    height, width = tensor.shape[height_dim], tensor.shape[width_dim]
    kernel = get_first_fake_tensor(conv.args[1]).shape[1:3]
    # The pinned CMSIS-NN MVE 1xN kernel mishandles asymmetric boundaries.
    if height == 1 and kernel[0] == 1:
        return None
    stride, padding, dilation = conv.args[3:6]
    if len(padding) != 2:
        return None
    fused_padding = [
        before[height_index] + padding[0],
        before[width_index] + padding[1],
        after[height_index] + padding[0],
        after[width_index] + padding[1],
    ]
    # Limit fusion to SAME padding, as used by the CMSIS-NN wrappers.
    total = [
        max(((size + step - 1) // step - 1) * step + dil * (k - 1) + 1 - size, 0)
        for size, step, dil, k in zip((height, width), stride, dilation, kernel)
    ]
    if fused_padding != [
        total[0] // 2,
        total[1] // 2,
        total[0] - total[0] // 2,
        total[1] - total[1] // 2,
    ]:
        return None
    return fused_padding


class FuseConvPaddingPass(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        conv_ops = {
            exir_ops.edge.cortex_m.quantized_conv2d.default: (False, 6),
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: (True, 6),
            exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default: (False, 7),
            exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default: (True, 7),
        }
        modified = False
        for node in graph_module.graph.nodes:
            if node.target not in conv_ops:
                continue
            explicit_nhwc, offset_index = conv_ops[node.target]
            pad = node.args[0]
            layout = None
            if explicit_nhwc and _is_nchw_to_nhwc(pad):
                if len(pad.users) != 1:
                    continue
                layout = pad
                pad = layout.args[0]
            if pad.target != exir_ops.edge.cortex_m.pad.default:
                continue
            source = pad.args[0]
            tensor = get_first_fake_tensor(source)
            if (
                tensor.dim() != 4
                or (layout is not None and not tensor.is_contiguous())
                or (
                    not explicit_nhwc
                    and not tensor.is_contiguous(memory_format=torch.channels_last)
                )
            ):
                continue
            height_dim, width_dim = (
                (1, 2) if explicit_nhwc and layout is None else (2, 3)
            )
            fused_padding = _get_fused_padding(
                node, pad, (height_dim, width_dim), offset_index
            )
            if fused_padding is None:
                continue
            args = list(node.args)
            args[0] = source
            if layout is not None:
                layout.args = (
                    source,
                    (
                        [tensor.shape[i] for i in (0, 2, 3, 1)]
                        if layout.target == exir_ops.edge.aten.view_copy.default
                        else layout.args[1]
                    ),
                )
                args[0] = layout
            args[4] = fused_padding
            node.args = tuple(args)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
