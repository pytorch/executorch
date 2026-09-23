# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


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
            pad = node.args[0]
            if pad.target != exir_ops.edge.cortex_m.pad.default:
                continue
            explicit_nhwc, offset_index = conv_ops[node.target]
            source, before, after, value = pad.args
            tensor = get_first_fake_tensor(source)
            if tensor.dim() != 4 or (
                not explicit_nhwc
                and not tensor.is_contiguous(memory_format=torch.channels_last)
            ):
                continue
            if (
                before[0]
                or before[3]
                or after[0]
                or after[3]
                or any(p < 0 for p in (*before, *after))
                or value != -node.args[offset_index]
            ):
                continue
            height, width = tensor.shape[1:3] if explicit_nhwc else tensor.shape[2:4]
            kernel = get_first_fake_tensor(node.args[1]).shape[1:3]
            # The pinned CMSIS-NN MVE 1xN kernel mishandles asymmetric boundaries.
            if height == 1 and kernel[0] == 1:
                continue
            stride, padding, dilation = node.args[3:6]
            if len(padding) != 2:
                continue
            fused_padding = [
                before[1] + padding[0],
                before[2] + padding[1],
                after[1] + padding[0],
                after[2] + padding[1],
            ]
            # Limit fusion to SAME padding, as used by the CMSIS-NN wrappers.
            total = [
                max(
                    ((size + step - 1) // step - 1) * step + dil * (k - 1) + 1 - size, 0
                )
                for size, step, dil, k in zip((height, width), stride, dilation, kernel)
            ]
            if fused_padding != [
                total[0] // 2,
                total[1] // 2,
                total[0] - total[0] // 2,
                total[1] - total[1] // 2,
            ]:
                continue
            args = list(node.args)
            args[0] = source
            args[4] = fused_padding
            node.args = tuple(args)
            modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
