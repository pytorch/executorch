# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from executorch.backends.qualcomm._passes.utils import copy_meta
from executorch.backends.qualcomm.builders.utils import get_attr_from_target
from executorch.exir.pass_base import ExportPass, PassResult
from torch._guards import detect_fake_mode
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


class FoldZeroPadIntoLinear(ExportPass):
    """
    Fold a zero constant pad of the last dim into the linears that read it:
    linear(pad(x, (left, right)), W, b) becomes linear(x, W[:, left:left + n], b),
    n being x's last dim. The padded zeros add nothing to the products.

    On quantized HTP, a Pad of 0 that feeds a FullyConnected pads with the
    quantized 0 rather than with 0.0 when the padded tensor has a nonzero zero
    point, which dequantizes to -zero_point * scale. For example, a 16-bit
    Linear(32, 64)(F.pad(x[1, 8], (0, 24))) of a randn input comes out of the
    QNN x86 HTP simulator with a cosine similarity of 0.27 to its QDQ graph.
    Folding the pad before annotation leaves no such Pad to lower.
    """

    _PAD_TARGETS = {
        torch.ops.aten.pad.default,
        torch.ops.aten.constant_pad_nd.default,
    }

    def _last_dim_zero_pad(
        self,
        node: torch.fx.Node,
    ) -> Optional[Tuple[int, int]]:
        """
        Return (left, right) if node adds left zeros before and right zeros
        after the last dimension, and pads no other dimension.
        """

        def is_pad(node: torch.fx.Node) -> bool:
            return node.op == "call_function" and node.target in self._PAD_TARGETS

        def is_zero_constant_pad(node: torch.fx.Node) -> bool:
            if node.target == torch.ops.aten.pad.default:
                mode = (
                    node.args[2]
                    if len(node.args) > 2
                    else node.kwargs.get("mode", "constant")
                )
                value = node.args[3] if len(node.args) > 3 else node.kwargs.get("value")
                return mode == "constant" and value in (None, 0)

            value = node.args[2] if len(node.args) > 2 else node.kwargs.get("value", 0)
            return value == 0

        def pads_only_last_dim(padding: List[int]) -> bool:
            """Whether only the last dimension is padded.

            For an aten.pad argument like

                (left, right, top, bottom)

            padding[0:2] applies to the last dimension and padding[2:] applies to
            earlier dimensions.
            """
            return all(pad_value == 0 for pad_value in padding[2:])

        if not is_pad(node) or not is_zero_constant_pad(node):
            return None

        padding = list(node.args[1])
        left, right = padding[0], padding[1]

        # Weight slicing can only eliminate padding on the input dimension of a
        # linear, which corresponds to the last tensor dimension.
        if not pads_only_last_dim(padding):
            return None

        if (
            # Negative padding crops the input dimension, which cannot be folded
            # by slicing the linear weight.
            left < 0
            or right < 0
            # Ignore a no-op pad.
            or (left == 0 and right == 0)
        ):
            return None

        return left, right

    def _can_fold_pad_into_linear(
        self,
        linear: torch.fx.Node,
        padded_last_dim_size: int,
    ) -> bool:
        if linear.target != torch.ops.aten.linear.default:
            return False

        weight = linear.args[1]

        if not isinstance(weight, torch.fx.Node):
            return False

        if weight.op != "get_attr":
            return False

        if weight.meta["val"].dim() != 2:
            return False

        return weight.meta["val"].shape[1] == padded_last_dim_size

    def _register_weight(
        self,
        graph_module: torch.fx.GraphModule,
        weight: torch.fx.Node,
        value: torch.Tensor,
    ) -> torch.fx.Node:
        name = get_new_attr_name_with_prefix(f"{weight.name}_unpadded")(graph_module)

        graph_module.register_buffer(name, value)

        with graph_module.graph.inserting_after(weight):
            node = graph_module.graph.get_attr(name)

        fake_mode = detect_fake_mode([weight.meta["val"]])

        node.meta = copy_meta(
            weight.meta,
            lambda m: {
                **m,
                "val": (
                    fake_mode.fake_tensor_converter.from_real_tensor(
                        fake_mode,
                        value,
                    )
                    if fake_mode
                    else value
                ),
            },
        )

        return node

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if (last_dim_padding := self._last_dim_zero_pad(node)) is None:
                continue

            left, right = last_dim_padding

            input_node = node.args[0]
            pad_input_last_dim_size = input_node.meta["val"].shape[-1]
            if not node.users or not all(
                self._can_fold_pad_into_linear(
                    user,
                    left + pad_input_last_dim_size + right,
                )
                for user in node.users
            ):
                continue

            for linear in list(node.users):
                weight = linear.args[1]

                value = get_attr_from_target(graph_module, weight.target)

                unpadded = value.detach()[
                    :, left : left + pad_input_last_dim_size
                ].contiguous()

                linear.replace_input_with(node, input_node)

                linear.replace_input_with(
                    weight,
                    self._register_weight(
                        graph_module,
                        weight,
                        unpadded,
                    ),
                )

            graph.erase_node(node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
