# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNStaticConstantPad,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node


@register_node_visitor
class StaticConstantPadVisitor(NodeVisitor):
    target = "aten.constant_pad_nd.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        self.define_nodes_tensor_inputs_outputs(node, xnn_graph, vals_to_ids)

        # input
        input_id = vals_to_ids[get_input_node(node, 0)]

        # output
        output_id = vals_to_ids[node]

        all_paddings = cast(List[int], node.args[1])

        check_or_raise(
            len(all_paddings) % 2 == 0,
            f"Expected even number of padding values, got {len(all_paddings)}",
        )

        # Explanation of padding as given by PyTorch vs as expected by XNNPACK:
        #
        # Let n be the number of dimensions in the input, and let k be the
        # number of dimensions which we want to pad (k <= n).
        # The list all_paddings, given by PyTorch, has length 2k and contains
        # the padding amounts for before and after each of the LAST k input
        # dimensions, but in descending order by dimension. i.e.
        # [
        #     padding before dim n - 1,
        #     padding after dim n - 1,
        #     padding before dim n - 2,
        #     padding after dim n - 2,
        #     ...
        #     padding before dim n - k,
        #     padding after dim n - k,
        # ]
        #
        # Ex. if n = 4 and k = 2, all_paddings will look like:
        # [
        #     padding before dim 3,
        #     padding after dim 3,
        #     padding before dim 2,
        #     padding after dim 2,
        # ]
        #
        # The way that XNNPACK expects padding amounts to be passed in is in
        # two lists, pre_paddings and post_paddings. pre_paddings should contain
        # n elements, which are the padding amounts for before each of the n
        # dimensions of the input in ascending order by dimensions.
        # post_paddings is the same but for the padding amounts after each
        # dimension. i.e. we want pre and post paddings to look like:
        # pre_paddings = [
        #     padding before dim 0,
        #     padding before dim 1,
        #     ...
        #     padding before dim n - 1,
        # ]
        # post_paddings = [
        #     padding after dim 0,
        #     padding after dim 1,
        #     ...
        #     padding after dim n - 1,
        # ]
        #
        # To get pre and post paddings in this form, we need to
        # a) Append 2(n - k) zeros to the end of all_paddings as the padding
        #    amounts for before and after each of the leading n - k input
        #    input dimensions
        # b) Extract the even index elements of all_paddings in reverse order
        #    as pre_paddings, and same for the odd index elements as
        #    post_paddings

        # a)
        num_padding_dims = 2 * len(
            get_tensor_value(xnn_graph.xvalues[input_id]).dims
        )  # 2n
        num_zero_padding_dims = num_padding_dims - len(all_paddings)  # 2(n - k)
        all_paddings = all_paddings + (
            [0] * num_zero_padding_dims
        )  # zeros have been appended

        # b)
        # tuple[0] = prepadding dim[-1]
        # tuple[1] = postpadding dim[-1]
        pre_paddings = all_paddings[-2::-2]  # even index elements in reverse order
        post_paddings = all_paddings[::-2]  # odd index elements in reverse order

        ser_node = XNode(
            xnode_union=XNNStaticConstantPad(
                pre_paddings=pre_paddings,
                post_paddings=post_paddings,
                padding_value=cast(float, node.args[2]),
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
