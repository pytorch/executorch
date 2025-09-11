# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import numpy as np

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.transforms import get_shape


@register_node_visitor
class ConstantPadNDVisitor(NodeVisitor):
    target = "aten.constant_pad_nd.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> None:
        input = node.args[0]
        input_id = self.define_tensor(input, enn_graph, vals_to_ids)

        # torch padding order starts from the last axis, change the order to fit samsung lite-core
        paddings = np.reshape(cast(List[int], node.args[1]), (-1, 2))[::-1].astype(
            np.uint32
        )
        in_shape = get_shape(input)
        paddings = paddings.reshape(-1).tolist()
        paddings = [0] * (2 * len(in_shape) - len(paddings)) + paddings
        paddings = paddings[::2] + paddings[1::2]

        padding_value = node.args[2]
        assert padding_value == 0.0, "Only Support pad constant 0 now."
        # output
        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        params = {
            "explicit_padding": paddings,
            "padding": "EXPLICIT",
            "padding_type": "CONSTANT",
        }

        enn_graph.define_op(node.name, "PAD", [input_id], [output_id], params)
