# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List, Tuple

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.transforms import get_shape


@register_node_visitor
class TopKVisitor(NodeVisitor):
    target = "aten.topk.default"

    @staticmethod
    def _resolve_dim(node: torch.fx.Node) -> int:
        """Resolve and validate the `dim` argument.

        Defaults to the last dimension; normalises negatives; requires
        the last dimension.
        """
        in_shape_len = len(get_shape(node.args[0]))
        dim = cast(int, node.args[2]) if len(node.args) > 2 else in_shape_len - 1
        if dim < 0:
            dim += in_shape_len
        if dim != in_shape_len - 1:
            raise AssertionError("Not supported dim not being last dimension!")
        return dim

    @staticmethod
    def _validate_flags(node: torch.fx.Node) -> None:
        """Validate the optional ``largest`` and ``sorted`` arguments."""
        if len(node.args) > 3 and not cast(bool, node.args[3]):
            raise AssertionError("Not supported largest = False.")
        if len(node.args) > 4 and not cast(bool, node.args[4]):
            raise AssertionError("Not supported sorted = False.")

    def _process_getitem_users(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> Tuple[str, List[int]]:
        """Inspect getitem users of the TopK node to determine outputs.

        Returns ``(output_type, tensor_ids)`` where *output_type* is one
        of ``"value"``, ``"index"``, or ``"both"``.
        """
        outputs: List[int] = []
        output_type = "both"
        num_users = len(node.users)

        for user in node.users.keys():
            if user.target.__name__ != "getitem" or len(user.args) <= 1:
                continue
            idx = user.args[1]
            if idx == 0:
                val_id = self.define_tensor(user, enn_graph, vals_to_ids)
                vals_to_ids[user] = val_id
                outputs.append(val_id)
                if num_users == 1:
                    output_type = "value"
            elif idx == 1:
                idx_id = self.define_tensor(user, enn_graph, vals_to_ids)
                vals_to_ids[user] = idx_id
                outputs.append(idx_id)
                if num_users == 1:
                    output_type = "index"

        return output_type, outputs

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:
        input_id = self.define_tensor(node.args[0], enn_graph, vals_to_ids)

        dim = self._resolve_dim(node)
        self._validate_flags(node)
        output_type, all_output_tensors = self._process_getitem_users(
            node, enn_graph, vals_to_ids
        )

        params = {
            "k_dims": cast(int, node.args[1]),
            "output": output_type,
            "axis": dim,
        }
        self._update_params_qdtype(node, params)
        enn_graph.define_op(node.name, "TopK", [input_id], all_output_tensors, params)

        return True
