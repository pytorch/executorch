# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from executorch.backends.samsung.builders.utils import (
    get_map_dtype,
    get_tensor,
    get_tensor_type,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.transforms.utils import is_param_node
from torch.export import ExportedProgram


class NodeVisitor:
    """
    Node visitor pattern for visiting nodes in an edge IR graph
    """

    def __init__(self, exported_program: ExportedProgram) -> None:
        self._exported_program = exported_program or None

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    def define_node(self, node: torch.fx.Node, enn_graph: EnnGraph):
        raise NotImplementedError("NodeVisitor must be extended!")

    def define_tensor(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        swap_nc_for_weights: bool = False,
        output_idx: Optional[int] = None,
    ) -> int:
        if node in vals_to_ids and (output_idx is None or output_idx == 0):
            return vals_to_ids[node]

        # Get tensor basic information
        tensor = get_tensor(self.exported_program, node)

        if output_idx is not None:
            tensor = tensor[output_idx]

        tensor_type = get_tensor_type(self.exported_program, node)
        data_type = get_map_dtype(tensor.dtype)

        const_data = None
        if is_param_node(self.exported_program, node):
            if swap_nc_for_weights:
                tensor = torch.swapdims(tensor, 0, 1)
            const_data = tensor.contiguous().detach().numpy()

        dims = [1] if len(tensor.size()) == 0 else list(tensor.size())

        enn_tensor_id = enn_graph.define_tensor(
            node.name,
            dims,
            data_type,
            tensor_type.name,
            const_data,
        )
        assert enn_tensor_id is not None
        vals_to_ids[node] = enn_tensor_id

        return enn_tensor_id


_node_visitor_dict = {}


def register_node_visitor(visitor):
    assert (
        isinstance(visitor, type)
        and issubclass(visitor, NodeVisitor)
        and hasattr(visitor, "target")
    ), f"Illformed NodeVisitor subclass, can't register!, got: {visitor}"
    if isinstance(visitor.target, str):
        _node_visitor_dict[visitor.target] = visitor
    elif isinstance(visitor.target, (list, tuple)):
        for target in visitor.target:
            _node_visitor_dict[target] = visitor
    else:
        raise TypeError(
            f"target of vistor should be str|Tuple[str]|List[str], not{type(visitor.target)}"
        )


def get_node_visitors(*args) -> Dict[str, NodeVisitor]:
    node_visitors = {}
    """
    Create a new class instance at runtime, and put them in a dict
    """
    for target, visitor in _node_visitor_dict.items():
        assert callable(visitor), "Expecting a callable class, "
        f"but got {visitor} of type {type(visitor)}"
        node_visitors[target] = visitor(*args)
    return node_visitors
