# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from typing import Any, Optional

import torch
import torch.fx as fx
from executorch.exir.tensor import TensorSpec


class ExportGraph:
    """
    ExportGraph serves as a layer between EXIR and FX Graph API.
    It enforces EXIR-specific invariants (ex. having nodes contain specs)
    """

    owning_module: fx.GraphModule
    _graph: fx.Graph

    def __init__(self, owning_module: fx.GraphModule, graph: fx.Graph) -> None:
        self.owning_module = owning_module
        self._graph = graph

    @property
    def nodes(self) -> fx.graph._node_list:
        """
        Get the list of Nodes that constitute this Graph.
        """
        return self._graph.nodes

    def erase_node(self, to_erase: fx.Node) -> None:
        """
        Erases a ``Node`` from the ``Graph``. Throws an exception if
        there are still users of that node in the ``Graph``.
        """
        return self._graph.erase_node(to_erase)

    def inserting_before(self, n: Optional[fx.Node] = None) -> fx.graph._InsertPoint:
        """
        Sets the point at which we will insert the graph.
        """
        return self._graph.inserting_before(n)

    # pyre-ignore
    def get_attr(self, qualified_name: str, type_expr: Optional[Any] = None) -> fx.Node:
        """
        Inserts a ``get_attr`` node into the Graph.
        """
        node = self._graph.get_attr(qualified_name, type_expr)

        # Gets the actual value of the attribute if it exists so that we can use
        # it to set the 'spec' metadata
        def _maybe_get_attr_value(
            mod: torch.nn.Module, qualified_name: str
        ) -> Optional[torch.Tensor]:
            module_path, _, name = qualified_name.rpartition(".")

            try:
                submod: torch.nn.Module = mod.get_submodule(module_path)
            except AttributeError:
                warnings.warn(f"Failed to fetch module {module_path}!", stacklevel=1)
                return None

            # See if the value is a buffer
            if name in submod._buffers:
                return submod._buffers[name]

            # See if the value is a parameter
            if hasattr(submod, name):
                attr = getattr(submod, name)
                if isinstance(attr, torch.nn.Parameter):
                    return attr

            return None

        buffer = _maybe_get_attr_value(self.owning_module, qualified_name)
        if buffer is not None:
            node.meta["spec"] = TensorSpec.from_tensor(buffer, True)

        return node
