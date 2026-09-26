# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Collection

import torch

from executorch.backends.transforms.utils import (
    get_param_tensor,
    is_param_node,
    set_param_tensor,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertConv1dToConv2dPass(ExportPass):
    """Express Conv1d as Conv2d with a unit-height spatial dimension.

    At the top level, only convolutions with lifted parameter, buffer, or
    constant weights are converted. Dynamic weights and unfolded QDQ weights
    are left unchanged.
    Weights passed across control-flow boundaries are unsqueezed inside the
    nested graph so its input signature remains unchanged.
    Root graph-local weights are skipped by default. Backends may provide an
    allow-list of safe producer targets when constructing the pass.
    The pass emits squeeze and unsqueeze boundaries for each converted
    convolution; downstream view cleanup can fold adjacent boundaries.
    """

    def __init__(
        self,
        exported_program: ExportedProgram,
        graph_local_weight_targets: Collection[torch.fx.node.Target] = (),
    ) -> None:
        """Initialize the transform.

        Args:
            exported_program (ExportedProgram): Program containing the graph
                and its lifted weights.
            graph_local_weight_targets (Collection[torch.fx.node.Target]):
                Producer targets whose three-dimensional outputs may be used
                as Conv1d weights. All graph-local weights are skipped when
                this is empty.
        """
        super().__init__()
        self.exported_program = exported_program
        self._graph_local_weight_targets = frozenset(graph_local_weight_targets)

    def _unsqueeze_weight(self, weight_node: torch.fx.Node) -> bool:
        weight = get_param_tensor(self.exported_program, weight_node)
        if weight is None:
            return False
        if weight.dim() == 4:
            return True
        if weight.dim() != 3:
            return False

        weight_2d = weight.unsqueeze(2).contiguous()
        set_param_tensor(self.exported_program, weight_node, weight_2d)
        weight_node.meta["val"] = weight_node.meta["val"].unsqueeze(2)
        return True

    def _conv1d_weight_node(self, node: torch.fx.Node) -> torch.fx.Node | None:
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.aten.convolution.default
            or len(node.args) != 9
        ):
            return None
        weight_node = node.args[1]
        if not isinstance(weight_node, torch.fx.Node) or not is_param_node(
            self.exported_program, weight_node
        ):
            return None
        weight = get_param_tensor(self.exported_program, weight_node)
        weight_meta = weight_node.meta.get("val")
        if (
            not isinstance(weight, torch.Tensor)
            or weight.dim() != 3
            or not isinstance(weight_meta, torch.Tensor)
            or weight_meta.dim() != 3
        ):
            return None
        return weight_node

    @staticmethod
    def _is_conv1d(node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and node.target == exir_ops.edge.aten.convolution.default
            and len(node.args) == 9
            and len(node.args[3]) == 1
        )

    @staticmethod
    def _convert_node(
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        input_node: torch.fx.Node,
        weight_node: torch.fx.Node,
        unsqueeze_weight: bool,
    ) -> None:
        """Rewrite one Conv1d node as an equivalent Conv2d operation.

        The rewrite adds a unit-height dimension to the input, expands the
        convolution attributes to two spatial dimensions, and removes the
        unit-height dimension from the output. When requested, it also inserts
        an unsqueeze that produces a weight with shape ``[O, I, 1, K]`` from
        the original ``[O, I, K]`` weight.

        Args:
            graph (torch.fx.Graph): Graph containing the convolution.
            node (torch.fx.Node): Conv1d node to rewrite.
            input_node (torch.fx.Node): Convolution input node.
            weight_node (torch.fx.Node): Convolution weight node.
            unsqueeze_weight (bool): Whether to add a graph-local weight
                unsqueeze. Nested control-flow graphs require this to preserve
                their input signatures.
        """
        with graph.inserting_before(node):
            input_2d = graph.call_function(
                exir_ops.edge.aten.unsqueeze_copy.default,
                args=(input_node, 2),
            )
            weight_2d = (
                graph.call_function(
                    exir_ops.edge.aten.unsqueeze_copy.default,
                    args=(weight_node, 2),
                )
                if unsqueeze_weight
                else weight_node
            )

        args = list(node.args)
        args[0] = input_2d
        args[1] = weight_2d
        args[3] = [1, *list(args[3])]  # stride
        args[4] = [0, *list(args[4])]  # padding
        args[5] = [1, *list(args[5])]  # dilation
        args[7] = [0, *list(args[7])]  # output padding
        node.args = tuple(args)

        with graph.inserting_after(node):
            output_1d = graph.call_function(
                exir_ops.edge.aten.squeeze_copy.dim,
                args=(node, 2),
            )
        for user in list(node.users):
            if user is not output_1d:
                user.replace_input_with(node, output_1d)

    def _convert_nested_graph(self, graph_module: torch.fx.GraphModule) -> bool:
        """Convert Conv1d nodes in a control-flow subgraph and its children.

        Each conversion unsqueezes the weight inside the subgraph, preserving
        the 3D weight expected at the control-flow boundary.

        Args:
            graph_module (torch.fx.GraphModule): Subgraph to convert.

        Returns:
            bool: True if this subgraph or one of its children was modified.
        """
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if not self._is_conv1d(node):
                continue
            input_node, weight_node = node.args[:2]
            if not isinstance(input_node, torch.fx.Node) or not isinstance(
                weight_node, torch.fx.Node
            ):
                continue
            weight_meta = weight_node.meta.get("val")
            if not isinstance(weight_meta, torch.Tensor) or weight_meta.dim() != 3:
                continue
            # A nested graph receives its weight through the control-flow
            # interface. Keep that interface 3D and create the 4D Conv2d
            # weight locally instead of changing the value supplied by its
            # parent graph.
            self._convert_node(
                graph,
                node,
                input_node,
                weight_node,
                unsqueeze_weight=True,
            )
            modified = True

        for child in graph_module.children():
            if isinstance(child, torch.fx.GraphModule):
                modified = self._convert_nested_graph(child) or modified

        if modified:
            graph_module.recompile()
        return modified

    def _convert_graph_local_weights(self, graph: torch.fx.Graph) -> bool:
        """Convert supported root Conv1d nodes with graph-local weights.

        A graph-local weight is produced by another graph operation rather
        than stored as a model parameter. Callers choose which producers are
        safe to convert. The original weight remains unchanged; an
        unsqueeze is inserted between the producer and the convolution.

        Args:
            graph (torch.fx.Graph): Root graph whose Conv1d nodes are examined.

        Returns:
            bool: True if at least one Conv1d node was converted.
        """
        modified = False
        for node in list(graph.nodes):
            if not self._is_conv1d(node):
                continue

            input_node, weight_node = node.args[:2]
            if not isinstance(input_node, torch.fx.Node) or not isinstance(
                weight_node, torch.fx.Node
            ):
                continue

            weight_meta = weight_node.meta.get("val")
            weight_source_is_allowed = (
                weight_node.target in self._graph_local_weight_targets
            )
            has_conv1d_weight_shape = (
                isinstance(weight_meta, torch.Tensor) and weight_meta.dim() == 3
            )
            if not weight_source_is_allowed or not has_conv1d_weight_shape:
                continue

            # The weight producer must keep its original output shape. Insert
            # the missing unit-height dimension only on the path to this conv:
            #
            #   graph-local weight [O, I, K]
            #                 |
            #             unsqueeze(2)
            #                 v
            #        Conv2d weight [O, I, 1, K]
            self._convert_node(
                graph,
                node,
                input_node,
                weight_node,
                unsqueeze_weight=True,
            )
            modified = True
        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        is_root = graph_module is self.exported_program.graph_module

        # Preserve the signature through which a parent supplies parameters to
        # a nested graph by using graph-local weight unsqueezes.
        modified = not is_root and self._convert_nested_graph(graph_module)

        # At the root, lifted weights belong to the exported program and can be
        # changed to 4D once instead of being unsqueezed while the model runs.
        candidates = (
            {
                node: weight_node
                for node in graph.nodes
                if (weight_node := self._conv1d_weight_node(node)) is not None
            }
            if is_root
            else {}
        )

        for node, weight_node in candidates.items():
            if any(
                candidates.get(user) is not weight_node
                or any(
                    arg is weight_node
                    for index, arg in enumerate(user.args)
                    if index != 1
                )
                or any(arg is weight_node for arg in user.kwargs.values())
                for user in weight_node.users
            ):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            if not self._unsqueeze_weight(weight_node):
                continue
            self._convert_node(
                graph,
                node,
                input_node,
                weight_node,
                unsqueeze_weight=False,
            )
            modified = True

        # Some backends create graph-local weights known to be safe before this
        # pass. A caller may opt those weights into the same local unsqueeze
        # used by nested graphs without enabling arbitrary dynamic weights.
        if is_root:
            modified = self._convert_graph_local_weights(graph) or modified

        # The normal entry point invokes this call method for the root graph.
        # Explicitly apply the rewrite to its child control-flow GraphModules.
        if is_root:
            for child in graph_module.children():
                if isinstance(child, torch.fx.GraphModule):
                    modified = self._convert_nested_graph(child) or modified

        if not modified:
            return PassResult(graph_module, False)

        graph_module.recompile()
        result = super().call(graph_module)
        return PassResult(result.graph_module, True)
