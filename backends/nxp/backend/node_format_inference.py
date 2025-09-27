# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator

from executorch.backends.nxp.backend.edge_program_converter import functions_converters
from executorch.backends.nxp.backend.node_format import NodeFormat, NXP_NODE_FORMAT
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from torch.export import ExportedProgram
from torch.fx import Node

logger = logging.getLogger(__name__)


class NodeFormatInference:
    # Dictionary with Edge Aten ops that always use channels first format.
    # The op in the dictionary is mapped to a dictionary, which holds indices to input nodes
    # that are always channels first.
    ops_with_channels_first_nodes = {
        exir_ops.edge.aten.avg_pool2d.default: {"inputs": [0]},
        exir_ops.edge.aten.convolution.default: {"inputs": [0, 1]},
        exir_ops.edge.aten.max_pool2d_with_indices.default: {"inputs": [0]},
        exir_ops.edge.aten.max_pool2d.default: {"inputs": [0]},
    }

    # A set of Edge Aten ops, which have the ability to change the format (for example - input nodes
    # are channels first but output is formatless).
    ops_that_can_change_tensor_format = {exir_ops.edge.aten.view_copy.default}

    _type_changed_during_last_run: bool

    # Mapping between Node and its ancestors (inputs)
    _node_inputs: dict[Node, list[Node]]

    # Mapping between Node and its children (outputs)
    _node_outputs: dict[Node, list[Node]]

    # List of all edge operations, which are supported by the converter.
    _known_targets: list[EdgeOpOverload]

    def __init__(self, edge_program: ExportedProgram):
        self._edge_program = edge_program

        self._nodes = edge_program.graph.nodes
        self._node_inputs = {
            node: node.all_input_nodes for node in edge_program.graph.nodes
        }
        self._node_outputs = {
            node: list(node.users.keys()) for node in edge_program.graph.nodes
        }

        self._type_changed_during_last_run = False

        self._known_targets = list(functions_converters) + [
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            operator.getitem,
        ]

    def identify_node_formats(self):
        self._type_changed_during_last_run = True

        # Re-run format inference until there are no changes
        while self._type_changed_during_last_run:
            self._type_changed_during_last_run = False

            for node in self._nodes:
                self._infer_format_of_nodes(node)

        for node in self._nodes:
            if self._get_node_op_type(node) is None:
                continue
            if not hasattr(node, "meta"):
                logging.warning(f"Node `{node}` does not have the `meta` attribute.")
                node.meta = {}
            if NXP_NODE_FORMAT not in node.meta:
                logging.warning(f"Node `{node}` does not have inferred format.")
                node.meta[NXP_NODE_FORMAT] = NodeFormat.NONE

    def _infer_format_of_nodes(self, node: Node):
        op_type = self._get_node_op_type(node)

        if op_type in self.ops_with_channels_first_nodes:
            self._handle_node_which_uses_channels_first_format(node)
        elif op_type in self.ops_that_can_change_tensor_format:
            if op_type == exir_ops.edge.aten.view_copy.default:  # view_copy
                self._assign_format_to_node(
                    self._node_outputs[node][0], NodeFormat.FORMATLESS
                )
            else:
                logger.error(
                    f"Node format inference for node type: {op_type} not found!"
                )
        elif node.op != "call_function" or (
            hasattr(node, "target") and node.target in self._known_targets
        ):
            # Generic node, or tensor.
            self._handle_node_which_can_use_any_node_format(node)

        else:
            # Don't infer the format for unknown nodes. These nodes will never be delegated, so they will divide
            #  delegated partitions. Propagating the format here could unnecessarily enforce the format in one of these
            #  partitions, which would require extra transpositions.
            for processed_node in self._node_inputs[node] + [node]:
                self._assign_format_to_node(processed_node, NodeFormat.NONE)

    def _infer_format_based_on_io_ranks(self, node: Node):
        """Determine the format of the output tensor of given "reshape style operator" based on the ranks of its input
        and output.
        """
        # noinspection PyBroadException
        try:
            main_input_rank = len(node.all_input_nodes[0].meta["val"].shape)
            main_output_rank = len(node.meta["val"].shape)

            if main_output_rank == main_input_rank:
                # Operator maintains the number of dimensions -> try to propagate the format.
                self._match_formats_of_nodes(node, node.prev)

            else:
                # Either the op 'flattens' the tensor, so output is formatless, or it scales it up, in which case the
                # format is assumed to be 'FORMATLESS', and may be back propagated as channels first later.
                self._assign_format_to_node(node, NodeFormat.FORMATLESS)

        except:
            # Some shape data is not known, so we cannot be extra clever. Just set the output to `FORMATLESS` and
            #  everything will be alright.
            self._assign_format_to_node(node, NodeFormat.FORMATLESS)

    def _match_formats_of_nodes(self, node_1, node_2):
        """If one of 'node_1' or 'node_2' is channels first, make the other channels first as well.
        If neither is channels first, make them both formatless.
        """

        format_1 = self._get_node_format(node_1)
        format_2 = self._get_node_format(node_2)

        if format_1.is_channels_first() or format_2.is_channels_first():
            # At least 1 is channels first
            if not format_1.is_channels_first():
                self._assign_format_to_node(node_1, NodeFormat.CHANNELS_FIRST)
            elif not format_2.is_channels_first():
                self._assign_format_to_node(node_2, NodeFormat.CHANNELS_FIRST)

        else:
            self._assign_format_to_node(node_1, NodeFormat.FORMATLESS)
            self._assign_format_to_node(node_2, NodeFormat.FORMATLESS)

    def _assign_format_to_node(self, node: Node, node_format: NodeFormat):
        """
        Assign format to node, but only if it's not channels first.
        """
        old_node_format = self._get_node_format(node)

        if old_node_format is NodeFormat.CHANNELS_FIRST:
            # Once CHANNEL_FIRST was assigned, we don't want to reassign
            return

        if node_format is NodeFormat.NONE and old_node_format is not NodeFormat.NONE:
            # A format has already been assigned to the node before. Don't replace it with `NONE`.
            return

        if old_node_format != node_format:
            self._type_changed_during_last_run = True

        node.meta[NXP_NODE_FORMAT] = node_format

    def _get_node_op_type(self, node: Node) -> str | None:
        """
        Get node's operation type or None if node is not callable function.
        """
        if node.op == "call_function":
            return node.target

        return None

    def _handle_node_which_uses_channels_first_format(self, node: Node):
        """
        Function for assigning format to nodes that require channels first input (Conv, MaxPool etc.)
        """
        op_type = self._get_node_op_type(node)

        for index, ancestor_node in enumerate(self._node_inputs[node]):
            # Go through input nodes and assign them correct format
            if index in self.ops_with_channels_first_nodes[op_type]["inputs"]:
                self._assign_format_to_node(ancestor_node, NodeFormat.CHANNELS_FIRST)

                # We need to propagate channels first format up to already visited nodes
                self._propagate_channels_first_format_up(ancestor_node)
            else:
                self._assign_format_to_node(ancestor_node, NodeFormat.FORMATLESS)

        # (TODO Lukas Sztefek): It is expected here, that CHANNELS_FIRST node always produces CHANNELS_FIRST output.
        # Validate the assumption.
        self._assign_format_to_node(node, NodeFormat.CHANNELS_FIRST)

    def _handle_node_which_can_use_any_node_format(self, node: Node):
        """
        Function for assigning format to nodes that don't care about format (Softmax, Abs).
        It stays formatless if there is no surrounding channels first ancestor/child node.
        """
        if not self._node_produces_or_consumes_channels_first_format(node):
            # Nor inputs or current node are channels first -> assign everything to formatless
            for processed_node in self._node_inputs[node] + [node]:
                self._assign_format_to_node(processed_node, NodeFormat.FORMATLESS)

        else:
            # Node produces or consumes channels first content
            for processed_node in self._node_inputs[node] + [node]:
                is_0d_to_2d = self._node_product_has_0_to_2_dimensions(processed_node)

                if self._get_node_format(processed_node).is_channels_first():
                    # Node output already channel first
                    continue
                elif is_0d_to_2d:
                    # Node has less than 3 dimensions so it cannot be considered CHANNELS_FIRST
                    self._assign_format_to_node(processed_node, NodeFormat.FORMATLESS)
                else:
                    # Node has more than 2D output -> make it channels first
                    self._assign_format_to_node(
                        processed_node, NodeFormat.CHANNELS_FIRST
                    )
                    self._propagate_channels_first_format_up(processed_node)

    def _propagate_channels_first_format_up(self, node: Node):
        if self._node_is_placeholder(node):
            # Input or buffer node -> there is no parent node so we can end propagation here
            self._assign_format_to_node(node, NodeFormat.CHANNELS_FIRST)
            return

        if node in self.ops_that_can_change_tensor_format:
            # Propagation ends here because processed node changing format.
            return

        for ancestor_node in self._node_inputs[node]:
            # Propagate channels first to ancestor nodes
            self._infer_format_of_nodes(ancestor_node)

    def _node_product_has_0_to_2_dimensions(self, node: Node) -> bool:
        assert "val" in node.meta, f"Node '{node.name}' doesn't contain 'val' metadata!"

        node_value_meta = node.meta["val"]

        # (TODO Lukas Sztefek): Some nodes contains multiple value metadata (MaxPool, ...). Find out why.
        if isinstance(node_value_meta, tuple):
            node_value_meta = node_value_meta[0]
        elif isinstance(node_value_meta, list):
            node_value_meta = node_value_meta[0]

        node_output_rank = len(node_value_meta.shape)

        return 0 <= node_output_rank <= 2

    def _node_produces_or_consumes_channels_first_format(self, node) -> bool:
        """
        Check if node itself produces output in channels first format or consumes it from ancestor node.
        """
        if self._get_node_format(node).is_channels_first():
            return True

        input_nodes = self._node_inputs[node]
        return any(
            self._get_node_format(ancestor_node).is_channels_first()
            for ancestor_node in input_nodes
        )

    def _get_node_format(self, node) -> NodeFormat:
        if not hasattr(node, "meta"):
            node.meta = {}
        return node.meta.get(NXP_NODE_FORMAT, NodeFormat.NONE)

    def _node_is_placeholder(self, node: Node) -> bool:
        return node.op == "placeholder"
