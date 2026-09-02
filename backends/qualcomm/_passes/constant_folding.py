# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from executorch.backends.qualcomm._passes.utils import copy_meta
from executorch.backends.qualcomm.builders.utils import (
    get_parameter,
    is_mutable_buffer_input,
    is_parameter,
)
from executorch.exir.operator.util import _QUANT_PRIMITIVES
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.replace_aten_with_edge_pass import aten_to_edge
from torch._guards import detect_fake_mode
from torch.utils import _pytree as pytree
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


# These are qdq edge targets
_EDGE_QDQ_TARGETS = set(_QUANT_PRIMITIVES) | {
    aten_to_edge(op) for op in _QUANT_PRIMITIVES
}

# copied from executorch/exir/passes/const_prop_pass.py
_PRIMITIVE_TYPES = (
    float,
    int,
    bool,
    str,
    torch.Tensor,
    torch.device,
    torch.dtype,
    torch.layout,
)


class ConstantFolding(ExportPass):
    """
    executorch/exir/passes/const_prop_pass.py runs at to_executorch stage, and it won't
    work if we run the pass at edge stage. This pass is to address the issue.
    One of biggest reason for this pass is that some ops are not supported by QNN and will
    fail during op validation. If the op can be constant folded, then there won't be partitions.


    Folds subgraphs whose leaves are parameters, lifted tensor constants, or
    *non-mutated* buffers. Mutated buffers are intentionally excluded — view ops
    over them would alias mutable storage if folded, and downstream passes /
    `run_decompositions` mis-handle that aliasing.
    """

    _TENSOR_CONSTANT_PREFIX = "_prop_tensor_constant_"

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super().__init__()
        # Run decomposition is required so graph_signature stores information about mutable buffer.
        self.decomposed_aten_program = edge_program.run_decompositions({})

    def _get_const_placeholders(
        self, graph_module: torch.fx.GraphModule
    ) -> dict[torch.fx.Node, torch.Tensor]:
        """
        Find all constant_tensor and store it in a dict {node : tensor}
        The tensor in dict has actual tensor with values, not fake tensor.
        """

        node_to_tensor = {}
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            # Don't fold mutable buffer
            if is_mutable_buffer_input(node, self.decomposed_aten_program):
                continue
            if not is_parameter(node, self.decomposed_aten_program):
                continue
            node_to_tensor[node] = get_parameter(node, self.decomposed_aten_program)
        return node_to_tensor

    def _propagate(  # noqa: C901
        self,
        graph_module: torch.fx.GraphModule,
        node_to_tensor: dict[torch.fx.Node, torch.Tensor],
    ) -> None:
        """
        Iterate the call_function node. compute the output of that node if it can be constant folded,
        and save it in node_to_tensor. Based on torch/fx/graph.py, graph.nodes should always return
        nodes in topological order.
        Example:
        constant_value_1 -> upsample_bicubic2d ----\
                                                    add_1 --------------------------\
                              constant_value_2 ----/                                add_2 ----> output
                                                      user_input(non-constant)------/    
        
        With graph above, when first enter this method, node_to_tensor dict looks like:
        node_to_tensor = {
            constant_value_1_node : constant_value_1_tensor,
            constant_value_2_node : constant_value_2_tensor,
        }
        
        
        At the end of the method, node_to_tensor dict looks like following:
        node_to_tensor = {
            constant_value_1_node : constant_value_1_tensor,
            constant_value_2_node : constant_value_2_tensor,
            upsample_bicubic2d_node : upsample_bicubic2d_output_tensor,
            add_1_node : add_1_output_tensor,
        }
        
        Tensor in dict will be actual values instead of fake tensor.
        For example, add_1_output_tensor value would be the result of adding upsample_bicubic2d_output_tensor and constant_value_2.
        """

        # If previous node is visited, this function won't recursive search all the way back to input node.
        # It should fall into the base case.
        # This prevents recursive search back to input source node everytime.
        def _is_const(arg, node_to_tensor):

            # For args case.
            if isinstance(arg, (tuple, list)):
                return all(_is_const(x, node_to_tensor) for x in arg)

            # For kwargs case
            if isinstance(arg, dict):
                return all(_is_const(x, node_to_tensor) for x in arg.values())

            # If a tensor is optional and not provided, it will be None.
            # Primitive_types is for constants like integers. These should be able to be folded.
            if arg is None or isinstance(arg, _PRIMITIVE_TYPES):
                return True

            if isinstance(arg, torch.fx.Node):
                # Base case
                return arg in node_to_tensor
            else:
                # If there are some unexpected args that doesn't know how to handle, just return False to be safe.
                return False

        # `pytree.tree_map` flattens containers and treats Node as a leaf, so this
        # only ever receives leaves.
        def _get_data(arg, node_to_tensor):
            if arg is None or isinstance(arg, _PRIMITIVE_TYPES):
                return arg
            if isinstance(arg, torch.fx.Node):
                return node_to_tensor.get(arg)
            return None

        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if not _is_const(node.args, node_to_tensor):
                continue
            if not _is_const(node.kwargs, node_to_tensor):
                continue

            # Copied from executorch/exir/passes/const_prop_pass.py
            # Retrieves args and kwargs required for the node to perform inference.
            args_data, kwargs_data = pytree.tree_map(
                lambda x: _get_data(x, node_to_tensor),
                (node.args, node.kwargs),
            )
            # Perform node inference
            with torch.no_grad():
                try:
                    result = node.target(*args_data, **kwargs_data)
                except Exception:
                    logging.warning(
                        f"Unable to fold the node {node.name}. Skip folding.",
                        exc_info=True,
                    )
                    continue

            if isinstance(result, torch.Tensor):
                result = result.detach().clone(memory_format=torch.contiguous_format)

            # Save the node's result to the map.
            node_to_tensor[node] = result

    def _materialize_as_buffer(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        tensor: torch.Tensor,
    ) -> None:
        buffer_name = get_new_attr_name_with_prefix(self._TENSOR_CONSTANT_PREFIX)(
            graph_module
        )
        graph_module.register_buffer(buffer_name, tensor)
        val = node.meta.get("val")
        fake_mode = detect_fake_mode(val) if val is not None else None
        with graph_module.graph.inserting_before(node):
            get_attr_node = graph_module.graph.get_attr(buffer_name)
            get_attr_node.meta = copy_meta(
                node.meta,
                lambda m: (
                    {
                        **m,
                        "val": fake_mode.fake_tensor_converter.from_real_tensor(
                            fake_mode, tensor
                        ),
                    }
                    if fake_mode is not None
                    else m
                ),
            )
        # Replace node's user with const node as input
        node.replace_all_uses_with(get_attr_node)

    def _materialize(
        self,
        graph_module: torch.fx.GraphModule,
        node_to_tensor: dict[torch.fx.Node, torch.Tensor],
    ) -> None:
        """
        The term "boundary" here refers to where node can no longer be folded.
        Boundry will be before add for this graph is const -> relu1 -> sqrt --- > add -> output
                                                                       input _|

        Rules:
        1) When creating buffer, start with reverse order, so just create the buffer before boundary.
        2) Only the boundary buffer will be created, won't recursive trace args and create unused buffer.
        3) For nodes like conv2d with quantizer, preserve dq node right afer weight and bias.
        """

        # Reverse order: process later (more-derived) constants first, so a
        # chain collapses to a single buffer at its boundary.
        # Align with rule 1.
        for node, tensor in reversed(list(node_to_tensor.items())):
            if node.op == "placeholder":
                continue

            # If all users can be constant folded, then don't need to fold at this level.
            # This behavior aligns with rule 2.
            if all(user in node_to_tensor for user in node.users):
                continue

            # Guarding cases like weight -> dq -> conv2d. Don't fold dq here.
            # Aligns with rule 3
            if node.target in _EDGE_QDQ_TARGETS:
                continue

            self._materialize_as_buffer(graph_module, node, tensor)

    def call(self, graph_module: torch.fx.GraphModule):
        node_to_tensor = self._get_const_placeholders(graph_module)
        if len(node_to_tensor) == 0:
            return PassResult(graph_module, False)
        self._propagate(graph_module, node_to_tensor)
        self._materialize(graph_module, node_to_tensor)
        dead_code_elimination_pass(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
