from __future__ import annotations

import logging
from typing import Optional, Sequence, Set

import torch
from torch.fx.node import _get_qualified_name
from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.utils import get_torch_inputs

logger = logging.getLogger(__name__)


def _extract_downstream_get_nodes(
    module_node: torch.fx.Node, output_indices: Set[int]
) -> Sequence[torch.fx.Node]:
    """Extracts downstream users of a node which get the item at a particular index

    Certain module-type nodes have multiple outputs (tuple of outputs). This function
    returns downstream nodes which call the _operator.getitem function, which extracts
    the element at a particular index in the tuple

    Args:
        module_node: FX module-type node to analyze
        output_index: Indices in the module node output to search for
    Returns:
        List of nodes which get the item at the specified index in the module node output
    """
    get_nodes = []

    # Iterate over all downstream users of the node object
    for user in module_node.users:
        # If the user is a "get" node accessing the specified index, store it
        if _get_qualified_name(user.target) == "_operator.getitem" and (
            user.args[1] in output_indices
        ):
            get_nodes.append(user)

    return get_nodes


def _repair_64bit_input(
    gm: torch.fx.GraphModule,
    position: int,
    submodule_name: str,
    submodule_outputs: Optional[torch.Tensor | Sequence[torch.Tensor]],
    dtype: torch.dtype,
) -> None:
    """Fixes a single Long/Double input to a TRT-accelerated subgraph

    In-Place modifies the provided graph

    Inserts a cast to the 32-bit equivalent type for TRT, then if necessary,
    inserts an upcast back to the 64-bit type for subsequent Torch operations

    Args:
        gm: FX GraphModule enclosing the TRT subgraph
        position: Index in the submodule inputs at which the long or double input is found
        submodule_name: Name of TRT-accelerated subgraph module in FX graph
        submodule_outputs: Output tensor(s) of TRT-accelerated subgraph (used for dtypes/structure)
        dtype: Data type of tensor at position in submodule (double/long)
    """
    assert dtype in (
        torch.float64,
    ), f"dtype argument must be torch.float64, got {dtype}"

    logger.info(
        f"Downcasting a 64-bit input at position {position} of submodule {submodule_name}"
    )

    # Determine target data type in 32 and 64 bit forms
    dtype_64bit = dtype
    dtype_32bit = torch.float32

    # Find the node representing the submodule in the graph
    module_node = None

    # Iterate over all nodes in the graph, seeking target module name match
    for n in gm.graph.nodes:
        if n.op == "call_module" and str(n.target) == submodule_name:
            module_node = n
            break

    if module_node is None:
        raise AssertionError(
            f"Sought module node {submodule_name}, could not find in graph:\n{gm.graph}"
        )

    # Extract the 64-bit node of the input
    node_64bit = module_node.all_input_nodes[position]

    # Prior to the module, insert a cast to the 32-bit equivalent node
    with gm.graph.inserting_before(module_node):
        node_32bit = gm.graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(node_64bit,),
            kwargs={"dtype": dtype_32bit},
        )

    # Replace 64-bit input to TRT module with new 32-bit cast node
    module_node.replace_input_with(node_64bit, node_32bit)

    output_positions_64bit = set()

    # Determine if any outputs of the model are 64-bit type and store their indices
    if submodule_outputs is not None:
        outputs_list = (
            [submodule_outputs]
            if isinstance(submodule_outputs, torch.Tensor)
            else submodule_outputs
        )

        for output_position, output in enumerate(outputs_list):
            if output.dtype == dtype_64bit:
                output_positions_64bit.add(output_position)

    # Only enter this code block if there exists a 64-bit output
    # This implies a cast is needed, since TRT cannot output 64-bit tensors
    if output_positions_64bit:
        # Determine whether the outputs of the module are tuple-type or not
        is_collection_output = False
        if isinstance(submodule_outputs, tuple):
            is_collection_output = True

        if not is_collection_output:
            # If the output is a single tensor, insert a cast back to int64
            with gm.graph.inserting_after(module_node):
                cast_node_64bit = gm.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(module_node,),
                    kwargs={"dtype": dtype_64bit},
                )

            # Replace all uses of the TRT module (except the cast node) with the 64-bit equivalent
            module_node.replace_all_uses_with(
                cast_node_64bit, delete_user_cb=lambda user: (user != cast_node_64bit)
            )

        else:
            # If the output is a tuple of tensors, extract downstream users for each 64-bit output
            get_nodes = _extract_downstream_get_nodes(
                module_node, output_positions_64bit
            )

            # For each downstream user, append a cast node back to the 64-bit precision
            for get_node in get_nodes:
                with gm.graph.inserting_after(get_node):
                    cast_node_64bit = gm.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        args=(get_node,),
                        kwargs={"dtype": torch.float64},
                    )

                get_node.replace_all_uses_with(
                    cast_node_64bit,
                    delete_user_cb=lambda user: (user != cast_node_64bit),
                )

    # Clean up graph and ensure invariants are preserved
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def repair_double_inputs(
    parent_graph: torch.fx.GraphModule,
    submodule: torch.fx.GraphModule,
    submodule_inputs: Sequence[Input],
    device: torch.device,
    submodule_name: Optional[str] = None,
) -> Sequence[Input]:
    """Fixes all Long/Double type inputs to a TRT-accelerated subgraph

    In-Place modifies the provided graph

    Inserts a cast to the 32-bit equivalent type for TRT, then if necessary,
    inserts an upcast back to the 64-bit type for subsequent Torch operations

    Args:
        parent_graph: FX GraphModule enclosing the TRT subgraph
        submodule: Child submodule to repair inputs on
        submodule_inputs: Input tensor(s) of TRT-accelerated subgraph (used for dtypes/structure)
        submodule_name: Optionally specify the name of the submodule target in the parent graph
    Returns:
        New submodule inputs, updated accordingly with long/double truncation
    """
    submodule_torch_inputs = get_torch_inputs(submodule_inputs, device)
    num_submodule_inputs = len(submodule_inputs)
    repaired_outputs_once = False

    # For each input to the TRT subgraph, check if its type is long/double
    for position in range(num_submodule_inputs):
        param = submodule_torch_inputs[position]

        # If the data type of the input is long/double, insert necessary
        # casts to replace the operation
        if isinstance(param, torch.Tensor) and param.dtype == torch.float64:
            # Ensure outputs are only repaired once per submodule to avoid
            # unnecessary ops showing up in the graph
            if not repaired_outputs_once:
                submodule_outputs = submodule(*submodule_torch_inputs)

            _repair_64bit_input(
                parent_graph,
                position,
                submodule_name if submodule_name is not None else submodule._get_name(),
                None if repaired_outputs_once else submodule_outputs,
                param.dtype,
            )

            repaired_outputs_once = True

            # Repair submodule inputs in accordance with inserted casts
            dtype_32bit = torch.float32
            submodule_torch_inputs = (
                list(submodule_torch_inputs[:position])
                + [
                    param.to(dtype_32bit),
                ]
                + list(submodule_torch_inputs[position + 1 :])
            )

            # Set the 32bit inputs and their types to the submodule Inputs
            for idx in range(len(submodule_inputs)):
                submodule_inputs[idx].torch_tensor = submodule_torch_inputs[idx]
                submodule_inputs[idx].dtype = dtype._from(
                    submodule_torch_inputs[idx].dtype
                )

    return submodule_inputs
