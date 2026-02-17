# This pass exists to propagate input spec metadata down through nested
# submodules. Specifically, metadata for the type of tensor - USER_INPUT, PARAM,
# BUFFER, corresponding to torch.export.graph_signature.InputKind. If the tensor
# is not a direct input in all paths, it's left left as None.
#
# Metadata is stored in the node meta["input_spec"] with a type of
# torch.export.graph_signature.InputSpec or None. It corresponds to the output
# value of the node, and can be a tuple for nodes that return tuples.
#
# After this pass runs, it should be present on all nodes, including arbitrarily
# nested submodules. This may become stale if the graph is mutated, though.

import torch

from torch.export import ExportedProgram
from torch.export.graph_signature import InputSpec
from torch.fx import GraphModule, Node
from typing import Any, Sequence

INPUT_SPEC_KEY = "input_spec"

def propagate_input_spec(ep: ExportedProgram) -> ExportedProgram:
    inputs = {
        s.arg.name: s for s in ep.graph_signature.input_specs
    }
    _propagate_input_spec_recursive(ep.graph_module, inputs)

def _collect_node_arg_specs(args: Sequence[Any]) -> list[InputSpec | None]:
    """
    Retrieve the input spec for each node arg.
    """
    return [
        n.meta.get(INPUT_SPEC_KEY, None) if hasattr(n, "meta") else None
        for n in args
    ]

def _propagate_input_spec_recursive(gm: GraphModule, inputs: dict[str, InputSpec] | Sequence[InputSpec]) -> None:
    """
    Given a dictionary or list of InputSpecs for graph inputs, propagate the specs
    to any nested submodules.
    """
    # Submodules don't have graph signatures, so we need to reconstruct the
    # placeholder -> spec mapping based on placeholder node order.
    if not isinstance(inputs, dict):
        input_dict = {}

        # This relies on placeholder node order matching graph inputs - but
        # this seems to be an implicit contract that pytorch already uses...
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                input_dict[node.target] = inputs[len(input_dict)]

        inputs = input_dict

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            _update_placeholder_meta(node, inputs)
        elif node.target == torch.ops.higher_order.cond:
            _update_cond_meta(node, inputs)
        elif node.target == torch.ops.higher_order.map_impl:
            _update_map_meta(node, inputs)
        elif node.target == torch.ops.higher_order.scan:
            _update_scan_meta(node, inputs)

def _update_placeholder_meta(node: Node, inputs: dict[str, InputSpec]) -> None:
    spec = inputs.get(node.target, None)
    existing_meta = node.meta.get(INPUT_SPEC_KEY, None)

    if spec is not None:
        # If this corresponds to a EP input and the node isn't already tagged,
        # set it. If there is a conflict, default to not tagging.
        if existing_meta is None:
            node.meta[INPUT_SPEC_KEY] = spec
        else:
            node.meta.pop(INPUT_SPEC_KEY, None)
    else:
        # Not an EP input.
        node.meta.pop(INPUT_SPEC_KEY, None)

def _update_cond_meta(node: Node, inputs: dict[str, InputSpec]) -> None:
    _, true_submodule_node, false_submodule_node, submodule_inputs = node.args
    submodule_input_specs = _collect_node_arg_specs(submodule_inputs)

    # Resolve get_attr nodes to actual submodules
    gm = node.graph.owning_module
    true_submodule = getattr(gm, true_submodule_node.target)
    false_submodule = getattr(gm, false_submodule_node.target)

    _propagate_input_spec_recursive(true_submodule, submodule_input_specs)
    _propagate_input_spec_recursive(false_submodule, submodule_input_specs)

def _update_map_meta(node: Node, inputs: dict[str, InputSpec]) -> None:
    f_node, mapped_args, operands = node.args
    mapped_arg_specs = _collect_node_arg_specs(mapped_args)
    operand_specs = _collect_node_arg_specs(operands)
    submodule_input_specs = mapped_arg_specs + operand_specs

    # Resolve get_attr node to actual submodule
    gm = node.graph.owning_module
    f = getattr(gm, f_node.target)

    _propagate_input_spec_recursive(f, submodule_input_specs)

def _update_scan_meta(node: Node, inputs: dict[str, InputSpec]) -> None:
    combine_fn_node, init, xs, additional_inputs = node.args
    init_specs = _collect_node_arg_specs(init)
    xs_specs = _collect_node_arg_specs(xs)
    additional_input_specs = _collect_node_arg_specs(additional_inputs)
    submodule_input_specs = init_specs + xs_specs + additional_input_specs

    # Resolve get_attr node to actual submodule
    gm = node.graph.owning_module
    combine_fn = getattr(gm, combine_fn_node.target)

    _propagate_input_spec_recursive(combine_fn, submodule_input_specs)
