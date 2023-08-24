# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import Dict, List, Tuple, Union

import torch
import torch.utils._pytree as pytree
from executorch.exir import CallSpec, ExportGraphSignature
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.delegate import executorch_call_delegate

from executorch.exir.graph_module import _get_submodule

from executorch.exir.tracer import Value
from torch._export.exported_program import ExportedProgram
from torch._subclasses import FakeTensor
from torch.fx.passes.utils.fuser_utils import (
    erase_nodes,
    fuse_as_graphmodule,
    insert_subgm,
    legalize_graph,
    NodeList,
    topo_sort,
)


class LoweredBackendModule(torch.nn.Module):
    """
    A subclass of nn.Module that is generated for modules containing
    delegated functions. This is can be created by calling `to_backend`.

    Private Attributes:
        * **backend_id**: The backend's name
        * **processed_bytes**: The delegate blobs created from backend.preprocess
        * **compile_specs**: A list of backend-specific objects with static
            metadata to configure the "compilation" process.
        * **original_module**: The original EXIR module
    """

    _backend_id: str
    _processed_bytes: bytes
    _compile_specs: List[CompileSpec]
    _original_module: ExportedProgram

    def __init__(
        self,
        edge_program: ExportedProgram,
        backend_id: str,
        processed_bytes: bytes,
        compile_specs: List[CompileSpec],
    ) -> None:
        super().__init__()
        self._original_module = edge_program
        self._backend_id = backend_id
        self._processed_bytes = processed_bytes
        self._compile_specs = compile_specs

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def processed_bytes(self) -> bytes:
        return self._processed_bytes

    @property
    def compile_specs(self) -> List[CompileSpec]:
        return self._compile_specs

    @property
    def original_module(self) -> ExportedProgram:
        return self._original_module

    # Used to patch each delegated function with a call_delegate call
    # @staticmethod
    def forward(
        self,
        *args: Value,
        **kwargs: Tuple[Value, ...],
    ) -> Value:
        return executorch_call_delegate(self, *args)


# TODO(zhxchen17) Try ExportPass
def _fixup_output_node(gm: torch.fx.GraphModule) -> None:
    for node in reversed(gm.graph.nodes):
        if node.op == "output":
            with gm.graph.inserting_before(node):
                assert len(node.args) == 1
                outputs = node.args[0]
                if isinstance(outputs, torch.fx.Node):
                    val = outputs.meta.get("val")
                    if isinstance(val, list):
                        # If a list is returned, in some cases it is represented as a
                        # singular node, like `split_copy_tensor` but EXIR will return a
                        # opened-up list like `[getitem1, getitem2]`
                        outputs = [
                            torch.fx.Proxy(outputs)[i].node for i in range(len(val))
                        ]
            returns, out_spec = pytree.tree_flatten(outputs)
            node.args = (returns,)
            return


def arrange_graph_placeholders(
    gm: torch.fx.GraphModule, owning_program: ExportedProgram
) -> torch.fx.GraphModule:
    """
    Modifies the graph of the given graphmodule with one that contains the same nodes as the original,
    but with placeholders in order of (Params + Buffers) (User Inputs)

    This is used by the delegate api which disturbs the placeholder ordering when creating a submodule
    from partitioned nodes

    Args:
        gm: The graph module that we want arranged
        owning_program: ExportedProgram that the submodule (gm) belongs to

    Returns:
        The graph module in-placed arranged
    """
    new_graph = torch.fx.Graph()

    node_map = {}  # mapping of nodes from old graph to new graph

    graph_sign = owning_program.graph_signature

    # Add all placeholders into the graph first:
    param_nodes = []
    buffer_nodes = []
    input_nodes = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue

        if node.name in graph_sign.inputs_to_parameters:
            param_nodes.append(node)
        elif node.name in graph_sign.inputs_to_buffers:
            buffer_nodes.append(node)
        else:
            input_nodes.append(node)

    for param_node in param_nodes:
        new_node = new_graph.node_copy(param_node, lambda x: node_map[x])
        node_map[param_node] = new_node
    for buffer_node in buffer_nodes:
        new_node = new_graph.node_copy(buffer_node, lambda x: node_map[x])
        node_map[buffer_node] = new_node
    for input_node in input_nodes:
        new_node = new_graph.node_copy(input_node, lambda x: node_map[x])
        node_map[input_node] = new_node

    # Now add all the other nodes in order
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue

        new_node = new_graph.node_copy(node, lambda x: node_map[x])
        node_map[node] = new_node

    # lint to ensure correctness
    new_graph.lint()

    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph

    return gm


def _get_new_signature(
    original_program: ExportedProgram, gm: torch.fx.GraphModule
) -> Tuple[ExportGraphSignature, Dict[str, Union[torch.Tensor, torch.nn.Parameter]]]:
    old_signature = original_program.graph_signature

    new_signature = ExportGraphSignature(
        parameters=[],
        buffers=[],
        user_inputs=[],
        user_outputs=[],
        inputs_to_parameters={},
        inputs_to_buffers={},
        buffers_to_mutate={},
        backward_signature=None,
    )
    new_state_dict = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in old_signature.inputs_to_parameters:
                parameter_name = old_signature.inputs_to_parameters[node.name]
                # add param to graph signature
                new_signature.parameters.append(parameter_name)
                new_signature.inputs_to_parameters[node.name] = parameter_name

                # add param to state_dict
                new_state_dict[parameter_name] = original_program.state_dict[
                    parameter_name
                ]
            elif node.name in old_signature.inputs_to_buffers:
                buffer_name = old_signature.inputs_to_buffers[node.name]
                # add buffer to graph signature
                new_signature.buffers.append(buffer_name)
                new_signature.inputs_to_buffers[node.name] = buffer_name

                # add param to new_state_dict
                new_state_dict[buffer_name] = original_program.state_dict[buffer_name]
            else:
                # not param or buffer then user input
                new_signature.user_inputs.append(node.name)
        if node.op == "output":
            for output in node.all_input_nodes:
                new_signature.user_outputs.append(output.name)

    return new_signature, new_state_dict


def create_exported_program_from_submodule(
    submodule: torch.fx.GraphModule,
    owning_program: ExportedProgram,
) -> ExportedProgram:
    """
    Creates an ExportedProgram from the given submodule using the parameters and buffers
    from the top-level owning program

    Args:
        submodule: submodule to create and exported program from
        owning_program: exported program containing the parameters and buffers used within
                the submodule

    Returns:
        The ExportedProgram created from submodule
    """
    # Arrange the submodule's placeholders in order
    submodule = arrange_graph_placeholders(submodule, owning_program)

    # Get updated graph signature
    subgraph_signature, subgraph_state_dict = _get_new_signature(
        owning_program, submodule
    )

    return ExportedProgram(
        root=submodule,
        graph=submodule.graph,
        graph_signature=subgraph_signature,
        call_spec=CallSpec(None, None),
        state_dict=subgraph_state_dict,
        range_constraints=copy.deepcopy(owning_program.range_constraints),
        equality_constraints=[],
        module_call_graph=[],
    )


def create_submodule_from_nodes(
    gm: torch.fx.GraphModule,
    node_list: NodeList,
    tag: str,
    skip_legalize_graph: bool = False,
) -> Tuple[torch.fx.GraphModule, torch.fx.Node]:
    """
    Modifies the given graph module in-place to separate out the given nodes
    into a submodule. The given node_list should form a fully connected
    subgraph.

    Args:
        gm: The graph module that we want to partition
        node_list: A list of nodes that belong in the partition

    Returns:
        The submodule that has been partitioned, the call_module node in the
        toplevel graph module calling the submodule
    """
    sorted_nodes = topo_sort(node_list)

    submodule_name = "fused_" + tag
    sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
        gm, sorted_nodes, submodule_name
    )

    _fixup_output_node(sub_gm)

    gm = insert_subgm(gm, sub_gm, orig_inputs, orig_outputs)
    if len(orig_outputs) == 1 and isinstance(orig_outputs[0].meta["val"], FakeTensor):
        # If the original output is a single tensor, it has been
        # pytree.tree_flatten-ed to be a singleton list, so we want to replace
        # all uses with a getitem call to the 0th index of the result
        for node in gm.graph.nodes:
            if node.op == "call_module":
                with gm.graph.inserting_after(node):
                    proxy_out = torch.fx.Proxy(node)[0].node  # type: ignore[index]
                    node.replace_all_uses_with(proxy_out, propagate_meta=True)
                    # Reset the args since it was overwritten in the previous line
                    proxy_out.args = (node, 0)

    erase_nodes(gm, sorted_nodes)

    # Topological sort original gm with newly created sub_gm
    # TODO : T153794167 Get rid of support for skipping legalize graph in create_submodule_from_nodes
    # once we transition to using fuse_by_partitions.
    if not skip_legalize_graph:
        legalize_graph(gm)

    # Get the call_module node
    submodule_node = None
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == submodule_name:
            submodule_node = node
        elif node.op == "call_module":
            raise RuntimeError(
                f"The submodule created with nodes {node_list} did not form \
                one fully contained subgraph. Check that these nodes form a \
                fully contained graph. Partitioned graph: {gm.graph}."
            )

    assert (
        submodule_node is not None
    ), f"No submodule was created with the nodes {node_list} in the graph {gm.graph}"

    return sub_gm, submodule_node


def get_lowered_submodules(
    graph_module: torch.fx.GraphModule,
) -> List[Tuple[str, LoweredBackendModule, torch.fx.Node]]:
    """
    Returns a list of lowered modules that are in the given graph (does not look
    into submodules). Specifically, the returned value is a list containing a
    tuple of (name of the lowered module that's stored in the graph module, the
    lowered module itself, and the fx node that called this lowered module).
    """
    lowered_submodules = []
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == executorch_call_delegate:
            name, module, node = _get_submodule(graph_module, node, 0)
            assert isinstance(module, LoweredBackendModule)
            lowered_submodules.append((name, module, node))
    return lowered_submodules


def get_lowered_backend_modules(
    graph_module: torch.fx.GraphModule,
) -> List[LoweredBackendModule]:
    """
    Returns a list of exported programs which were lowered by backen delegates
    """
    lowered_programs = []
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == executorch_call_delegate:
            lowered_backend_module = getattr(graph_module, node.args[0].name)
            lowered_programs.append(lowered_backend_module)

    return lowered_programs
