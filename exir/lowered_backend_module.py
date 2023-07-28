# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple

import torch
import torch.utils._pytree as pytree
from executorch.backends.compile_spec_schema import CompileSpec
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
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue

        if (
            node.name in graph_sign.inputs_to_parameters
            or node.name in graph_sign.inputs_to_buffers
        ):
            # Insert place holder at at beginning if it is parameter
            with new_graph.inserting_before():
                new_node = new_graph.node_copy(node)
        else:
            new_node = new_graph.node_copy(node)
        node_map[node] = new_node

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
