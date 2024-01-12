# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import operator
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from executorch.exir._serialize import _serialize_pte_binary
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.delegate import executorch_call_delegate, get_lowered_module_name
from executorch.exir.emit import emit_program

from executorch.exir.graph_module import _get_submodule

from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.spec_prop_pass import make_spec, SpecPropPass
from executorch.exir.schema import Program

from executorch.exir.tracer import Value

from torch._export.exported_program import ExportedProgram
from torch._subclasses import FakeTensor
from torch.export.exported_program import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
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
    """

    _backend_id: str  # The backend's name
    _processed_bytes: bytes  # The delegate blobs created from backend.preprocess
    _compile_specs: List[
        CompileSpec
    ]  # A list of backend-specific objects with static metadata to configure the "compilation" process.
    _original_module: ExportedProgram  # The original EXIR module

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
        """
        Returns the backends name.
        """
        return self._backend_id

    @property
    def processed_bytes(self) -> bytes:
        """
        Returns the delegate blob created from backend.preprocess
        """
        return self._processed_bytes

    @property
    def compile_specs(self) -> List[CompileSpec]:
        """
        Returns a list of backend-specific objects with static metadata to configure the "compilation" process.
        """
        return self._compile_specs

    @property
    def original_module(self) -> ExportedProgram:
        """
        Returns the original EXIR module
        """
        return self._original_module

    # TODO(chenlai): consolidate the seriailization config with serialize_to_flatbuffer api
    def buffer(
        self,
        extract_delegate_segments: bool = False,
        segment_alignment: int = 4096,
        constant_tensor_alignment: Optional[int] = None,
        delegate_alignment: Optional[int] = None,
    ) -> bytes:
        """
        Returns a buffer containing the serialized ExecuTorch binary.
        """
        out = _serialize_pte_binary(
            program=self.program(),
            extract_delegate_segments=extract_delegate_segments,
            segment_alignment=segment_alignment,
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        )
        return out

    # TODO(chenlai): re-consider recapture instead of manually constructing the program because
    # the meta data construction is done manually.
    def program(self, emit_stacktrace: bool = False) -> Program:
        # Fix autodpes introuces cyclic dependencies:
        # program -> verifier -> lowered_backend_module -> program
        # @manual
        from executorch.exir.program._program import (
            _get_updated_graph_signature,
            _transform,
        )

        """
        Returns the object that represents the ExecuTorch binary before serialization.
        """
        # Creates a new module based on the original module. The original module will
        # look something like following:
        #
        # opcode         name                 target            args                                        kwargs
        # -------------  -------------------  ----------------  ------------------------------------------  --------
        # placeholder    arg0_1               arg0_1            ()                                          {}
        # placeholder    arg1_1               arg1_1            ()                                          {}
        # call_function  aten_repeat_default  *                 (arg1_1, [4, 1])                            {}
        # call_function  aten_mul_tensor      *                 (aten_repeat_default, aten_repeat_default)  {}
        # call_function  aten_add_tensor      *                 (arg1_1, arg1_1)                            {}
        # output         output               output            ([aten_mul_tensor, aten_add_tensor],)       {}
        #
        # if the whole module is lowered, the resulting lowered module look like
        #
        # opcode         name                      target                       args                                kwargs
        # -------------  ------------------------  ---------------------------  ----------------------------------  --------
        # placeholder    arg0_1                    arg0_1                       ()                                  {}
        # placeholder    arg1_1                    arg1_1                       ()                                  {}
        # get_attr       lowered_module_0          lowered_module_0             ()                                  {}
        # call_function  executorch_call_delegate  executorch_call_delegate     (lowered_module_0, arg0_1, arg1_1)  {}
        # call_function  getitem                   <built-in function getitem>  (executorch_call_delegate, 0)       {}
        # call_function  getitem_1                 <built-in function getitem>  (executorch_call_delegate, 1)       {}
        # output         output_1                  output                       ([getitem, getitem_1],)             {}
        #
        # We'll remove all call_function nodes, insert an call_delegate node, inserting getitems nodes to get the result for call_delegate node
        # and return the list of getitems as the output

        lowered_exported_program = copy.deepcopy(self.original_module)

        # The real input nodes are the ones not buffer or parameter
        all_input_nodes = [
            node
            for node in lowered_exported_program.graph.nodes
            if (
                node.op == "placeholder"
                and node.name
                not in lowered_exported_program.graph_signature.inputs_to_buffers
                and node.name
                not in lowered_exported_program.graph_signature.inputs_to_parameters
            )
        ]

        output_node = [
            node for node in lowered_exported_program.graph.nodes if node.op == "output"
        ]
        assert len(output_node) == 1, "There should be only one output node"

        # Step 1. Cleaning up the graph before inserting the call_delegate node
        # Remove the original output node
        lowered_exported_program.graph.erase_node(output_node[0])

        # Remove all the everything else except the input
        for node in reversed(lowered_exported_program.graph.nodes):
            if node.op != "placeholder":
                lowered_exported_program.graph.erase_node(node)

        # Find placeholders that are parameters or buffers, remove them from the main graph
        for node in lowered_exported_program.graph.nodes:
            if node.op == "placeholder" and (
                node.name in lowered_exported_program.graph_signature.inputs_to_buffers
                or node.name
                in lowered_exported_program.graph_signature.inputs_to_parameters
            ):
                lowered_exported_program.graph.erase_node(node)

        # Step 2. Start constructing the graph
        lowered_name = get_lowered_module_name(
            lowered_exported_program.graph_module, self
        )
        # Insert the lowered module to the graph module as an attibute
        lowered_node = lowered_exported_program.graph.get_attr(lowered_name)

        # Insert a call_delegate node to the graph module, with arguments from the arg list
        delegate_node = lowered_exported_program.graph.call_function(
            executorch_call_delegate, (lowered_node, *all_input_nodes)
        )
        # Get the output list. Since the output node is a tuple of list, like ([aten_mul_tensor, aten_add_tensor],)
        # We add some handling logic to get the list `[aten_mul_tensor, aten_add_tensor]` properly
        original_output_nodes = [
            node for node in self.original_module.graph.nodes if node.op == "output"
        ][0].args[0]

        delegate_node.meta["spec"] = tuple(
            [make_spec(node.meta["val"]) for node in original_output_nodes]
        )
        delegate_node.meta["val"] = tuple(
            [node.meta["val"] for node in original_output_nodes]
        )

        # The getitem nodes that are going to be inserted to the lowered graph module
        getitem_nodes = []
        for i in range(len(original_output_nodes)):
            getitem_node = lowered_exported_program.graph.call_function(
                operator.getitem,
                args=(delegate_node, i),
            )
            getitem_node.meta["val"] = delegate_node.meta["val"][i]
            getitem_nodes.append(getitem_node)
        lowered_exported_program.graph.output(getitem_nodes)

        lowered_exported_program.graph_module.recompile()
        lowered_exported_program.graph.lint()

        # Users output will be the get items nodes instead
        output_specs = [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name=getitem_node.name),
                target=None,
            )
            for getitem_node in getitem_nodes
        ]
        # All data are consumed by the delegates so they should be removed from the state dict.
        inputs_to_parameters = (
            lowered_exported_program.graph_signature.inputs_to_parameters
        )
        inputs_to_buffers = lowered_exported_program.graph_signature.inputs_to_buffers
        input_specs = [
            InputSpec(
                kind=InputKind.USER_INPUT,
                arg=TensorArgument(name=node.name),
                target=None,
            )
            for user_input in lowered_exported_program.graph_signature.user_inputs
            if user_input not in inputs_to_parameters
            and user_input not in inputs_to_buffers
        ]

        # Double check the ExportedProgram data(especially everything except graph) is good
        exported_program = ExportedProgram(
            root=lowered_exported_program.graph_module,
            graph=lowered_exported_program.graph,
            graph_signature=_get_updated_graph_signature(
                ExportGraphSignature(
                    input_specs=input_specs, output_specs=output_specs
                ),
                lowered_exported_program.graph_module,
            ),
            # TODO: May need to set lowered_exported_program.call_spec = CallSpec(None, None)
            # somewhere as we should pass it a list of tensors to the lowered module and output a
            # list of tensors. Putting call_spec=lowered_exported_program.call_spec is correct here as the
            # inputs/outputs to the toplevel program will be in the format of the eager module.
            state_dict={},  # None because all data are consumed by delegate
            range_constraints=lowered_exported_program.range_constraints,
            module_call_graph=lowered_exported_program.module_call_graph,
            example_inputs=None,
            verifier=lowered_exported_program.verifier,
        )
        exported_program = _transform(
            exported_program, SpecPropPass(), MemoryPlanningPass("greedy")
        )
        emitted_program = emit_program(
            exported_program, emit_stacktrace=emit_stacktrace
        ).program
        return emitted_program

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


# TODO Don't regenerate new signature manually.
def _get_new_signature(
    original_program: ExportedProgram, gm: torch.fx.GraphModule
) -> Tuple[ExportGraphSignature, Dict[str, Union[torch.Tensor, torch.nn.Parameter]]]:
    old_signature = original_program.graph_signature

    input_specs = []
    output_specs = []
    new_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )
    new_state_dict = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in old_signature.inputs_to_parameters:
                parameter_name = old_signature.inputs_to_parameters[node.name]
                # add param to graph signature
                input_specs.append(
                    InputSpec(
                        kind=InputKind.PARAMETER,
                        arg=TensorArgument(name=node.name),
                        target=parameter_name,
                    )
                )

                # add param to state_dict
                new_state_dict[parameter_name] = original_program.state_dict[
                    parameter_name
                ]
            elif node.name in old_signature.inputs_to_buffers:
                buffer_name = old_signature.inputs_to_buffers[node.name]
                # add buffer to graph signature
                input_specs.append(
                    InputSpec(
                        kind=InputKind.BUFFER,
                        arg=TensorArgument(name=node.name),
                        target=buffer_name,
                    )
                )

                # add param to new_state_dict
                new_state_dict[buffer_name] = original_program.state_dict[buffer_name]
            else:
                # not param or buffer then user input
                input_specs.append(
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=node.name),
                        target=None,
                    )
                )
        if node.op == "output":
            for output in pytree.tree_leaves((node.args, node.kwargs)):
                if not isinstance(output, torch.fx.Node):
                    continue
                output_specs.append(
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=output.name),
                        target=None,
                    )
                )

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
        state_dict=subgraph_state_dict,
        range_constraints=copy.deepcopy(owning_program.range_constraints),
        module_call_graph=[],
        verifier=owning_program.verifier,
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
    submodule_node = None
    for node in gm.graph.nodes:
        if node.op == "call_module":
            if node.target == submodule_name:
                submodule_node = node
            else:
                raise RuntimeError(
                    f"The submodule created with nodes {node_list} did not form \
                    one fully contained subgraph. Check that these nodes form a \
                    fully contained graph. Partitioned graph: {gm.graph}."
                )

    if len(orig_outputs) == 1 and isinstance(orig_outputs[0].meta["val"], FakeTensor):
        # If the original output is a single tensor, it has been
        # pytree.tree_flatten-ed to be a singleton list, so we want to replace
        # all uses with a getitem call to the 0th index of the result
        with gm.graph.inserting_after(submodule_node):
            proxy_out = torch.fx.Proxy(submodule_node)[0].node  # type: ignore[index]
            submodule_node.replace_all_uses_with(proxy_out)
            proxy_out.meta["val"] = submodule_node.meta["val"]
            # Reset the args since it was overwritten in the previous line
            proxy_out.args = (submodule_node, 0)
    else:
        # fuse_as_graphmodule will automatically propagate the metadata of the
        # partition's last node to the getitem nodes that appear after the
        # call_module node. However, in the case of delegation we do not want
        # these getitem nodes to contain irrelevant previous metadata
        # (ex. source_fn, # nn_module_stack)
        for user_node in submodule_node.users:
            user_node.meta.pop("nn_module_stack", None)
            user_node.meta.pop("source_fn_stack", None)

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
