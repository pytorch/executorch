# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from contextlib import contextmanager
from functools import singledispatch
from typing import Generator, List

import torch

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.backend.utils import is_identical_graph

from executorch.exir.delegate import executorch_call_delegate, get_lowered_module_name

from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.lowered_backend_module import (
    _get_new_signature,
    create_exported_program_from_submodule,
    create_submodule_from_nodes,
    LoweredBackendModule,
)
from executorch.exir.pass_base import ExportPass
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export import ExportedProgram


@singledispatch
def to_backend(args):
    """
    A generic function the dispatch happens on the type of the first argument. There are currently to overloaded to_backend function:

    Note: Python is dynamically-typed language and therefore cannot have proper method overloading as that requires the language to
    be able to discriminate between types at compile-time. @to_backend.register will attach the function to to_backend() base on the type of the first
    argument (type annotation is required). However, it can't take multiple types as arguments.

    ::

     def to_backend(
         backend_id: str,
         edge_graph_module: ExportedProgram,
         compile_specs: List[CompileSpec],
     ) -> LoweredBackendModule:

     def to_backend(
         graph_module: torch.fx.GraphModule,
         partitioner: Type[TPartitioner],
     ) -> torch.fx.GraphModule
    """
    pass


@to_backend.register
def _(
    backend_id: str,
    edge_program: ExportedProgram,
    compile_specs: List[CompileSpec],
) -> LoweredBackendModule:
    """
    Add overloaded implementations for to_backend:

    ::

     def to_backend(
         backend_id: str,
         edge_program: ExportedProgram,
         compile_specs: List[CompileSpec],
     ) -> LoweredBackendModule:


    Requires the passed in exported program in Edge dialect to be executed in
    the backend identified by backend_id. The forward method of the given
    edge_graph_module will be targeted for execution.

    Args:
        backend_id: The backend identifier.
        exported_program: An exported program in Edge dialect to target for
        lowering to the backend.
        compile_specs: A list of backend-specific objects with static
            metadata to configure the "compilation" process (e.g. it could be
            another dictionary itself).

    Returns:
        LoweredBackendModule: A Module that has been lowered to the target backend.
        Internally, the lowered Module contains these special attributes:
        backend_id (str: backend id), __processed_module__ (str: a compiled module)
        compile_spec, original_module (original exported program)

    Raises:
        NotImplementedError: The backend is not implemented (e.g. it was not found).
        This exception is derived from RuntimeError and should be caught accordingly.
        RuntimeError: The module cannot be processed by the backend.
    """
    assert isinstance(edge_program, ExportedProgram)

    # All backend implementation are final, so we don't need to consider nested subclasses.
    for cls in BackendDetails.__subclasses__():
        if backend_id == cls.__name__:
            copied_edge_program = copy.deepcopy(edge_program)
            preprocess_result: PreprocessResult = cls.preprocess(
                copied_edge_program,
                compile_specs,
            )
            lowered_module = LoweredBackendModule(
                edge_program=edge_program,
                backend_id=backend_id,
                processed_bytes=preprocess_result.processed_bytes,
                compile_specs=compile_specs,
            )
            lowered_module.meta = {
                "debug_handle_map": preprocess_result.debug_handle_map
            }
            return lowered_module
    raise NotImplementedError(f"Backend {backend_id} was not found.")


_ENABLE_VALIDATION: bool = True


def disable_validation() -> None:
    """Disables validation"""
    global _ENABLE_VALIDATION
    _ENABLE_VALIDATION = False


@contextmanager
def validation_disabled() -> Generator[None, None, None]:
    """
    Disables checking functions (ex. if the partitioned graph is identical to
    the original graph). This context manager should only be used in certain
    scenarios (such as when it has been profiled that checks are taking too
    long, and are not necessarily needed)
    """
    global _ENABLE_VALIDATION
    existing_setting = _ENABLE_VALIDATION
    disable_validation()
    try:
        yield
    finally:
        _ENABLE_VALIDATION = existing_setting


def _get_node_list_with_same_tag(
    tagged_graph_module: torch.fx.GraphModule,
    tag: str,
    owning_program: ExportedProgram,
) -> List[torch.fx.Node]:
    """
    Return a list of nodes with the same tag.
    """
    node_list = []
    for node in tagged_graph_module.graph.nodes:
        if node.meta.get("delegation_tag", "") == tag:
            if node.op == "output":
                raise RuntimeError(f"output node {node} should not be tagged")
            if node.op == "placeholder":
                if (
                    not is_param(owning_program, node)
                    and not is_buffer(owning_program, node)
                    and not is_lifted_tensor_constant(owning_program, node)
                ):
                    raise RuntimeError(
                        f"placeholder node for non-params, non-buffer, and non-tensor constants should not be tagged: {node} "
                    )
            node_list.append(node)
    return node_list


def _partition_and_lower_one_graph_module(
    tagged_graph_module: torch.fx.GraphModule,
    partition_result: PartitionResult,
    owning_program: ExportedProgram,
) -> torch.fx.GraphModule:
    """
    Partitioned and lowered the graph module based on the partition tag, this is to handle one graph module.
    """
    for tag, delegation_spec in partition_result.partition_tags.items():
        # Create partition with nodes containing this tag. There should only be
        # one contained submodule per tag
        node_list = _get_node_list_with_same_tag(
            tagged_graph_module, tag, owning_program
        )

        if len(node_list) == 0:
            logging.debug(f"Did not find any nodes for tag {tag}")
            continue

        logging.debug(f"For tag {tag}, found nodes {node_list}")
        # Tag the nodes that are params as buffers, so we can order the submodule as (Parms + Buffers) (User Inputs)
        submodule, call_module_node = create_submodule_from_nodes(
            tagged_graph_module, node_list, tag
        )
        tagged_graph_module_output_node = [
            node for node in tagged_graph_module.graph.nodes if node.op == "output"
        ]
        submodule_output_node = [
            node for node in submodule.graph.nodes if node.op == "output"
        ]
        # Copy the output node meta from the original output node, because create_submodule_from_nodes doesn't cover the meta field
        submodule_output_node[0].meta = tagged_graph_module_output_node[0].meta
        logging.debug(f"Partitioned graph module: {tagged_graph_module}")

        submodule_program = create_exported_program_from_submodule(
            submodule, owning_program
        )

        lowered_submodule = to_backend(
            delegation_spec.backend_id,
            submodule_program,
            delegation_spec.compile_specs,
        )

        # call delegate args should only use user_inputs
        call_delegate_args = []
        for inp in call_module_node.all_input_nodes:
            if inp.name in submodule_program.graph_signature.user_inputs:
                call_delegate_args.append(inp)

        # Replace the partitioned submodule with a lowered submodule
        # Add call_method node with function "forward"
        with tagged_graph_module.graph.inserting_before(call_module_node):
            lowered_name = get_lowered_module_name(
                tagged_graph_module, lowered_submodule
            )
            lowered_node = tagged_graph_module.graph.get_attr(lowered_name)
            call_delegate_node = tagged_graph_module.graph.call_function(
                executorch_call_delegate,
                (lowered_node,) + tuple(call_delegate_args),
                call_module_node.kwargs,
            )
            call_delegate_node.meta["debug_handle"] = len(
                tagged_graph_module.graph.nodes
            )
            call_module_node.replace_all_uses_with(call_delegate_node)
            tagged_graph_module.graph.erase_node(call_module_node)

        # Delete all parameters/buffers consumed by the created exported program
        toplevel_signature = owning_program.graph_signature
        for node in tagged_graph_module.graph.nodes:
            # Find placeholders consumed by the delegate
            if node.op != "placeholder" or len(node.users) != 0:
                continue

            if node.name in toplevel_signature.inputs_to_buffers:
                # Delete the consumed buffers
                buffer_name = toplevel_signature.inputs_to_buffers.pop(node.name)
                toplevel_signature.buffers.remove(buffer_name)
                if buffer_name in owning_program.state_dict:
                    owning_program.state_dict.pop(buffer_name)
                else:
                    owning_program.constants.pop(buffer_name)
                tagged_graph_module.graph.erase_node(node)
            elif node.name in toplevel_signature.inputs_to_parameters:
                # Delete the consumed parameters
                param_name = toplevel_signature.inputs_to_parameters.pop(node.name)
                toplevel_signature.parameters.remove(param_name)
                owning_program.state_dict.pop(param_name)
                tagged_graph_module.graph.erase_node(node)

        tagged_graph_module.recompile()
    return tagged_graph_module


def _partition_and_lower(
    tagged_graph_module: torch.fx.GraphModule,
    partition_result: PartitionResult,
    owning_program: ExportedProgram,
) -> torch.fx.GraphModule:
    """
    Partitions the graph module into submodules based on tags, and then lowered the nodes with the same tag as one lowered module, including the submodule from control flow
    """

    partitioned_module = _partition_and_lower_one_graph_module(
        tagged_graph_module, partition_result, owning_program
    )

    # Recursively partition and lower for submodules
    for name, submod, _node in get_control_flow_submodules(partitioned_module):
        partitioned_submodule = _partition_and_lower(
            submod, partition_result, owning_program
        )
        tagged_graph_module.add_module(name, partitioned_submodule)

    # Run the export pass over the graph module so that the call delegate
    # nodes will match Edge dialect
    # TODO(angelayi): ExportPass will rerun the graph, however all we need
    # here is to add metadata to the call delegate nodes to preserve Edge
    # dialect.  There's work going on to generate a random tensor from a
    # fake tensor and possibly it can help to address the issue.
    res = ExportPass()(tagged_graph_module)
    assert res is not None
    tagged_graph_module = res.graph_module

    return tagged_graph_module


@to_backend.register
def _(
    edge_program: ExportedProgram,
    partitioner_instance: Partitioner,
) -> ExportedProgram:
    """
    Add overloaded implementations for to_backend:

    ::

     def to_backend(
         edge_program: ExportedProgram,
         partitioner: Partitioner,
     ) -> ExportedProgram:

    Returns a semantically-equivalent program to the one given as input (represented
    as a graph module in Edge dialect), but with portions of the program targeted for
    delegation as determined by the partitioner.

    Args:
        ExportedProgram: Program in Edge dialect.

        partitioner: An instance of the partitioner, in charge with tagging
        portions of the input program for delegation. A valid partitioner must return PartitionerResult
        including both tagged exported program and partitioner_tag: Dict[str, DelegationSpec], where each key is a tag name and
        the nodes with same tag will be fused a one subgraph and delegated to backend specififed in delegation spec.


    Returns:
        ExportedProgram: The input program, with some portions targeted for delegation.
    """
    copied_edge_program = copy.copy(edge_program)  # MKG: deepcopy
    partitioner_result = partitioner_instance(copied_edge_program)
    tagged_exported_program = partitioner_result.tagged_exported_program

    # Check that the partitioner did not modify the original graph
    if _ENABLE_VALIDATION:
        assert is_identical_graph(
            tagged_exported_program.graph_module,
            edge_program.graph_module,
        ), f"The partitioner {partitioner_instance} should not modify the graph module"
    else:
        logging.warning("Disabled validating the partitioner.")

    assert (
        partitioner_result.partition_tags is not None
    ), f"Partitioner {partitioner_instance} needs a `partition_tags` field containing a mapping of tags to delegate spec"

    tagged_graph_module = _partition_and_lower(
        tagged_exported_program.graph_module, partitioner_result, edge_program
    )

    # TODO(angelayi): Update this signature in a less manual way (maybe through
    # retracing)
    new_signature, new_state_dict, new_constants = _get_new_signature(
        edge_program, tagged_graph_module
    )
    return ExportedProgram(
        root=tagged_graph_module,
        graph=tagged_graph_module.graph,
        graph_signature=new_signature,
        state_dict=new_state_dict,
        range_constraints=copy.deepcopy(edge_program.range_constraints),
        module_call_graph=copy.deepcopy(edge_program.module_call_graph),
        example_inputs=None,
        verifier=edge_program.verifier,
        constants=new_constants,
    )
