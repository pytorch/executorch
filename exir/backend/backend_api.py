# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import singledispatch
from typing import Dict, Generator, List, Mapping

import torch

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.backend.utils import (
    _maybe_duplicate_constant_nodes,
    is_identical_graph,
)

from executorch.exir.delegate import executorch_call_delegate, get_lowered_module_name

from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.lowered_backend_module import (
    _unsafe_adjust_original_program,
    create_exported_program_from_submodule,
    create_submodule_from_nodes,
    LoweredBackendModule,
)
from executorch.exir.program._fake_program import (
    get_fake_program,
    update_to_real_program,
)
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export.exported_program import ExportedProgram, InputSpec, OutputSpec


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
         edge_program: ExportedProgram,
         partitioner: Partitioner,
     ) -> ExportedProgram:
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
                named_data_store_output=preprocess_result.data_store_output,
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
                else:
                    # check that the users all belong to the same tag
                    for user in node.users:
                        users_tag = user.meta.get("delegation_tag", None)
                        if users_tag != tag:
                            raise RuntimeError(
                                f"constant data node ({node}) is tagged with ({tag}) but has user ({user}) which has tag ({users_tag})"
                            )
            node_list.append(node)
    return node_list


def _insert_lowered_submodule(
    submodule_program: ExportedProgram,
    owning_program: ExportedProgram,
    call_submodule_node: torch.fx.Node,
    submodule_output_node: torch.fx.Node,
    lowered_module: LoweredBackendModule,
    is_submodule: bool,
    toplevel_input_specs_to_delete: Dict[str, InputSpec],
    toplevel_output_specs_to_delete: Dict[str, OutputSpec],
):
    owning_graph_module = call_submodule_node.graph.owning_module
    # call delegate args should only use user_inputs
    call_delegate_args = []
    # names of input_specs to delete
    input_specs_to_delete = toplevel_input_specs_to_delete
    # Delete owned constants from the call_submodule_node args
    for call_sm_input in call_submodule_node.args:
        if (
            isinstance(call_sm_input, torch.fx.Node)
            and call_sm_input.name in input_specs_to_delete.keys()
        ):
            continue
        call_delegate_args.append(call_sm_input)

    def generate_debug_handle(ep: ExportedProgram) -> int:
        """
        Generate a debug handle for the given ExportedProgram.
        """
        debug_handle = 0
        for node in ep.graph_module.graph.nodes:
            debug_handle = max(debug_handle, node.meta.get("debug_handle", 0))
        return debug_handle + 1

    # Replace the partitioned submodule with a lowered submodule
    # Add call_method node with function "forward"
    with owning_graph_module.graph.inserting_before(call_submodule_node):
        lowered_name = get_lowered_module_name(owning_graph_module, lowered_module)
        lowered_node = owning_graph_module.graph.get_attr(lowered_name)
        call_delegate_node = owning_graph_module.graph.call_function(
            executorch_call_delegate,
            (lowered_node,) + tuple(call_delegate_args),
            call_submodule_node.kwargs,
        )
        call_delegate_node.meta["debug_handle"] = generate_debug_handle(owning_program)
        call_delegate_node.meta["val"] = [
            out_arg.meta["val"] for out_arg in submodule_output_node.args[0]
        ]
        call_submodule_node.replace_all_uses_with(call_delegate_node)
        owning_graph_module.graph.erase_node(call_submodule_node)
    if is_submodule:
        assert len(toplevel_input_specs_to_delete) == 0
        assert len(toplevel_output_specs_to_delete) == 0
    elif (
        len(toplevel_input_specs_to_delete) > 0
        or len(toplevel_output_specs_to_delete) > 0
    ):
        _unsafe_adjust_original_program(
            owning_program,
            call_delegate_node,
            toplevel_input_specs_to_delete,
            toplevel_output_specs_to_delete,
        )


def _partition_and_lower_one_graph_module(
    tagged_graph_module: torch.fx.GraphModule,
    partition_result: PartitionResult,
    owning_program: ExportedProgram,
    is_submodule: bool,
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

        replace_ctx = (
            tagged_graph_module._set_replace_hook(
                owning_program.graph_signature.get_replace_hook()
            )
            if not is_submodule
            else nullcontext()
        )
        with replace_ctx:
            submodule, call_module_node = create_submodule_from_nodes(
                tagged_graph_module, node_list, tag
            )

        tagged_graph_module_output_node = tagged_graph_module.graph.output_node()
        submodule_output_node = submodule.graph.output_node()
        # Copy the output node meta from the original output node, because
        # create_submodule_from_nodes doesn't cover the meta field
        submodule_output_node.meta = tagged_graph_module_output_node.meta
        logging.debug(f"Partitioned graph module: {tagged_graph_module}")

        (
            submodule_program,
            toplevel_input_specs_to_delete,
            toplevel_output_specs_to_delete,
        ) = create_exported_program_from_submodule(
            submodule,
            owning_program,
            tag,
            call_module_node,
            is_submodule,
        )

        lowered_submodule = to_backend(
            delegation_spec.backend_id,
            submodule_program,
            delegation_spec.compile_specs,
        )

        _insert_lowered_submodule(
            submodule_program,
            owning_program,
            call_module_node,
            submodule_output_node,
            lowered_submodule,
            is_submodule,
            toplevel_input_specs_to_delete,
            toplevel_output_specs_to_delete,
        )
        owning_program._validate()

    return tagged_graph_module


def _partition_and_lower(
    tagged_graph_module: torch.fx.GraphModule,
    partition_result: PartitionResult,
    owning_program: ExportedProgram,
    is_submodule: bool = False,
) -> torch.fx.GraphModule:
    """
    Partitions the graph module into submodules based on tags, and then lowered the nodes with the same tag as one lowered module, including the submodule from control flow
    """

    partitioned_module = _partition_and_lower_one_graph_module(
        tagged_graph_module, partition_result, owning_program, is_submodule
    )

    # Recursively partition and lower for submodules
    for name, submod, _node in get_control_flow_submodules(partitioned_module):
        partitioned_submodule = _partition_and_lower(
            submod, partition_result, owning_program, is_submodule=True
        )
        tagged_graph_module.add_module(name, partitioned_submodule)

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
    edge_program._validate()

    # Use fake program, with FakeTensors in the state dict, to avoid copying large constant values.
    # Fall back to deepcopy if no fake mode is found. TODO(T182910699): Remove this fallback.
    try:
        fake_edge_program = get_fake_program(edge_program)
    except Exception as e:
        logging.warning(
            f"Error in get_fake_program for graph {edge_program.graph_module}, fallback to deepcopy: {e}"
        )
        fake_edge_program = copy.deepcopy(edge_program)
    partitioner_result = partitioner_instance(fake_edge_program)
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

    update_to_real_program(tagged_exported_program, edge_program)

    for tag, _ in partitioner_result.partition_tags.items():
        _maybe_duplicate_constant_nodes(tagged_exported_program, tag)

    tagged_graph_module = _partition_and_lower(
        tagged_exported_program.graph_module,
        partitioner_result,
        tagged_exported_program,
    )

    # Partitioner added delegation tags to the graph module nodes,
    # we make sure to remove them after we finished partition_and_lower
    for node in tagged_graph_module.graph.nodes:
        node.meta.pop("delegation_tag", None)

    return ExportedProgram(
        root=tagged_graph_module,
        graph=tagged_graph_module.graph,
        graph_signature=tagged_exported_program.graph_signature,
        state_dict=tagged_exported_program.state_dict,
        range_constraints=copy.deepcopy(tagged_exported_program.range_constraints),
        module_call_graph=copy.deepcopy(tagged_exported_program.module_call_graph),
        example_inputs=None,
        constants=tagged_exported_program.constants,
        verifiers=[tagged_exported_program.verifier],
    )


def _create_partitions_in_graph_module(
    tagged_graph_module: torch.fx.GraphModule,
    partition_result: PartitionResult,
    owning_program: ExportedProgram,
    is_submodule: bool,
) -> Dict[str, List[torch.fx.Node]]:
    backend_id_to_submodule_name = {}
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

        replace_ctx = (
            tagged_graph_module._set_replace_hook(
                owning_program.graph_signature.get_replace_hook()
            )
            if not is_submodule
            else nullcontext()
        )
        with replace_ctx:
            submodule, call_module_node = create_submodule_from_nodes(
                tagged_graph_module, node_list, tag
            )

        submodule_output_node = submodule.graph.output_node()
        # Copy the output node meta from the original output node, because
        # create_submodule_from_nodes doesn't cover the meta field
        logging.debug(f"Partitioned graph module: {tagged_graph_module}")
        (
            submodule_program,
            toplevel_input_specs_to_delete,
            toplevel_output_specs_to_delete,
        ) = create_exported_program_from_submodule(
            submodule,
            owning_program,
            tag,
            call_module_node,
            is_submodule,
        )
        call_module_node.meta["backend_id"] = delegation_spec.backend_id
        call_module_node.meta["compile_spec"] = delegation_spec.compile_specs
        call_module_node.meta["submodule_program"] = submodule_program
        call_module_node.meta["toplevel_input_specs_to_delete"] = (
            toplevel_input_specs_to_delete
        )
        call_module_node.meta["toplevel_output_specs_to_delete"] = (
            toplevel_output_specs_to_delete
        )
        call_module_node.meta["is_submodule"] = is_submodule
        call_module_node.meta["submodule_output_node"] = submodule_output_node

        if delegation_spec.backend_id not in backend_id_to_submodule_name:
            backend_id_to_submodule_name[delegation_spec.backend_id] = []

        # The call_module_node created here might not be the same node instance as
        # the one in the final graph module. This is because this node might be replaced
        # in future edits to the graph. As a result, we just keep track of the node's name
        # and at the end we search for this node in our final graph module
        backend_id_to_submodule_name[delegation_spec.backend_id].append(
            call_module_node.target
        )

    created_submodule_nodes = {key: [] for key in backend_id_to_submodule_name.keys()}
    for backend_id, submodule_name in backend_id_to_submodule_name.items():
        for node in tagged_graph_module.graph.nodes:
            if node.op == "call_module" and node.target in submodule_name:
                created_submodule_nodes[backend_id].append(node)

    # check the number of submodule_names and submodule_nodes are equal
    for backend_id in created_submodule_nodes.keys():
        assert len(created_submodule_nodes[backend_id]) == len(
            backend_id_to_submodule_name[backend_id]
        )

    return created_submodule_nodes


def _create_partitions(
    tagged_graph_module: torch.fx.GraphModule,
    partition_result: PartitionResult,
    owning_program: ExportedProgram,
    is_submodule: bool = False,
) -> Dict[str, List[torch.fx.Node]]:
    backend_id_to_call_submodules = _create_partitions_in_graph_module(
        tagged_graph_module, partition_result, owning_program, is_submodule
    )

    # Recursively partition and lower for submodules
    for _, submod, _ in get_control_flow_submodules(tagged_graph_module):
        nested_backend_id_to_call_submodules = _create_partitions(
            submod, partition_result, owning_program, is_submodule=True
        )
        for (
            backend_id,
            nested_submodules,
        ) in nested_backend_id_to_call_submodules.items():
            if backend_id not in backend_id_to_call_submodules:
                backend_id_to_call_submodules[backend_id] = nested_submodules
            else:
                backend_id_to_call_submodules[backend_id].extend(nested_submodules)

    return backend_id_to_call_submodules


def lower_all_submodules_to_backend(
    backend_id: str,
    method_to_submodules_nodes: Dict[str, List[torch.fx.Node]],
    method_to_tagged_edge_program: Dict[str, ExportedProgram],
) -> None:
    """
    Lower all submodules nodes given in the method_to_submodule_nodes map to backend_id.
    """
    # The created exported program for the submodules are in the call_module node's meta data
    # We just map the method_to_submodule_nodes directly to the method_to_partitioned_exported_programs
    method_to_partitioned_program = {
        method_name: [
            # perform deep copy here in case backends change graph inside preprocess method
            copy.deepcopy(node.meta["submodule_program"])
            for node in call_submodule_nodes
        ]
        for method_name, call_submodule_nodes in method_to_submodules_nodes.items()
    }
    method_to_compile_specs = {
        method_name: [node.meta["compile_spec"] for node in call_submodule_nodes]
        for method_name, call_submodule_nodes in method_to_submodules_nodes.items()
    }

    backend_name_to_subclass = {
        subclass.__name__: subclass for subclass in BackendDetails.__subclasses__()
    }
    if backend_id not in backend_name_to_subclass:
        raise NotImplementedError(f"Backend {backend_id} was not found.")

    method_to_preprocess_result: dict[str, List[PreprocessResult]] = (
        backend_name_to_subclass[backend_id].preprocess_multimethod(
            method_to_partitioned_program, method_to_compile_specs
        )
    )

    for method_name in method_to_preprocess_result.keys():
        owning_program = method_to_tagged_edge_program[method_name]
        list_of_preprocess_results = method_to_preprocess_result[method_name]
        list_of_call_submodule_nodes = method_to_submodules_nodes[method_name]
        list_of_compile_specs = method_to_compile_specs[method_name]
        for preprocess_result, call_submodule_node, compile_spec in zip(
            list_of_preprocess_results,
            list_of_call_submodule_nodes,
            list_of_compile_specs,
        ):
            submodule_program = call_submodule_node.meta["submodule_program"]
            lowered_module = LoweredBackendModule(
                edge_program=submodule_program,
                backend_id=backend_id,
                processed_bytes=preprocess_result.processed_bytes,
                compile_specs=compile_spec,
                named_data_store_output=preprocess_result.data_store_output,
            )
            lowered_module.meta = {
                "debug_handle_map": preprocess_result.debug_handle_map,
            }
            is_submodule = call_submodule_node.meta["is_submodule"]
            toplevel_input_specs_to_delete = call_submodule_node.meta[
                "toplevel_input_specs_to_delete"
            ]
            toplevel_output_specs_to_delete = call_submodule_node.meta[
                "toplevel_output_specs_to_delete"
            ]
            submodule_output_node = call_submodule_node.meta["submodule_output_node"]

            _insert_lowered_submodule(
                submodule_program,
                owning_program,
                call_submodule_node,
                submodule_output_node,
                lowered_module,
                is_submodule,
                toplevel_input_specs_to_delete,
                toplevel_output_specs_to_delete,
            )


def remove_used_metadata(graph: torch.fx.Graph) -> None:
    """
    Remove the used metadata from the graph.
    """
    for node in graph.nodes:
        node.meta.pop("delegation_tag", None)
        node.meta.pop("backend_id", None)
        node.meta.pop("submodule_program", None)
        node.meta.pop("toplevel_input_specs_to_delete", None)
        node.meta.pop("toplevel_output_specs_to_delete", None)
        node.meta.pop("is_submodule", None)
        node.meta.pop("submodule_output_node", None)


@dataclass
class MethodProgramsPartitionerSpec:
    """
    Since single dispatch for to_backend requires the first argument to be a
    valid class, we create the following dataclass spec to hold the dictionaries
    mapping the method name to the corresponding program, partitioner
    """

    method_to_edge_program: Mapping[str, ExportedProgram]
    method_to_partitioner: Mapping[str, Partitioner]


@to_backend.register
def _(
    method_edge_program_partitioners: MethodProgramsPartitionerSpec,
) -> Dict[str, ExportedProgram]:
    """
    Add overloaded implementations for to_backend:

    ::

     def to_backend(
        method_edge_program_partitioners: MethodProgramsPartitionerSpec
    ) -> Dict[str, ExportedProgram]:

    Returns a semantically-equivalent dictionary of programs to the programs given as input (represented
    as a graph module in Edge dialect), but with portions of the program targeted for
    delegation as determined by the partitioner.

    Args:
        method_edge_program_partitioners: contains two mappings,
        - method_to_edge_program: mapping of method names to their respective programs in Edge dialect.
        - method_to_partitioner: mapping of method names to an instance of the partitioner, in charge with tagging
        portions of the specified program for delegation. A valid partitioner must return PartitionerResult
        including both tagged exported program and partitioner_tag: Dict[str, DelegationSpec], where each key is a tag name and
        the nodes with same tag will be fused a one subgraph and delegated to backend specififed in delegation spec.


    Returns:
        ExportedProgram: The input program, with some portions targeted for delegation.
    """
    method_to_edge_program = method_edge_program_partitioners.method_to_edge_program
    method_to_partitioner = method_edge_program_partitioners.method_to_partitioner

    partitioned_and_lowered_exported_programs = {}
    backend_id_to_method_submodules_map = {}
    method_to_tagged_exported_program = {}

    for method_name, partitioner_instance in method_to_partitioner.items():
        assert (
            method_name in method_to_edge_program
        ), f"Partitioner for method {method_name} is not provided"
        edge_program = method_to_edge_program[method_name]
        edge_program._validate()

        # Use fake program, with FakeTensors in the state dict, to avoid copying large constant values.
        # Fall back to deepcopy if no fake mode is found. TODO(T182910699): Remove this fallback.
        try:
            fake_edge_program = get_fake_program(edge_program)
        except Exception as e:
            logging.warning(
                f"Error in get_fake_program for graph {edge_program.graph_module}, fallback to deepcopy: {e}"
            )
            fake_edge_program = copy.deepcopy(edge_program)
        partitioner_result = partitioner_instance(fake_edge_program)
        tagged_exported_program = partitioner_result.tagged_exported_program
        tagged_exported_program.example_inputs = edge_program.example_inputs

        method_to_tagged_exported_program[method_name] = tagged_exported_program

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

        update_to_real_program(tagged_exported_program, edge_program)

        for tag, _ in partitioner_result.partition_tags.items():
            _maybe_duplicate_constant_nodes(tagged_exported_program, tag)

        backend_id_to_call_submodule_nodes = _create_partitions(
            tagged_exported_program.graph_module,
            partitioner_result,
            tagged_exported_program,
        )
        for (
            backend_id,
            call_submodule_nodes,
        ) in backend_id_to_call_submodule_nodes.items():
            if backend_id not in backend_id_to_method_submodules_map:
                backend_id_to_method_submodules_map[backend_id] = {}
            backend_id_to_method_submodules_map[backend_id][
                method_name
            ] = call_submodule_nodes

    for (
        backend_id,
        method_to_submodule_nodes,
    ) in backend_id_to_method_submodules_map.items():
        lower_all_submodules_to_backend(
            backend_id,
            method_to_submodule_nodes,
            method_to_tagged_exported_program,
        )

    for method_name in method_to_edge_program.keys():
        if method_name in method_to_tagged_exported_program:
            tagged_exported_program = method_to_tagged_exported_program[method_name]
            tagged_exported_program._validate()
            remove_used_metadata(tagged_exported_program.graph_module.graph)
            partitioned_and_lowered_exported_programs[method_name] = ExportedProgram(
                root=tagged_exported_program.graph_module,
                graph=tagged_exported_program.graph_module.graph,
                graph_signature=tagged_exported_program.graph_signature,
                state_dict=tagged_exported_program.state_dict,
                range_constraints=copy.deepcopy(
                    tagged_exported_program.range_constraints
                ),
                module_call_graph=copy.deepcopy(
                    tagged_exported_program.module_call_graph
                ),
                example_inputs=None,
                constants=tagged_exported_program.constants,
                verifiers=[tagged_exported_program.verifier],
            )
        else:
            # this edge program wasn't partitioned, so we can just return it as is
            partitioned_and_lowered_exported_programs[method_name] = (
                method_to_edge_program[method_name]
            )

    return partitioned_and_lowered_exported_programs
