# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from contextlib import contextmanager, nullcontext
from functools import singledispatch
from typing import Generator, List

import torch
import torch.utils._pytree as pytree

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

        tagged_graph_module_output_node = [
            node for node in tagged_graph_module.graph.nodes if node.op == "output"
        ][0]
        submodule_output_node = [
            node for node in submodule.graph.nodes if node.op == "output"
        ][0]
        # Copy the output node meta from the original output node, because
        # create_submodule_from_nodes doesn't cover the meta field
        submodule_output_node.meta = tagged_graph_module_output_node.meta
        submodule_output_node.meta["val"] = pytree.tree_map(
            lambda arg: arg.meta.get("val") if isinstance(arg, torch.fx.Node) else arg,
            submodule_output_node.args,
        )
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

        # call delegate args should only use user_inputs
        call_delegate_args = []
        # Preserve input order as user_inputs
        for inp_name in submodule_program.graph_signature.user_inputs:
            for inp_node in call_module_node.all_input_nodes:
                if inp_node.name == inp_name:
                    call_delegate_args.append(inp_node)
                    break

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
            call_delegate_node.meta["val"] = submodule_output_node.meta["val"]
            call_module_node.replace_all_uses_with(call_delegate_node)
            tagged_graph_module.graph.erase_node(call_module_node)

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
