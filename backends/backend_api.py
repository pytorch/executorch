import copy
import logging
from contextlib import contextmanager
from functools import singledispatch
from typing import Dict, Generator, List, Type, Union

from executorch.backends.backend_details import BackendDetails
from executorch.backends.compile_spec_schema import CompileSpec

from executorch.backends.partitioner import Partitioner, TPartitioner
from executorch.backends.utils import is_identical_graph
from executorch.exir import (
    attach_export_graph_metadata,
    CallSpec,
    ExirExportedProgram,
    ExportGraphModule,
    ExportGraphSignature,
    get_exir_meta,
    MultiMethodExirExportedProgram,
    pytree,
)

from executorch.exir.delegate import (
    create_submodule_from_nodes,
    executorch_call_delegate,
    get_lowered_module_name,
    LoweredBackendModule,
    patch_lowered_functions,
)
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.pass_base import ExportPass


@singledispatch
def to_backend(args):
    """
    A generic function the dispatch happens on the type of the first argument. There are currently to overloaded to_backend function:

    def to_backend(
        backend_id: str,
        edge_graph_module: ExportGraphModule,
        compile_specs: List[CompileSpec],
    ) -> LoweredBackendModule:

    def to_backend(
        graph_module: ExportGraphModule,
        partitioner: Type[TPartitioner],
    ) -> ExportGraphModule

    Note: Python is dynamically-typed language and therefore cannot have proper method overloading as that requires the language to
    be able to discriminate between types at compile-time. @to_backend.register will attach the function to to_backend() base on the type of the first
    argument (type annotation is required). However, it can't take multiple types as arguments.
    """
    pass


@to_backend.register
def _(
    backend_id: str,
    edge_graph_module: ExportGraphModule,
    compile_specs: List[CompileSpec],
) -> LoweredBackendModule:
    """
    Add overloaded implementations for to_backend:
    def to_backend(
        backend_id: str,
        edge_graph_module: ExportGraphModule,
        compile_specs: List[CompileSpec],
    ) -> LoweredBackendModule:
    Requires the passed in Module in Edge dialect to be executed in the backend identified
    by backend_id. The forward method of the given edge_graph_module will be
    targeted for execution.

    Args:
        backend_id: The backend identifier.
        edge_graph_module: A module in Edge dialect to target for lowering to the backend.
        compile_specs: A list of backend-specific objects with static
            metadata to configure the "compilation" process (e.g. it could be
            another dictionary itself).

    Returns:
        LoweredBackendModule: A Module that has been lowered to the target backend.
        Internally, the lowered Module contains these special attributes:
        backend_id (str: backend id), __processed_module__ (str: a compiled module)
        compile_spec, original_module (original exported module)

    Raises:
        NotImplementedError: The backend is not implemented (e.g. it was not found).
        This exception is derived from RuntimeError and should be caught accordingly.
        RuntimeError: The module cannot be processed by the backend.
    """
    # All backend implementation are final, so we don't need to consider nested subclasses.
    for cls in BackendDetails.__subclasses__():
        if backend_id == cls.__name__:
            copied_graph_module = copy.deepcopy(edge_graph_module)
            processed_bytes = cls.preprocess(
                copied_graph_module,
                compile_specs,
            )
            lowered_module = LoweredBackendModule(
                edge_graph_module,
                backend_id,
                processed_bytes,
                compile_specs,
            )
            patch_lowered_functions(lowered_module)
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


def _partition_and_lower(
    tagged_graph_module: ExportGraphModule, partitioner_instance: Partitioner
) -> ExportGraphModule:
    for tag, delegation_spec in partitioner_instance.partition_tags.items():
        # Create partition with nodes containing this tag. There should only be
        # one contained submodule per tag
        node_list = []
        for node in tagged_graph_module.graph.nodes:
            if node.meta.get("delegation_tag", "") == tag:
                node_list.append(node)

        if len(node_list) == 0:
            logging.debug(f"Did not find any nodes for tag {tag}")
            continue

        logging.debug(f"For tag {tag}, found nodes {node_list}")
        submodule, call_module_node = create_submodule_from_nodes(
            tagged_graph_module, node_list, tag
        )
        logging.debug(f"Partitioned graph module: {tagged_graph_module}")

        lowered_submodule = to_backend(
            delegation_spec.backend_id,
            submodule,
            delegation_spec.compile_specs,
        )

        # Replace the partitioned submodule with a lowered submodule
        # Add call_method node with function "forward"
        with tagged_graph_module.graph.inserting_before(call_module_node):
            lowered_name = get_lowered_module_name(
                tagged_graph_module, lowered_submodule
            )
            lowered_node = tagged_graph_module.graph.get_attr(lowered_name)
            call_delegate_node = tagged_graph_module.graph.call_function(
                executorch_call_delegate,
                (lowered_node,) + call_module_node.args,
                call_module_node.kwargs,
            )
            call_module_node.replace_all_uses_with(call_delegate_node)
            tagged_graph_module.graph.erase_node(call_module_node)

        tagged_graph_module.recompile()

    # Recursively partition and lower for submodules
    for name, submod, _node in get_control_flow_submodules(tagged_graph_module):
        partitioned_submodule = _partition_and_lower(submod, partitioner_instance)
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
    edge_graph_module: ExportGraphModule,
    partitioner: Type[TPartitioner],
) -> ExportGraphModule:
    """
    Add overloaded implementations for to_backend:
    def to_backend(
        edge_graph_module: ExportGraphModule,
        partitioner: Type[TPartitioner],
    ) -> ExportGraphModule

    Returns a semantically-equivalent program to the one given as input (represented
    as a graph module in Edge dialect), but with portions of the program targeted for
    delegation as determined by the partitioner.

    Args:
        ExportGraphModule: Program in Edge dialect.

        partitioner: An instance of the Partitioner class type, in charge with tagging
        portions of the input program for delegation. A valid partitioner must have
        partition_tags: Dict[str, DelegationSpec], where each key is a tag name and the nodes
        with same tag will be fused a one subgraph and delegated to backend specififed in delegation
        spec.


    Returns:
        ExportGraphModule: The input program, with some portions targeted for delegation.
    """
    copied_graph_module = copy.deepcopy(edge_graph_module)
    # Call the partitioner on the given graph module
    partitioner_instance: Partitioner = partitioner()
    tagged_graph_module = partitioner_instance(copied_graph_module)

    # Check that the partitioner did not modify the original graph
    if _ENABLE_VALIDATION:
        assert is_identical_graph(
            tagged_graph_module,
            edge_graph_module,
        ), f"The partitioner {partitioner} should not modify the graph module"
    else:
        logging.warning("Disabled validating the partitioner.")

    assert (
        hasattr(partitioner_instance, "partition_tags")
        and partitioner_instance.partition_tags is not None
    ), f"Partitioner {partitioner} needs a `partition_tags` field containing a mapping of tags to delegate spec"

    tagged_graph_module = _partition_and_lower(
        tagged_graph_module, partitioner_instance
    )

    return tagged_graph_module


def to_backend_multiple(
    multi_method_program: MultiMethodExirExportedProgram,
    partitioner: Union[Dict[str, Type[TPartitioner]], Type[TPartitioner]],
) -> MultiMethodExirExportedProgram:
    """
    Returns a semantically-equivalent program to the one given as input (represented
    as a graph module in Edge dialect), but with portions of each method in the
    program targeted for delegation as determined by the partitioner.

    Args:
        MultiMethodExirExportedProgram: A multiple method exported program in Edge dialect.

        partitioner: The partitioner can either be a Partitioner subclass, or a
            dictionary mapping method names to Partitioner subclass. If it is a
            Partitioner subclass, all methods in the given multi-method exported
            program will be lowered using the given partitioner. If it is a
            dictionary, only method names specified in the dictionary will be
            lowered with the given partitioner.

            THe Partitioner subclass is in charge with tagging portions of the
            input program for delegation. A valid partitioner must have
            partition_tags: Dict[str, DelegationSpec], where each key is a tag
            name and the nodes with same tag will be fused a one subgraph and
            delegated to backend specififed in delegation spec.

    Returns:
        MultiMethodExirExportedProgram: The input program, with some portions
        targeted for delegation in each method of the program.
    """
    if not (isinstance(partitioner, dict) or issubclass(partitioner, Partitioner)):
        raise TypeError(
            "partitioner should either be a dictionary of method names to"
            + "partitioner subclass, or a partitioner subclass."
        )

    method_name_to_delegated_gm = {}
    for method_name, prog in multi_method_program.methods().items():
        gm = prog.graph_module
        if isinstance(partitioner, dict):
            if method_name in partitioner:
                method_name_to_delegated_gm[method_name] = to_backend(
                    gm, partitioner[method_name]
                )
            else:
                method_name_to_delegated_gm[method_name] = gm
        else:
            method_name_to_delegated_gm[method_name] = to_backend(gm, partitioner)

    def gm_to_program(gm: ExportGraphModule):
        ep = ExirExportedProgram(
            gm,
            gm.graph,
            ExportGraphSignature([], [], [], [], {}, {}, {}, None),
            # pyre-ignore[6]
            CallSpec(None, None),
            {},
            {},
            [],
            True,
        )
        ep.graph_module.meta.update(gm.meta)
        attach_export_graph_metadata(ep.graph_module, get_exir_meta(gm))
        return ep

    method_name_to_delegated_program = pytree.tree_map(
        gm_to_program, method_name_to_delegated_gm
    )
    return MultiMethodExirExportedProgram(method_name_to_delegated_program)
