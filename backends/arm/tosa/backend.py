# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide TOSA backend entry points for the Arm ExecuTorch integration.

Implement the Ahead-of-Time (AoT) preprocessing path that lowers an
``ExportedProgram`` to a TOSA flatbuffer using Arm's lowering pipeline. Use
this module either as a standalone backend that produces a TOSA artifact or as
part of a composed pipeline for hardware backends that consume TOSA as an
intermediate form.

Use ``TOSABackend.preprocess`` to return the serialized TOSA flatbuffer that
subsequent stages (for example, JIT or hardware-specific compilers) consume.

"""

import logging
import tempfile
from collections import deque
from itertools import count
from typing import cast, Dict, final, List, Set

import tosa_serializer as ts
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.common.debug import debug_fail, debug_tosa_dump
from executorch.backends.arm.debug.schema import DebugHook
from executorch.backends.arm.process_node import (
    process_call_function,
    process_output,
    process_placeholder,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.mapping import TOSA_TENSOR_NAME_META
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.graph_module import get_control_flow_submodules
from torch.export.exported_program import ExportedProgram
from torch.fx import Graph, GraphModule, Node

# TOSA backend debug functionality
logger = logging.getLogger(__name__)


def _annotate_external_ids(ep_graph: Graph) -> Dict[str, int]:
    """Assign deterministic output IDs to nodes reachable from graph outputs.

    Args:
        ep_graph (Graph): FX graph produced by export preprocessing.

    Returns:
        dict[str, int]: Mapping from node name to external output index.

    """
    node2external_id = {}

    def bfs_mark(start_nodes: List[Node], idx: int, seen: Set[Node]):
        """Walk producer graph from ``start_nodes`` and record external IDs."""
        q = deque(start_nodes)
        while q:
            n = q.popleft()
            if n in seen:
                continue
            seen.add(n)
            node2external_id[n.name] = idx
            # Walk backwards so we touch every producer
            q.extend(n.all_input_nodes)

    out = ep_graph.output_node()
    # First argument of output node is tuple of outputs
    output_list = cast(tuple, out.args[0])
    seen: Set[Node] = set()
    for idx, val in enumerate(output_list):
        bfs_mark([val], idx, seen)
    return node2external_id


def _sort_outputs(graph_module: GraphModule, node_to_id_map: dict[str, int]):
    """Reorder graph outputs to match ascending external IDs.

    Args:
        graph_module (GraphModule): Graph to reorder in place.
        node_to_id_map (dict[str, int]): Mapping from node name to output index.

    Returns:
        GraphModule: Updated graph module with deterministic output ordering.

    """

    def _external_id(n: Node, node_2_id, fallback: int) -> int:
        """Return the external ID for ``n`` or ``fallback`` when absent."""
        return node_2_id.get(n.name, fallback)

    out_node = graph_module.graph.output_node()
    out_list = cast(tuple, out_node.args[0])
    _counter = count()

    # sort nodes by the key that is id
    def _sort_key(t: Node) -> int:
        """Key function that orders outputs by external ID or position."""
        return _external_id(t, node_to_id_map, next(_counter))

    orig_ord = tuple(sorted(out_list, key=_sort_key))

    current_order = tuple(out_list)
    if orig_ord != current_order:
        replacement = list(orig_ord) if isinstance(out_node.args[0], list) else orig_ord
        out_node.args = (replacement,)
        graph_module.graph.lint()
        graph_module.recompile()

    return graph_module


def arm_get_first_delegation_tag(graph_module) -> str:
    """Return the first delegation tag discovered in the FX graph.

    Args:
        graph_module (GraphModule): Module produced by Arm partitioning.

    Returns:
        str: First non-empty delegation tag or an empty string when no tag is
        recorded.

    """
    for node in graph_module.graph.nodes:
        tag = node.meta.get("delegation_tag")
        if tag:
            return tag

    logger.debug("No delegation tag found in partition.")
    return ""


@final
class TOSABackend(BackendDetails):
    """Provide a backend for lowering programs to TOSA.

    Use this class standalone to produce a TOSA representation, or as part of a
    composed pipeline for hardware backends that consume TOSA.

    """

    @staticmethod
    def preprocess(edge_program: ExportedProgram, compile_specs: List[CompileSpec]):
        """Convert an exported program using the provided compile specs.

        Args:
            edge_program (ExportedProgram): Program generated by Torch export.
            compile_specs (List[CompileSpec]): Raw compile specifications from
                ``executorch.apply_backend``.

        Returns:
            PreprocessResult: Result containing serialized TOSA bytes.

        """
        return TOSABackend._preprocess(
            edge_program, TosaCompileSpec.from_list(compile_specs)
        )

    @staticmethod
    def _preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: TosaCompileSpec,
    ) -> PreprocessResult:
        """Lower an exported program to a TOSA flatbuffer.

        Apply Arm transformation passes to ``edge_program``, then walk the
        transformed FX graph to emit a TOSA graph via the serializer. When
        requested in ``compile_spec``, write additional debug artifacts.

        Args:
            edge_program (ExportedProgram): Program to lower to TOSA.
            compile_spec (TosaCompileSpec): Backend options. Recognized keys:
                - output_format: Must be "tosa".
                - tosa_spec: Target TOSA version/capabilities.
                - debug_artifact_path: Directory for debug outputs.
                - compile_flags: Optional backend flags.
                - dump_debug_info: Enable extra debug JSON dump.

        Returns:
            PreprocessResult: Result containing processed_bytes with the
            serialized TOSA flatbuffer.

        Raises:
            ValueError: If output_format is not "tosa" or the TOSA
                specification is missing from compile_spec.
            RuntimeError: If an unsupported FX node type is encountered.

        """
        # if a debug/test build capture output files from TOSA stage
        artifact_path = compile_spec.get_intermediate_path()
        tosa_spec = compile_spec.tosa_spec
        dump_debug_info = compile_spec.tosa_debug_mode
        debug_hook = None
        if dump_debug_info is not None:
            debug_hook = DebugHook(dump_debug_info)

        logger.info(f"Converting ExportedProgram to TOSA: {tosa_spec}")

        # Converted output for this subgraph, serializer needs path early as it emits
        # const data directly. Path created and data written only in debug builds.
        if not artifact_path:
            artifact_path = ""

        version = tosa_spec.version
        tosa_graph = ts.TosaSerializer(
            artifact_path,
            targetMajor=version.major,
            targetMinor=version.minor,
            targetPatch=version.micro,
            targetDraft=False,
        )

        if not (
            tosa_spec.version.major == ts.TOSA_VERSION_MAJOR
            and tosa_spec.version.minor == ts.TOSA_VERSION_MINOR
        ):
            raise RuntimeError(
                f"TOSA serializer version "
                f"({ts.TOSA_VERSION_MAJOR}.{ts.TOSA_VERSION_MINOR}) "
                f"doesn't match specification {tosa_spec}"
            )

        TOSABackend._preprocess_module(
            edge_program.graph_module,
            edge_program,
            compile_spec,
            tosa_graph,
            debug_hook,
        )
        # Serialize and return the TOSA flatbuffer.
        binary = tosa_graph.serialize()

        if artifact_path:
            tag = arm_get_first_delegation_tag(edge_program.graph_module)

            # Only dump TOSA if we are not saving to temporary folder.
            if len(
                tempdir := tempfile.gettempdir()
            ) > 0 and not artifact_path.startswith(tempdir):
                debug_tosa_dump(
                    binary,
                    artifact_path,
                    suffix="{}".format(f"_{tag}" if tag else "") + (f"_{tosa_spec}"),
                )

            if debug_hook is not None:
                if debug_hook.mode == ArmCompileSpec.DebugMode.JSON:
                    json_output = debug_hook.serialize()
                    with open(f"{artifact_path}/debug.json", "w") as f:
                        f.write(json_output)

        return PreprocessResult(processed_bytes=binary)

    @staticmethod
    def _preprocess_module(  # noqa: C901
        graph_module: GraphModule,
        edge_program: ExportedProgram,
        compile_spec: TosaCompileSpec,
        tosa_graph: ts.TosaSerializer,
        debug_hook: DebugHook | None,
        submodule_name: str | None = None,
    ):
        """Convert an FX ``graph_module`` to TOSA serializer calls.

        Args:
            graph_module (GraphModule): Module to lower recursively.
            edge_program (ExportedProgram): Original exported program.
            compile_spec (TosaCompileSpec): Backend options with TOSA settings.
            tosa_graph (ts.TosaSerializer): Serializer receiving operators.
            debug_hook (DebugHook | None): Optional debug instrumentation.
            submodule_name (str | None): Name used when visiting nested blocks.

        Raises:
            RuntimeError: If an FX node with an unsupported op kind is found.

        """
        tosa_spec = compile_spec.tosa_spec
        node_to_id_map = _annotate_external_ids(graph_module.graph)
        artifact_path = compile_spec.get_intermediate_path()

        # TODO: Fix the need to lazily import this.
        from executorch.backends.arm._passes import ArmPassManager

        graph_module = ArmPassManager(tosa_spec).transform_to_backend_pipeline(  # type: ignore
            exported_program=edge_program, graph_module=graph_module
        )

        # TODO: Fix the need to lazily import this.
        from executorch.backends.arm.operators.node_visitor import get_node_visitors

        node_visitors = get_node_visitors(edge_program, tosa_spec, debug_hook)
        graph_module = _sort_outputs(graph_module, node_to_id_map)

        if submodule_name is not None:
            tosa_graph.startRegion(submodule_name)
            tosa_graph.currRegion.addBasicBlock(submodule_name)
            suffix = f"_{submodule_name}"
            for loop_node in graph_module.graph.nodes:
                loop_node.meta[TOSA_TENSOR_NAME_META] = suffix

        for node in graph_module.graph.nodes:
            node = cast(Node, node)
            try:
                if node.op == "call_function":
                    process_call_function(node, tosa_graph, node_visitors, tosa_spec)
                elif node.op == "placeholder":
                    if len(node.users) == 0:
                        continue
                    process_placeholder(node, tosa_graph, edge_program, tosa_spec)
                elif node.op == "output":
                    process_output(node, tosa_graph, tosa_spec)
                elif node.op == "get_attr":
                    attr = getattr(graph_module, str(node.target), None)
                    if attr is None:
                        raise RuntimeError(
                            "get_attr node is not targeting anything in graph module."
                        )
                    if not isinstance(attr, GraphModule):
                        raise RuntimeError(
                            "get_attr node is not targeting a GraphModule."
                        )

                    # If the above conditions are ok, we don't need to handle this node here.
                    # Only the string value of node.target is important.
                else:
                    # This will only happen if an unpartitioned graph is passed without
                    # any checking of compatibility.
                    raise RuntimeError(f"{node.name} is unsupported op {node.op}")
            except Exception:
                debug_fail(node, graph_module, tosa_graph, artifact_path)
                raise

        # Recursively preprocess controlflow submodules.
        for name, submodule, _ in get_control_flow_submodules(graph_module):
            TOSABackend._preprocess_module(
                submodule,
                edge_program,
                compile_spec,
                tosa_graph,
                debug_hook,
                submodule_name=name,
            )

    @staticmethod
    def filter_tosa_compile_specs(
        compile_spec: ArmCompileSpec,
    ) -> TosaCompileSpec:
        """Extract the TOSA-specific settings from a composite compile spec.

        Args:
            compile_spec (ArmCompileSpec): Compile specification that may
                include both TOSA and hardware-specific options.

        Returns:
            TosaCompileSpec: TOSA-only specification ready for
            ``TOSABackend.preprocess``.

        """
        return (
            TosaCompileSpec(compile_spec.tosa_spec)
            .dump_intermediate_artifacts_to(compile_spec.get_intermediate_path())
            .dump_debug_info(compile_spec.tosa_debug_mode)
        )
