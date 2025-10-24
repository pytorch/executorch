# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#
import logging
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
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram
from torch.fx import Graph, Node

# TOSA backend debug functionality
logger = logging.getLogger(__name__)


def _annotate_external_ids(ep_graph: Graph) -> Dict[str, int]:
    """
    Returns dictionary: node name -> external ids

    Assign id to an output node of the model so we can trace it.
    """
    node2external_id = {}

    def bfs_mark(start_nodes: List[Node], idx: int, seen: Set[Node]):
        q = deque(start_nodes)
        while q:
            n = q.popleft()
            if n in seen:
                continue
            seen.add(n)
            node2external_id[n.name] = idx
            # Walk backwards so we touch every producer
            q.extend(n.all_input_nodes)

    out = next(n for n in ep_graph.nodes if n.op == "output")
    seen: Set[Node] = set()
    for idx, val in enumerate(out.args[0]):
        bfs_mark([val], idx, seen)
    return node2external_id


def arm_get_first_delegation_tag(graph_module) -> str:
    """Get the first delegation tag from the graph_module or return empty string."""
    for node in graph_module.graph.nodes:
        tag = node.meta.get("delegation_tag")
        if tag:
            return tag

    logger.debug("No delegation tag found in partition.")
    return ""


@final
class TOSABackend(BackendDetails):
    """
    BackendDetails subclass for lowering to TOSA.
    Is used either by itself to get to a TOSA representation, or with composition
    to be used as a separate step to target TOSA compliant hardware.
    """

    @staticmethod
    def preprocess(edge_program: ExportedProgram, compile_specs: List[CompileSpec]):
        return TOSABackend._preprocess(
            edge_program, TosaCompileSpec.from_list(compile_specs)
        )

    @staticmethod
    def _preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: TosaCompileSpec,
    ) -> PreprocessResult:
        # if a debug/test build capture output files from TOSA stage
        artifact_path = compile_spec.get_intermediate_path()
        tosa_spec = compile_spec.tosa_spec
        dump_debug_info = compile_spec.tosa_debug_mode

        # Assign to every node external id
        node_2_id = _annotate_external_ids(edge_program.graph)

        logger.info(f"Converting ExportedProgram to TOSA: {tosa_spec}")

        # Converted output for this subgraph, serializer needs path early as it emits
        # const data directly. Path created and data written only in debug builds.
        if not artifact_path:
            artifact_path = ""

        tosa_graph = ts.TosaSerializer(artifact_path)

        if not (
            tosa_spec.version.major == ts.TOSA_VERSION_MAJOR
            and tosa_spec.version.minor == ts.TOSA_VERSION_MINOR
        ):
            raise RuntimeError(
                f"TOSA serializer version "
                f"({ts.TOSA_VERSION_MAJOR}.{ts.TOSA_VERSION_MINOR}) "
                f"doesn't match specification {tosa_spec}"
            )

        # TODO: Fix the need to lazily import this.
        from executorch.backends.arm._passes import ArmPassManager

        graph_module = ArmPassManager(tosa_spec).transform_to_backend_pipeline(  # type: ignore
            exported_program=edge_program
        )

        debug_hook = None
        if dump_debug_info is not None:
            debug_hook = DebugHook(dump_debug_info)

        # TODO: Fix the need to lazily import this.
        from executorch.backends.arm.operators.node_visitor import get_node_visitors

        node_visitors = get_node_visitors(edge_program, tosa_spec, debug_hook)

        # Re-shuffle output nodes to preserve author's order
        def _external_id(n: Node, node_2_id, fallback: int) -> int:
            return node_2_id.get(n.name, fallback)

        out_node = next(n for n in graph_module.graph.nodes if n.op == "output")
        _counter = count()

        # sort nodes by the key that is id
        def _sort_key(t: Node) -> int:
            return _external_id(t, node_2_id, next(_counter))

        orig_ord = tuple(sorted(out_node.args[0], key=_sort_key))

        current_order = tuple(out_node.args[0])
        if orig_ord != current_order:
            replacement = (
                list(orig_ord) if isinstance(out_node.args[0], list) else orig_ord
            )
            out_node.args = (replacement,)
            graph_module.graph.lint()
            graph_module.recompile()

        input_count = 0
        for node in graph_module.graph.nodes:
            node = cast(Node, node)
            try:
                if node.op == "call_function":
                    process_call_function(node, tosa_graph, node_visitors, tosa_spec)
                elif node.op == "placeholder":
                    if len(node.users) == 0:
                        continue
                    process_placeholder(node, tosa_graph, edge_program, tosa_spec)
                    if node.name in edge_program.graph_signature.user_inputs:
                        input_count += 1
                elif node.op == "output":
                    process_output(node, tosa_graph)
                else:
                    # This will only happen if an unpartitioned graph is passed without
                    # any checking of compatibility.
                    raise RuntimeError(f"{node.name} is unsupported op {node.op}")
            except Exception:
                debug_fail(node, graph_module, tosa_graph.serialize(), artifact_path)
                raise

        # Serialize and return the TOSA flatbuffer.
        binary = tosa_graph.serialize()

        if artifact_path:
            tag = arm_get_first_delegation_tag(graph_module)
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
    def filter_tosa_compile_specs(
        compile_spec: ArmCompileSpec,
    ) -> TosaCompileSpec:
        """
        Filter out the CompileSpec elements relevant for the TOSA backend.
        This is needed to compose a backend targetting hardware IP with the
        TOSABackend, since we first want to use the TOSABackend to generate
        the TOSA flatbuffer representation as an intermediate step. The TOSA
        flatbuffer can then be consumed by the backend targetting specific
        hardware.
        """

        return (
            TosaCompileSpec(compile_spec.tosa_spec)
            .dump_intermediate_artifacts_to(compile_spec.get_intermediate_path())
            .dump_debug_info(compile_spec.tosa_debug_mode)
        )
