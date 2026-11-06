# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import GRAPH_KV_FORWARD
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilationResult,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.default_compiler_adapter import (
    DefaultCompilerAdapter,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
    from executorch.exir.backend.compile_spec_schema import CompileSpec

logger = logging.getLogger(__name__)

# The decoder's fallback custom op. Sharding inserts it to mark the split
# points, so it must stay outside the delegated partitions.
_LLM_SKIP_NODE_OPS = frozenset({"llama.fallback.default"})

_KEY_SKIP_NODE_OP_SET = "skip_node_op_set"


class LLMCompilerAdapter(DefaultCompilerAdapter):
    """Compiler adapter for LLM decoder and token-embedding graph groups.

    Adds the three things a decoder needs that are not true of graphs in
    general, and inherits the rest:

    * the fallback custom op is never delegated (sharding uses it as a marker);
    * ``constant_methods`` default to the **decode** graph's ``meta``, since one
      set covers the ``.pte`` and the runtime uses decode's KV-cache scales for
      both graphs;
    * delegation info is printed per graph when ``verbose`` is set, matching the
      legacy flow.

    Args:
        verbose: Whether to print per-graph delegation info after lowering.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    @property
    def verbose(self) -> bool:
        """Whether per-graph delegation info is printed."""
        return self._verbose

    def compile_graph_group(
        self,
        graphs: Dict[str, "GraphBundle"],
        compile_specs: Dict[str, List["CompileSpec"]],
        artifact_dir: Path,
        file_name: str,
        soc_model: Any,
        backend_type: Any,
        executorch_config: Any,
        constant_methods: Optional[Dict[str, Any]] = None,
        dep_table: Optional[Dict[str, Any]] = None,
        passes_job: Optional[Dict[str, Any]] = None,
        artifact_key: str = ARTIFACT_TEXT_DECODER,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        """Lower an LLM graph group, applying the decoder-specific defaults.

        Args:
            graphs: The graphs to lower, keyed by graph name.
            compile_specs: QNN compiler specifications, keyed by graph name.
            artifact_dir: Directory to store the compiled .pte artifact.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset.
            backend_type: QNN backend type.
            executorch_config: The ``to_executorch`` configuration.
            constant_methods: Methods returning constants in eager mode. Taken
                from the decode graph's ``meta`` when omitted.
            dep_table: Pass dependency tables, keyed by graph name.
            passes_job: Pass configurations, keyed by graph name.
            artifact_key: Name to key the written artifact under.
            extra_options: Optional tuning knobs. Any ``skip_node_op_set`` is
                unioned with the LLM fallback op rather than replaced.

        Returns:
            CompilationResult holding the written artifact and optional ETRecord.
        """
        return super().compile_graph_group(
            graphs=graphs,
            compile_specs=compile_specs,
            artifact_dir=artifact_dir,
            file_name=file_name,
            soc_model=soc_model,
            backend_type=backend_type,
            executorch_config=executorch_config,
            constant_methods=(
                constant_methods
                if constant_methods is not None
                else self._decode_constant_methods(graphs)
            ),
            dep_table=dep_table,
            passes_job=passes_job,
            artifact_key=artifact_key,
            extra_options=self._with_llm_skip_ops(extra_options),
        )

    def _on_lowered(self, edge_prog_mgr: Any, graph_names: List[str]) -> None:
        """Print per-graph delegation info when ``verbose`` is set.

        Args:
            edge_prog_mgr: The lowered edge program manager.
            graph_names: The names of the graphs that were lowered.
        """
        if not self._verbose:
            return

        from executorch.devtools.backend_debug import print_delegation_info

        for graph_name in graph_names:
            print_delegation_info(
                edge_prog_mgr.exported_program(graph_name).graph_module
            )

    @staticmethod
    def _with_llm_skip_ops(
        extra_options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Add the LLM fallback op to any caller-supplied skip set.

        Unioned rather than defaulted: a caller skipping an op of its own must
        not silently re-enable delegation of the sharding marker.

        Args:
            extra_options: The caller's options, possibly ``None``.

        Returns:
            A new options dict whose ``skip_node_op_set`` includes the LLM op.
        """
        options = dict(extra_options or {})
        options[_KEY_SKIP_NODE_OP_SET] = set(
            options.get(_KEY_SKIP_NODE_OP_SET) or ()
        ) | set(_LLM_SKIP_NODE_OPS)
        return options

    @staticmethod
    def _decode_constant_methods(
        graphs: Dict[str, "GraphBundle"],
    ) -> Optional[Dict[str, Any]]:
        """Take the group's constant methods from the decode graph's ``meta``.

        Looked up by name rather than by position so the caller's dict ordering
        does not decide which graph is authoritative.

        Args:
            graphs: The graphs being lowered, keyed by graph name.

        Returns:
            The decode graph's ``meta``, or ``None`` if the group has no decode
            graph -- as for the token embedding, which carries no constants.
        """
        bundle = graphs.get(GRAPH_KV_FORWARD)
        if bundle is None:
            logger.debug(
                "No '%s' graph in this group; lowering without constant methods",
                GRAPH_KV_FORWARD,
            )
            return None

        return dict(bundle.meta)
