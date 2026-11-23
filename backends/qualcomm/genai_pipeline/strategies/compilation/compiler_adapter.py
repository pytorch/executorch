# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Tuple,
    TYPE_CHECKING,
)

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
    from executorch.exir.backend.compile_spec_schema import CompileSpec


@dataclass
class CompilationResult:
    """Result of a compilation operation.

    Attributes:
        artifact_paths: Compiled ``.pte`` artifacts, keyed by artifact name (see
            ``artifact_keys``). Keyed rather than ordered because the consumer
            needs to address artifacts individually -- the on-device runner
            takes a decoder path, a token-embedding path and an encoder path as
            separate arguments -- and because which artifacts exist varies by
            model, so position carries no reliable meaning.
        etrecord: Optional ETRecord for debugging.
    """

    artifact_paths: Dict[str, Path] = field(default_factory=dict)
    etrecord: Optional[Any] = None


@runtime_checkable
class CompilerAdapter(Protocol):
    """Protocol for compilation operations.

    Wraps ``to_edge_transform_and_lower_to_qnn`` behind an injectable interface
    for testability. The parameter list of ``compile_model`` deliberately
    mirrors that function so that no required lowering input has to travel as a
    string-keyed option.

    Two entry points, because they are different units of work:
    ``compile_model`` lowers one graph to one ``.pte``, while
    ``compile_graph_group`` lowers several graphs into one multi-method ``.pte``
    so they share weights. The group form cannot be expressed as repeated
    single-graph calls -- weight sharing is a property of the one lowering call
    -- but *which* graphs form a group stays a decision of the compilation
    strategy.

    .. note::
        Only *deployed* graphs reach this adapter. A hybrid decoder additionally
        builds a full-auto-regressive calibration graph, but that exists purely
        to source quantization encodings and is never lowered.
    """

    def compile_model(
        self,
        model: Any,
        example_inputs: Tuple[Any, ...],
        compile_specs: Any,
        artifact_dir: Path,
        file_name: str,
        soc_model: Any,
        backend_type: Any,
        executorch_config: Any,
        constant_methods: Optional[Dict[str, Any]] = None,
        dep_table: Optional[Dict] = None,
        passes_job: Optional[Any] = None,
        artifact_key: str = ARTIFACT_TEXT_DECODER,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        """Compile the model to on-device .pte artifacts.

        The parameter list mirrors ``to_edge_transform_and_lower_to_qnn``: every
        argument that lowering genuinely needs is explicit, and
        ``extra_options`` is reserved for optional tuning knobs. Passing a
        required input as a string-keyed option is deliberately avoided -- it
        hides the contract and fails at runtime rather than at the call site.

        Args:
            model: The model to compile (nn.Module or quantized model).
            example_inputs: Positional example inputs for ``torch.export``,
                sourced from the model itself.
            compile_specs: QNN compiler specifications for backend delegation.
            artifact_dir: Directory to store compiled artifacts.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset. The target lowering actually uses is
                the one carried in ``compile_specs``; implementations should
                treat a disagreement as an error rather than silently preferring
                one, since ops validated for one SoC and compiled for another
                fail only on device.
            backend_type: QNN backend type (HTP, GPU, LPAI). Carried in
                ``compile_specs`` as well; see ``soc_model``.
            executorch_config: The ``to_executorch`` configuration. Chosen by
                the strategy per component, since memory planning differs.
            constant_methods: Methods returning constants in eager mode. For a
                decoder this carries the quantization attributes written during
                quantization, so it is only complete after that stage.
            dep_table: Per-graph pass dependency table.
            passes_job: Per-graph pass configuration.
            artifact_key: Name to key the resulting artifact under (see
                ``artifact_keys``). Supplied by the caller because a single-graph
                adapter cannot tell a text decoder from a vision encoder -- both
                arrive as ``model`` -- and the inference stage addresses
                artifacts by name.
            extra_options: Optional tuning knobs (``skip_node_id_set``,
                ``skip_node_op_set``, ``skip_mutable_buffer``,
                ``convert_linear_to_conv2d``, ``generate_etrecord``).

        Returns:
            CompilationResult with artifact paths and optional etrecord.
        """
        ...

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
        """Lower several graphs into one multi-method ``.pte``.

        Each graph has its own module, example inputs, dependency table and pass
        configuration, so the lowering call takes graph-name-keyed dicts.
        Grouping them into one call is what lets them share weights, and is also
        what makes a single spill-fill allocation sizeable across their contexts.

        Args:
            graphs: The graphs to lower, keyed by graph name (see
                ``graph_names``). All of them land in one ``.pte``.
            compile_specs: QNN compiler specifications, keyed by the same graph
                names. Keys must match ``graphs`` exactly.
            artifact_dir: Directory to store the compiled .pte artifact.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset; checked against ``compile_specs``.
            backend_type: QNN backend type; checked against ``compile_specs``.
            executorch_config: The ``to_executorch`` configuration. One for the
                whole ``.pte``.
            constant_methods: Methods returning constants in eager mode. One set
                covers the whole ``.pte``, so a caller with several graphs picks
                the authoritative one -- for a decoder, the decode graph's
                ``meta``.
            dep_table: Pass dependency tables, keyed by the same graph names.
                Derived by the strategy, as ``compile_specs`` is: sharding and
                quantized-IO tagging are settings of this lowering call rather
                than properties a graph carries.
            passes_job: Pass configurations, keyed by the same graph names.
            artifact_key: Name to key the resulting artifact under (see
                ``artifact_keys``).
            extra_options: Optional tuning knobs, as for ``compile_model``.

        Returns:
            CompilationResult holding the one written artifact under
            ``artifact_key``, and the ETRecord when one was requested.
        """
        ...
