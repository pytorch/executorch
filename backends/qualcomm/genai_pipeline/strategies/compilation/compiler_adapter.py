# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Tuple


@dataclass
class CompilationResult:
    """Result of a compilation operation.

    Attributes:
        artifact_paths: Paths to the compiled .pte artifacts.
        etrecord: Optional ETRecord for debugging.
    """

    artifact_paths: List[Path] = field(default_factory=list)
    etrecord: Optional[Any] = None


@runtime_checkable
class CompilerAdapter(Protocol):
    """Protocol for compilation operations.

    Wraps ``to_edge_transform_and_lower_to_qnn`` behind an injectable interface
    for testability. The parameter list of ``compile_model`` deliberately
    mirrors that function so that no required lowering input has to travel as a
    string-keyed option.

    .. note::
        ``compile_model`` lowers a **single graph**, mirroring the underlying
        ``to_edge_transform_and_lower_to_qnn`` API. Models exported as several
        graphs from the same weights (a hybrid decoder's AR-N prefill and AR-1
        decode, plus an optional token-embedding graph) are looped over by the
        **compilation strategy**; adapters that group them into one multi-method
        ``.pte`` may return a single artifact path covering several methods.

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
        constant_methods: Optional[Dict[str, Any]] = None,
        dep_table: Optional[Dict] = None,
        passes_job: Optional[Any] = None,
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
            soc_model: Target SoC chipset.
            backend_type: QNN backend type (HTP, GPU, LPAI).
            constant_methods: Methods returning constants in eager mode. For a
                decoder this carries the quantization attributes written during
                quantization, so it is only complete after that stage.
            dep_table: Per-graph pass dependency table.
            passes_job: Per-graph pass configuration.
            extra_options: Optional tuning knobs (``skip_node_id_set``,
                ``skip_node_op_set``, ``skip_mutable_buffer``,
                ``convert_linear_to_conv2d``, ``generate_etrecord``,
                ``executorch_backend_config``).

        Returns:
            CompilationResult with artifact paths and optional etrecord. May hold
            several artifact paths, or a single multi-method .pte, depending on
            how the implementation groups graphs.
        """
        ...
