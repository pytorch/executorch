# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


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

    Wraps external compilation APIs (ExportSession, to_edge_transform_and_lower_to_qnn)
    behind an injectable interface for testability.

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
        compile_specs: Any,
        artifact_dir: Path,
        file_name: str,
        soc_model: Any,
        backend_type: Any,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        """Compile the model to on-device .pte artifacts.

        Args:
            model: The model to compile (nn.Module or quantized model).
            compile_specs: QNN compiler specifications for backend delegation.
            artifact_dir: Directory to store compiled artifacts.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset.
            backend_type: QNN backend type (HTP, GPU, LPAI).
            extra_options: Additional compilation options.

        Returns:
            CompilationResult with artifact paths and optional etrecord. May hold
            several artifact paths, or a single multi-method .pte, depending on
            how the implementation groups graphs.
        """
        ...
