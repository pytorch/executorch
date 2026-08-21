# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilationResult,
)


class DefaultCompilerAdapter:
    """Default adapter delegating to the ExportSession recipe-based pipeline.

    .. note::
        The body is deliberately unimplemented **in this PR only**: this PR
        establishes the adapter interfaces, and the compilation strategy that
        drives this adapter lands in the following PR, so the implementation
        ships alongside its caller rather than ahead of it.

        The APIs it will delegate to -- ``ExportRecipe.get_recipe`` with
        ``QNNRecipeType.FP16`` and ``ExportSession`` -- are already available
        in-tree; nothing external is blocking it. Until the follow-up lands,
        inject a custom ``CompilerAdapter`` implementation.
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
        """Compile the model using ExportSession.

        Placeholder for the recipe-based compilation flow, which uses
        ``ExportRecipe.get_recipe(QNNRecipeType.FP16, ...)`` combined with
        ``ExportSession``. Implemented in the compilation-strategy PR.

        Args:
            model: The model to compile (nn.Module or quantized model).
            compile_specs: QNN compiler specifications (currently unused when
                using recipe-based flow; kept for future custom compile spec support).
            artifact_dir: Directory to store compiled .pte artifacts.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset enum value.
            backend_type: QNN backend type (HTP, GPU, LPAI).
            extra_options: Additional compilation options. Supported keys:
                - ``example_inputs``: Sample inputs for torch.export.
                - ``generate_etrecord``: Whether to generate ETRecord.
                - ``constant_methods``: Dict of constant methods.

        Returns:
            CompilationResult with artifact paths and optional etrecord.

        Raises:
            NotImplementedError: Always raised in this PR. The body lands with
                the compilation strategy that drives it; inject a custom
                CompilerAdapter until then.
        """
        raise NotImplementedError(
            "DefaultCompilerAdapter is not implemented in this PR: the body "
            "lands together with the compilation strategy that drives it. "
            "Inject a custom CompilerAdapter implementation until then."
        )
