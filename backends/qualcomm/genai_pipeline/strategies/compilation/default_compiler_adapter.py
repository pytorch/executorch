# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilationResult,
)

logger = logging.getLogger(__name__)


class DefaultCompilerAdapter:
    """Default adapter delegating to ``to_edge_transform_and_lower_to_qnn``.

    .. note::
        The signature below is the final one -- it mirrors
        ``to_edge_transform_and_lower_to_qnn`` argument-for-argument, so the
        per-graph inputs (``compile_specs``, ``dep_table``, ``passes_job``,
        ``constant_methods``) are explicit parameters rather than
        ``extra_options`` keys.

        The **body** is not implemented in this PR. Lowering is implementation
        work rather than interface work, and the version this package needs is
        the multi-graph one: ``to_edge_transform_and_lower_to_qnn`` accepts
        graph-name-keyed dicts for ``module`` / ``inputs`` / ``compiler_specs``
        / ``dep_table`` / ``passes_job``, and a hybrid decoder groups its graphs
        into a single multi-method ``.pte`` for weight sharing. Writing a
        single-graph body here and then replacing it would mean implementing the
        lowering twice, so it lands with the strategy-level fan-out that calls
        it.

        A recipe-based body (``ExportRecipe.get_recipe(QNNRecipeType.FP16)`` +
        ``ExportSession``) was considered and rejected: ``QNNRecipeProvider``
        accepts only ``soc_model`` and the three ``skip_*`` keys and silently
        ignores the rest, so ``dep_table``, ``passes_job``, ``constant_methods``
        and ``convert_linear_to_conv2d`` are unreachable through it, and it
        hardcodes ``use_fp16=True``.

        Until the body lands, inject a custom ``CompilerAdapter``.
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
        """Compile the model via ``to_edge_transform_and_lower_to_qnn``.

        Args:
            model: The model to compile (nn.Module or quantized model).
            example_inputs: Positional example inputs for ``torch.export``,
                sourced from the model itself.
            compile_specs: QNN compiler specifications for backend delegation.
            artifact_dir: Directory to store compiled .pte artifacts.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset enum value.
            backend_type: QNN backend type (HTP, GPU, LPAI).
            constant_methods: Methods returning constants in eager mode. For a
                decoder this carries the quantization attributes written into
                ``meta`` during quantization, so it is only complete once that
                stage has run.
            dep_table: Per-graph pass dependency table.
            passes_job: Per-graph pass configuration.
            extra_options: Optional tuning knobs forwarded to lowering:
                ``skip_node_id_set``, ``skip_node_op_set``,
                ``skip_mutable_buffer``, ``convert_linear_to_conv2d``,
                ``generate_etrecord``, and ``executorch_backend_config`` to
                override the ``to_executorch`` configuration.

        Returns:
            CompilationResult with artifact paths and optional etrecord.

        Raises:
            NotImplementedError: Always raised in this PR; the body lands with
                the multi-graph lowering that calls it.
        """
        raise NotImplementedError(
            "DefaultCompilerAdapter has no body yet: lowering is implemented "
            "together with the strategy-level multi-graph fan-out that calls "
            "it, so that to_edge_transform_and_lower_to_qnn is wired up once "
            "in its graph-keyed form. Inject a custom CompilerAdapter until "
            "then."
        )
