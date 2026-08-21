# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilationResult,
)

logger = logging.getLogger(__name__)

# Options forwarded to lowering when present in ``extra_options``. Each maps to
# a parameter of ``to_edge_transform_and_lower_to_qnn`` of the same name.
_LOWERING_OPTION_KEYS = (
    "convert_linear_to_conv2d",
    "generate_etrecord",
    "skip_mutable_buffer",
    "skip_node_id_set",
    "skip_node_op_set",
)

# ``extra_options`` key overriding the ``to_executorch`` configuration wholesale.
_KEY_EXECUTORCH_BACKEND_CONFIG = "executorch_backend_config"


def _decode_qnn_options(compile_specs: Any) -> Optional[Any]:
    """Recover the ``QnnExecuTorchOptions`` carried by a list of compile specs.

    ``generate_qnn_executorch_compiler_spec`` serialises the target settings
    into a single flatbuffer-valued ``CompileSpec``, so the target the graph
    will actually be built for is readable back out of the specs themselves.

    Args:
        compile_specs: The value passed as ``compile_specs``; a non-iterable is
            accepted, since callers may inject a stub.

    Returns:
        The decoded options, or ``None`` if ``compile_specs`` does not carry a
        QNN spec -- which is the case for a test double or a non-QNN backend.

    Raises:
        Exception: If a spec *is* keyed as the QNN spec but does not decode. A
            schema change that broke decoding would otherwise silently disable
            the target check, so it is surfaced rather than swallowed.
    """
    from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
        flatbuffer_to_option,
    )
    from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

    try:
        specs = iter(compile_specs)
    except TypeError:
        return None

    for spec in specs:
        if getattr(spec, "key", None) != QCOM_QNN_COMPILE_SPEC:
            continue
        return flatbuffer_to_option(spec.value)

    return None


def _verify_target_matches_specs(
    compile_specs: Any,
    soc_model: Any,
    backend_type: Any,
) -> None:
    """Check that ``soc_model`` / ``backend_type`` agree with the compile specs.

    Lowering takes its target exclusively from ``compile_specs``; these two
    arguments are informational. That makes a disagreement silent and expensive
    -- a graph whose ops were validated for one SoC but compiled for another --
    so it is rejected here instead.

    Both arguments are normalised before comparing, because the pipeline carries
    ``soc_model`` as a string (``"SM8750"``) while the specs carry a
    ``QcomChipset``; comparing the two raw would fail for every correct target.

    Args:
        compile_specs: QNN compiler specifications for this graph.
        soc_model: The SoC the caller believes it is targeting, as a name or a
            ``QcomChipset``.
        backend_type: The backend the caller believes it is targeting, as a name
            or a ``QnnExecuTorchBackendType``.

    Raises:
        ValueError: If either value contradicts the specs, or names no known
            SoC / backend.
    """
    from executorch.backends.qualcomm.genai_pipeline.compilation.compile_spec_builder import (
        resolve_backend_type,
        resolve_soc_model,
    )

    options = _decode_qnn_options(compile_specs)
    if options is None:
        return

    spec_soc_model = options.soc_info.soc_model
    target_soc_model = None if soc_model is None else resolve_soc_model(soc_model)
    if target_soc_model is not None and target_soc_model != spec_soc_model:
        raise ValueError(
            f"soc_model {soc_model!r} contradicts the compile specs, which "
            f"target {spec_soc_model!r}. Lowering follows the specs, so the "
            "mismatch would otherwise pass silently."
        )

    spec_backend_type = options.backend_options.backend_type
    target_backend_type = (
        None if backend_type is None else resolve_backend_type(backend_type)
    )
    if target_backend_type is not None and target_backend_type != spec_backend_type:
        raise ValueError(
            f"backend_type {backend_type!r} contradicts the compile specs, "
            f"which target {spec_backend_type!r}. Lowering follows the specs, "
            "so the mismatch would otherwise pass silently."
        )


def _default_executorch_config() -> Any:
    """Build the ``to_executorch`` configuration for QNN graph I/O.

    Graph inputs and outputs are deliberately left unallocated: with a shared
    buffer the caller supplies the addresses, which are allocated from RPC
    memory rather than by memory planning. ``BuildQuantIo`` then gives the
    quantized I/O tensors their types.

    Returns:
        An ``ExecutorchBackendConfig`` suitable for QNN lowering.
    """
    from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    return ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        ),
        passes=[BuildQuantIo()],
    )


class DefaultCompilerAdapter:
    """Default adapter delegating to ``to_edge_transform_and_lower_to_qnn``.

    Lowers one graph to a single ``.pte``: the model is lowered to QNN, the
    resulting edge program is converted to an ExecuTorch program, and that is
    written to ``artifact_dir``.

    .. note::
        This adapter is a **1:1 wrapper over one graph**, mirroring the
        single-graph form of ``to_edge_transform_and_lower_to_qnn``. Two things
        therefore sit deliberately outside it:

        * **Multi-graph grouping.** That function also accepts graph-name-keyed
          dicts, which is how several graphs (a hybrid decoder's prefill and
          decode) become one multi-method ``.pte`` that shares weights. Fanning
          out over graphs is the compilation *strategy*'s job; passing dicts
          through here would make the adapter's contract depend on which of two
          shapes its arguments take.
        * **Spill-fill sizing.** A sharded model wants one spill-fill
          allocation reused across its contexts, which ``update_spill_fill_size``
          computes from the lowered program. It is a property of the group, so
          it belongs with the call that lowers the group; for a single graph
          there is nothing to share.

        A recipe-based implementation (``ExportRecipe`` + ``ExportSession``) was
        considered and rejected: ``QNNRecipeProvider`` accepts only ``soc_model``
        and the three ``skip_*`` keys and warns-and-ignores the rest, so
        ``dep_table``, ``passes_job``, ``constant_methods`` and
        ``convert_linear_to_conv2d`` cannot be expressed through it, and its
        FP16 recipe hardcodes ``use_fp16=True``, leaving no quantized path.
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
        artifact_key: str = ARTIFACT_TEXT_DECODER,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        """Compile the model via ``to_edge_transform_and_lower_to_qnn``.

        Args:
            model: The model to compile (nn.Module or quantized model).
            example_inputs: Positional example inputs for ``torch.export``,
                sourced from the model itself.
            compile_specs: QNN compiler specifications for backend delegation.
            artifact_dir: Directory to store the compiled .pte artifact. Created
                if it does not exist.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset. Lowering reads the target from
                ``compile_specs``, so this is not applied independently; it is
                logged and checked against the specs.
            backend_type: QNN backend type. As with ``soc_model``, lowering
                reads this from ``compile_specs``; it is logged and checked.
            constant_methods: Methods returning constants in eager mode. For a
                decoder this carries the quantization attributes written into
                ``meta`` during quantization, so it is only complete once that
                stage has run.
            dep_table: Pass dependency table for this graph.
            passes_job: Pass configuration for this graph.
            artifact_key: Name to key the written artifact under (see
                ``artifact_keys``). A 1:1 adapter cannot tell which component it
                was handed -- a vision encoder and a text decoder arrive as the
                same argument -- so the caller names it. Defaults to the text
                decoder, the only artifact a text-only model produces.
            extra_options: Optional tuning knobs forwarded to lowering:
                ``skip_node_id_set``, ``skip_node_op_set``,
                ``skip_mutable_buffer``, ``convert_linear_to_conv2d``,
                ``generate_etrecord``, and ``executorch_backend_config`` to
                override the ``to_executorch`` configuration.

        Returns:
            CompilationResult holding the written artifact under
            ``artifact_key``, and the ETRecord when one was requested.

        Raises:
            ValueError: If ``example_inputs`` is missing, since ``torch.export``
                cannot trace without it, or if ``soc_model`` / ``backend_type``
                contradict ``compile_specs``.
        """
        from executorch.backends.qualcomm.utils.utils import (
            to_edge_transform_and_lower_to_qnn,
        )

        if example_inputs is None:
            raise ValueError(
                "example_inputs is required to compile a model; it defines the "
                "exported graph's positional signature and is produced from "
                "the model by ModelLoaderAdapter.get_example_inputs"
            )

        _verify_target_matches_specs(compile_specs, soc_model, backend_type)

        options = dict(extra_options or {})
        lowering_options = {
            key: options[key] for key in _LOWERING_OPTION_KEYS if key in options
        }
        generate_etrecord = bool(lowering_options.get("generate_etrecord", False))

        logger.info(
            "Compiling '%s' for SoC=%s, backend=%s",
            file_name,
            getattr(soc_model, "name", soc_model),
            backend_type,
        )

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=model,
            inputs=example_inputs,
            compiler_specs=compile_specs,
            constant_methods=constant_methods,
            dep_table=dep_table,
            passes_job=passes_job,
            **lowering_options,
        )

        executorch_config = (
            options.get(_KEY_EXECUTORCH_BACKEND_CONFIG) or _default_executorch_config()
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pte_path = artifact_dir / f"{file_name}.pte"
        with open(pte_path, "wb") as file:
            exec_prog_mgr.write_to_file(file)

        logger.info("Wrote artifact to %s", pte_path)

        etrecord = None
        if generate_etrecord:
            etrecord = exec_prog_mgr.get_etrecord()

        return CompilationResult(
            artifact_paths={artifact_key: pte_path},
            etrecord=etrecord,
        )
