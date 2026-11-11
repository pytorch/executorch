# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilationResult,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
    from executorch.exir.backend.compile_spec_schema import CompileSpec

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


def _needs_spill_fill(compile_specs: Any) -> bool:
    """Whether a graph's specs ask for the multi-context spill-fill buffer.

    ``update_spill_fill_size`` initialises a QNN backend and context cache per
    partition, so it is only worth calling for graphs that actually requested
    multiple contexts. It is also HTP-only, and is set only when a graph is
    sharded, which makes this equivalent to the legacy ``num_sharding > 1`` gate
    while reading the answer off the specs rather than needing the count.

    Args:
        compile_specs: The compile specs for one graph.

    Returns:
        ``True`` if the specs target HTP with ``use_multi_contexts`` set.
    """
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )

    options = _decode_qnn_options(compile_specs)
    if options is None:
        return False

    if options.backend_options.backend_type != QnnExecuTorchBackendType.kHtpBackend:
        return False

    return bool(options.backend_options.htp_options.use_multi_contexts)


def _write_program(
    edge_prog_mgr: Any,
    executorch_config: Any,
    artifact_dir: Path,
    file_name: str,
    artifact_key: str,
    generate_etrecord: bool,
) -> CompilationResult:
    """Convert a lowered edge program to a ``.pte`` on disk.

    Shared by the single-graph and group paths, which differ only in how they
    reach the edge program.

    Args:
        edge_prog_mgr: The lowered edge program manager.
        executorch_config: The ``to_executorch`` configuration.
        artifact_dir: Directory to write into. Created if it does not exist.
        file_name: Base name for the output .pte file.
        artifact_key: Name to key the written artifact under.
        generate_etrecord: Whether to return the ETRecord as well.

    Returns:
        CompilationResult holding the written artifact and optional ETRecord.
    """
    exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pte_path = artifact_dir / f"{file_name}.pte"
    with open(pte_path, "wb") as file:
        exec_prog_mgr.write_to_file(file)

    logger.info("Wrote artifact to %s", pte_path)

    return CompilationResult(
        artifact_paths={artifact_key: pte_path},
        etrecord=exec_prog_mgr.get_etrecord() if generate_etrecord else None,
    )


class DefaultCompilerAdapter:
    """Default adapter delegating to ``to_edge_transform_and_lower_to_qnn``.

    ``compile_model`` lowers one graph to one ``.pte``; ``compile_graph_group``
    lowers several graphs into one multi-method ``.pte`` so they share weights,
    and sizes the spill-fill buffer shared across their contexts. In both cases
    the edge program is converted to an ExecuTorch program and written to
    ``artifact_dir``.

    .. note::
        Deciding *which* graphs form a group is the compilation **strategy**'s
        job -- this adapter lowers the group it is handed.

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
        executorch_config: Any,
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
            executorch_config: The ``to_executorch`` configuration, chosen by
                the strategy per component.
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
                ``skip_mutable_buffer``, ``convert_linear_to_conv2d`` and
                ``generate_etrecord``.

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

        return _write_program(
            edge_prog_mgr=edge_prog_mgr,
            executorch_config=executorch_config,
            artifact_dir=artifact_dir,
            file_name=file_name,
            artifact_key=artifact_key,
            generate_etrecord=generate_etrecord,
        )

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

        One ``to_edge_transform_and_lower_to_qnn`` call with graph-name-keyed
        dicts, which is what lets the graphs share weights. Spill-fill is sized
        afterwards, per graph whose specs asked for multiple contexts.

        Args:
            graphs: The graphs to lower, keyed by graph name.
            compile_specs: QNN compiler specifications, keyed by graph name.
            artifact_dir: Directory to store the compiled .pte artifact.
            file_name: Base name for the output .pte file.
            soc_model: Target SoC chipset; checked against each graph's specs.
            backend_type: QNN backend type; checked against each graph's specs.
            executorch_config: The ``to_executorch`` configuration. One for the
                whole ``.pte``.
            constant_methods: Methods returning constants in eager mode. One set
                for the whole ``.pte``.
            dep_table: Pass dependency tables, keyed by graph name. Derived by
                the strategy from the graphs' metadata, as ``compile_specs`` is;
                ``None`` leaves lowering to apply its own defaults.
            passes_job: Pass configurations, keyed by graph name. Per graph
                rather than per group, since the quantized-IO tagging depends on
                a graph's AR length.
            artifact_key: Name to key the written artifact under.
            extra_options: Optional tuning knobs, as for ``compile_model``.

        Returns:
            CompilationResult holding the one written artifact under
            ``artifact_key``, and the ETRecord when one was requested.

        Raises:
            ValueError: If ``graphs`` is empty, if its keys do not match
                ``compile_specs`` or a supplied ``dep_table`` / ``passes_job``,
                if a bundle has no ``inputs``, or if ``soc_model`` /
                ``backend_type`` contradict any graph's specs.
        """
        from executorch.backends.qualcomm.utils.utils import (
            to_edge_transform_and_lower_to_qnn,
            update_spill_fill_size,
        )

        self._validate_group(graphs, compile_specs, dep_table, passes_job)

        for specs in compile_specs.values():
            _verify_target_matches_specs(specs, soc_model, backend_type)

        options = dict(extra_options or {})
        lowering_options = {
            key: options[key] for key in _LOWERING_OPTION_KEYS if key in options
        }
        generate_etrecord = bool(lowering_options.get("generate_etrecord", False))

        graph_names = list(graphs)
        logger.info(
            "Compiling '%s' as one .pte over %d graph(s) (%s) for SoC=%s, "
            "backend=%s",
            file_name,
            len(graph_names),
            ", ".join(graph_names),
            getattr(soc_model, "name", soc_model),
            backend_type,
        )

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module={name: graphs[name].module for name in graph_names},
            inputs={name: graphs[name].inputs for name in graph_names},
            compiler_specs={name: compile_specs[name] for name in graph_names},
            constant_methods=constant_methods,
            dep_table=dep_table,
            passes_job=passes_job,
            **lowering_options,
        )

        for graph_name in graph_names:
            if _needs_spill_fill(compile_specs[graph_name]):
                logger.debug("Sizing spill-fill buffer for graph '%s'", graph_name)
                update_spill_fill_size(edge_prog_mgr.exported_program(graph_name))

        self._on_lowered(edge_prog_mgr, graph_names)

        return _write_program(
            edge_prog_mgr=edge_prog_mgr,
            executorch_config=executorch_config,
            artifact_dir=artifact_dir,
            file_name=file_name,
            artifact_key=artifact_key,
            generate_etrecord=generate_etrecord,
        )

    def _on_lowered(self, edge_prog_mgr: Any, graph_names: List[str]) -> None:
        """Hook run after a group is lowered and before ``to_executorch``.

        A no-op here; subclasses use it to inspect the lowered programs, which is
        only possible at this point in the call.

        Args:
            edge_prog_mgr: The lowered edge program manager.
            graph_names: The names of the graphs that were lowered.
        """

    @staticmethod
    def _validate_group(
        graphs: Dict[str, "GraphBundle"],
        compile_specs: Dict[str, List["CompileSpec"]],
        dep_table: Optional[Dict[str, Any]] = None,
        passes_job: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Check a graph group is lowerable before any graph work starts.

        Args:
            graphs: The graphs to lower, keyed by graph name.
            compile_specs: QNN compiler specifications, keyed by graph name.
            dep_table: Pass dependency tables, keyed by graph name, if supplied.
            passes_job: Pass configurations, keyed by graph name, if supplied.

        Raises:
            ValueError: If the group is empty, a per-graph mapping's key set
                differs from the graph names, or a bundle carries no example
                inputs.
        """
        if not graphs:
            raise ValueError("graphs is empty; nothing to compile")

        if set(graphs) != set(compile_specs):
            raise ValueError(
                f"compile_specs keys {sorted(compile_specs)} do not match graph "
                f"names {sorted(graphs)}; every graph in the group needs its own "
                "specs"
            )

        for name, keyed in (("dep_table", dep_table), ("passes_job", passes_job)):
            if keyed is not None and set(keyed) != set(graphs):
                raise ValueError(
                    f"{name} keys {sorted(keyed)} do not match graph names "
                    f"{sorted(graphs)}; a graph lowered with another graph's "
                    "pass configuration would be silently mis-transformed"
                )

        missing = sorted(name for name, b in graphs.items() if b.inputs is None)
        if missing:
            raise ValueError(
                f"graph(s) {missing} have no inputs; torch.export cannot trace "
                "without the positional signature produced by "
                "ModelLoaderAdapter.get_example_inputs"
            )
