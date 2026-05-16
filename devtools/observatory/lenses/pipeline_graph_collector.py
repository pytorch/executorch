# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline Graph Collector Lens — auto-collects graphs at compilation stages.

This lens installs monkey-patches on framework-level functions to transparently
capture graph artifacts at each stage of the export → quantize → lower pipeline.
All patches are installed on session start and removed on session end.

Region structure (cf. RFC §4.5):

    Session "<script-name>"        ← outermost (opened by CLI / user wrapper)
    ├── quantization/              ← top-level coarse stage; aten-graph work
    │   ├── prepare_pt2e/          ← per-call regions
    │   │     └── record  Annotated Model
    │   └── convert_pt2e/
    │         ├── record  Calibrated Model
    │         └── record  Quantized Model
    └── edge/                      ← top-level coarse stage; edge-dialect work
        ├── to_edge_transform_and_lower/
        │     └── record  EdgeProgramManager EP   (+ Pre-EdgeTransform/<method>)
        └── etrecord/              ← lazy nested region under edge; opens on
            │                        first ETRecord.add_* call
            ├── exported_program/
            │     └── record  ETRecord Exported/<method>
            ├── edge_dialect_program/
            │     └── record  ETRecord Edge/<method>
            └── extra_export_modules/
                  └── record  ETRecord Extra/<module>

`quantization` and `edge` are **sibling** top-level regions (not nested).
Annotated/Calibrated/Quantized models are aten graphs (not edge dialect),
so they live under `quantization`. `to_edge_transform_and_lower` and the
ETRecord operations live under `edge`.

Because the runtime region stack holds only one chain at a time, the lens
opens these top-level regions **lazily** through transition helpers:
`_transition_to_quantization` opens quantization (closing edge+etrecord if
they were open from a backward transition). `_transition_to_edge` closes
quantization (if open) and opens edge. `_ensure_etrecord_region` ensures
edge is open and then opens the nested etrecord region. All open stacks
are closed in `on_session_end`.

Lens lifecycle hooks fire **once** per CLI invocation: `on_session_start`
records the framework's `enter_context`/`collect` callables and installs
patches; `on_session_end` closes any open transition stacks and restores
patches.

Collection points (in pipeline order):
  1. torch.export.export        → "Exported Float" (ExportedProgram)
  2. prepare_pt2e               → "Annotated Model" (GraphModule with observers)
  3. convert_pt2e (input)       → "Calibrated Model" (post-calibration, pre-convert)
  4. convert_pt2e (output)      → "Quantized Model" (GraphModule with Q/DQ ops)
  5. to_edge_transform_and_lower → "Pre-EdgeTransform/{method}" and "EdgeProgramManager EP"
  6. ETRecord.add_*             → "ETRecord Exported/…", "ETRecord Edge/…", etc.

Patching strategy:
  - Framework-level patches (torchao, executorch.exir) work for ALL backends.
  - Backend-specific patches are installed after framework-level patches to avoid
    early module-import alias freezing.
  - Contract: a backend-specific patch that emits "Exported Float" must also
    populate `_last_calibration_dataset` so AccuracyLens can auto-configure.
  - ETRecord patches fire when generate_etrecord=True (forced by the
    to_edge_transform_and_lower patch).
  - All originals are saved in _originals and restored on session end.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, List, Optional

from ..interfaces import AnalysisResult, Lens, ObservationContext, RecordDigest


class PipelineGraphCollectorLens(Lens):
    """Unified graph collector — owns all graph-related monkey-patches."""

    _installed: bool = False
    _originals: Dict[str, Any] = {}
    _collect_fn: Optional[Callable[[str, Any], None]] = None
    _enter_context_fn: Optional[Callable[..., Any]] = None
    _quantization_stack: Optional[contextlib.ExitStack] = None
    _edge_stack: Optional[contextlib.ExitStack] = None
    _etrecord_stack: Optional[contextlib.ExitStack] = None
    # Cross-lens contract for AccuracyLens fallback dataset.
    _last_calibration_dataset: Optional[list] = None
    # Backend-specific patch installers registered via register_backend_patches().
    _backend_patch_installers: List[Callable] = []
    # Backend-specific uninstallers registered during patch installation.
    _backend_uninstallers: List[Callable] = []

    @classmethod
    def register_backend_patches(
        cls, installer: Callable[["PipelineGraphCollectorLens"], None]
    ) -> None:
        """Register a backend-specific patch installer.

        The installer receives the lens class and should use cls._originals,
        cls._collect_fn, cls._enter_context_fn, and
        cls._set_accuracy_fallback_dataset() for standard integration.
        It may also append to cls._backend_uninstallers to register cleanup
        logic.
        """
        if installer not in cls._backend_patch_installers:
            cls._backend_patch_installers.append(installer)

    @classmethod
    def get_name(cls) -> str:
        return "pipeline_graph_collector"

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> None:
        if cls._installed:
            return

        from ..observatory import Observatory

        cls._collect_fn = Observatory.collect
        cls._enter_context_fn = Observatory.enter_context

        # Top-level "quantization" and "edge" regions are opened lazily by
        # the transition helpers (`_transition_to_quantization`,
        # `_transition_to_edge`). The "etrecord" nested region inside edge
        # is opened lazily on the first ETRecord.add_* call.

        # Install backend-agnostic patches first.
        cls._install_quantizer_patches()
        cls._install_edge_lower_patch()
        cls._install_etrecord_patches()
        # Install backend-specific patches registered via register_backend_patches().
        for installer in cls._backend_patch_installers:
            try:
                installer(cls)
            except Exception as exc:
                logging.warning(
                    "[PipelineGraphCollector] Backend patch failed: %s", exc
                )
        cls._installed = True

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> None:
        cls._uninstall_all()

    @classmethod
    def clear(cls) -> None:
        cls._uninstall_all()
        cls._last_calibration_dataset = None
        cls._backend_patch_installers.clear()
        cls._backend_uninstallers.clear()

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        return None

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return None

    @staticmethod
    def analyze(records: List[RecordDigest], config: Dict[str, Any]) -> AnalysisResult:
        return AnalysisResult()

    @classmethod
    def _transition_to_quantization(cls) -> None:
        """Ensure the top-level `quantization` region is the active sibling.

        Closes edge+etrecord if they were open from a previous transition
        (rare backward order; supported defensively).
        """

        if cls._enter_context_fn is None:
            return
        if cls._etrecord_stack is not None:
            cls._etrecord_stack.close()
            cls._etrecord_stack = None
        if cls._edge_stack is not None:
            cls._edge_stack.close()
            cls._edge_stack = None
        if cls._quantization_stack is None:
            cls._quantization_stack = contextlib.ExitStack()
            cls._quantization_stack.enter_context(
                cls._enter_context_fn("quantization")
            )

    @classmethod
    def _transition_to_edge(cls) -> None:
        """Ensure the top-level `edge` region is the active sibling.

        Closes the `quantization` region first (if it was open). Idempotent
        when edge is already open.
        """

        if cls._enter_context_fn is None:
            return
        if cls._quantization_stack is not None:
            cls._quantization_stack.close()
            cls._quantization_stack = None
        if cls._edge_stack is None:
            cls._edge_stack = contextlib.ExitStack()
            cls._edge_stack.enter_context(cls._enter_context_fn("edge"))

    @classmethod
    def _ensure_etrecord_region(cls) -> None:
        """Lazy-open the nested `etrecord` region inside `edge`.

        Guarantees that `edge` is the active top-level region first
        (calls `_transition_to_edge`), then opens `etrecord` as its child.
        Idempotent when etrecord is already open.
        """

        cls._transition_to_edge()
        if cls._etrecord_stack is None and cls._enter_context_fn is not None:
            cls._etrecord_stack = contextlib.ExitStack()
            cls._etrecord_stack.enter_context(cls._enter_context_fn("etrecord"))

    @classmethod
    def _set_accuracy_fallback_dataset(cls, dataset: Any, source: str) -> None:
        """Store dataset for AccuracyLens fallback.

        Backend-specific patch contract:
        any patch that emits "Exported Float" should call this helper first.
        """
        try:
            dataset_list = list(dataset) if not isinstance(dataset, list) else dataset
            if not dataset_list:
                return
            cls._last_calibration_dataset = dataset_list
            logging.debug(
                "[PipelineGraphCollector] Stored fallback dataset from %s (%d samples)",
                source,
                len(dataset_list),
            )
        except Exception:
            # Best-effort only; collection flow must not fail on dataset capture.
            pass

    # ------------------------------------------------------------------
    # Patch: prepare_pt2e, convert_pt2e
    # Captures annotated model (post-prepare) and quantized model
    # (post-convert). Also captures the calibrated model (convert input).
    # ------------------------------------------------------------------

    @classmethod
    def _install_quantizer_patches(cls) -> None:
        try:
            import torchao.quantization.pt2e.quantize_pt2e as qt_module

            # prepare_pt2e
            original_prepare = qt_module.prepare_pt2e
            cls._originals["prepare_pt2e"] = original_prepare

            def patched_prepare_pt2e(model, *args, **kwargs):
                cls._transition_to_quantization()
                result = original_prepare(model, *args, **kwargs)
                try:
                    cls._collect_fn("Annotated Model", result)
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] collect skipped (Annotated Model): %s",
                        exc,
                    )
                return result

            qt_module.prepare_pt2e = patched_prepare_pt2e
            logging.info("[PipelineGraphCollector] Installed patch: prepare_pt2e")

            # convert_pt2e — collect both input (calibrated) and output (quantized)
            original_convert = qt_module.convert_pt2e
            cls._originals["convert_pt2e"] = original_convert

            def patched_convert_pt2e(model, *args, **kwargs):
                cls._transition_to_quantization()
                try:
                    cls._collect_fn("Calibrated Model", model)
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] collect skipped (Calibrated Model): %s",
                        exc,
                    )
                result = original_convert(model, *args, **kwargs)
                try:
                    cls._collect_fn("Quantized Model", result)
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] collect skipped (Quantized Model): %s",
                        exc,
                    )
                return result

            qt_module.convert_pt2e = patched_convert_pt2e
            logging.info("[PipelineGraphCollector] Installed patch: convert_pt2e")
        except Exception as exc:
            logging.warning(
                "[PipelineGraphCollector] Failed to patch quantizer APIs: %s", exc
            )

    # ------------------------------------------------------------------
    # Patch: to_edge_transform_and_lower
    # Forces generate_etrecord=True and collects:
    # 1) pre-transform input programs, and
    # 2) post-call EdgeProgramManager.exported_program().
    # ------------------------------------------------------------------

    @classmethod
    def _install_edge_lower_patch(cls) -> None:
        try:
            import executorch.exir.program._program as program_module
            import executorch.exir as exir_module

            def _collect_pre_edge_transform_inputs(args, kwargs):
                programs = kwargs.get("programs")
                if programs is None and len(args) > 0:
                    programs = args[0]
                if programs is None:
                    return

                if isinstance(programs, dict):
                    for method_name, program in programs.items():
                        try:
                            cls._collect_fn(f"Pre-EdgeTransform/{method_name}", program)
                        except Exception as exc:
                            logging.debug(
                                "[PipelineGraphCollector] collect skipped (Pre-EdgeTransform/%s): %s",
                                method_name,
                                exc,
                            )
                else:
                    try:
                        cls._collect_fn("Pre-EdgeTransform/forward", programs)
                    except Exception as exc:
                        logging.debug(
                            "[PipelineGraphCollector] collect skipped (Pre-EdgeTransform/forward): %s",
                            exc,
                        )

            def _make_patched_to_edge_transform_and_lower(original_fn):
                def patched_to_edge_transform_and_lower(*args, **kwargs):
                    cls._transition_to_edge()
                    _collect_pre_edge_transform_inputs(args, kwargs)
                    kwargs["generate_etrecord"] = True
                    result = original_fn(*args, **kwargs)
                    try:
                        cls._collect_fn(
                            "EdgeProgramManager EP", result.exported_program()
                        )
                    except Exception as exc:
                        logging.debug(
                            "[PipelineGraphCollector] collect skipped (EdgeProgramManager EP): %s",
                            exc,
                        )
                    return result

                return patched_to_edge_transform_and_lower

            for i, module in enumerate([program_module, exir_module]):
                original = module.to_edge_transform_and_lower
                cls._originals[f"to_edge_transform_and_lower_{i}"] = original
                module.to_edge_transform_and_lower = _make_patched_to_edge_transform_and_lower(
                    original
                )
            logging.info(
                "[PipelineGraphCollector] Installed patch: to_edge_transform_and_lower"
            )
        except Exception as exc:
            logging.warning(
                "[PipelineGraphCollector] Failed to patch to_edge_transform_and_lower: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Patch: ETRecord methods
    # Auto-collects graph observations when ETRecord APIs are called.
    # Absorbed from the former auto_collect.py module.
    # ------------------------------------------------------------------

    @classmethod
    def _install_etrecord_patches(cls) -> None:
        try:
            from executorch.devtools.etrecord._etrecord import ETRecord
        except Exception as exc:
            logging.warning(
                "[PipelineGraphCollector] Failed to import ETRecord: %s", exc
            )
            return

        collect = cls._collect_fn

        def _safe_collect(name: str, artifact: Any) -> None:
            try:
                collect(name, artifact)
            except Exception as exc:
                logging.debug(
                    "[PipelineGraphCollector] ETRecord auto-collect skipped (%s): %s",
                    name,
                    exc,
                )

        def _wrap_add_exported_program(original):
            def wrapped(self, exported_program):
                cls._ensure_etrecord_region()
                result = original(self, exported_program)
                if exported_program is None:
                    return result
                if isinstance(exported_program, dict):
                    for method_name, program in exported_program.items():
                        _safe_collect(f"ETRecord Exported/{method_name}", program)
                else:
                    _safe_collect("ETRecord Exported/forward", exported_program)
                return result

            return wrapped

        def _wrap_add_edge_dialect_program(original):
            def wrapped(self, edge_dialect_program):
                cls._ensure_etrecord_region()
                result = original(self, edge_dialect_program)
                processed = getattr(self, "edge_dialect_program", None)
                if isinstance(processed, dict):
                    for method_name, program in processed.items():
                        _safe_collect(f"ETRecord Edge/{method_name}", program)
                elif processed is not None:
                    _safe_collect("ETRecord Edge/forward", processed)
                return result

            return wrapped

        def _wrap_add_extra_export_modules(original):
            def wrapped(self, extra_recorded_export_modules):
                cls._ensure_etrecord_region()
                result = original(self, extra_recorded_export_modules)
                graph_map = getattr(self, "graph_map", {}) or {}
                for module_name, program in graph_map.items():
                    _safe_collect(f"ETRecord Extra/{module_name}", program)
                return result

            return wrapped

        patches = {
            "add_exported_program": _wrap_add_exported_program,
            "add_edge_dialect_program": _wrap_add_edge_dialect_program,
            "add_extra_export_modules": _wrap_add_extra_export_modules,
        }

        for method_name, wrap_builder in patches.items():
            original = getattr(ETRecord, method_name, None)
            if original is None:
                continue
            cls._originals[f"ETRecord.{method_name}"] = original
            setattr(ETRecord, method_name, wrap_builder(original))

        logging.info("[PipelineGraphCollector] Installed ETRecord patches")

    # ------------------------------------------------------------------
    # Uninstall all patches
    # ------------------------------------------------------------------

    @classmethod
    def _uninstall_all(cls) -> None:
        if not cls._installed:
            return

        for key, original in cls._originals.items():
            try:
                if key == "prepare_pt2e":
                    import torchao.quantization.pt2e.quantize_pt2e as qt_module

                    qt_module.prepare_pt2e = original
                elif key == "convert_pt2e":
                    import torchao.quantization.pt2e.quantize_pt2e as qt_module

                    qt_module.convert_pt2e = original
                elif key.startswith("to_edge_transform_and_lower"):
                    import executorch.exir.program._program as program_module
                    import executorch.exir as exir_module
                    for i, module in enumerate([program_module, exir_module]):
                        if str(i) == key[-1]:
                            module.to_edge_transform_and_lower = original

                elif key.startswith("ETRecord."):
                    try:
                        from executorch.devtools.etrecord._etrecord import ETRecord

                        method_name = key.split(".", 1)[1]
                        setattr(ETRecord, method_name, original)
                    except Exception:
                        pass
                else:
                    # Backend-specific patches store (module_attr, module) tuples
                    # or are handled by their own uninstall logic via _originals.
                    # Generic fallback: skip keys we don't recognize.
                    pass
            except Exception as exc:
                logging.warning(
                    "[PipelineGraphCollector] Failed to restore %s: %s", key, exc
                )

        cls._originals.clear()
        cls._collect_fn = None
        cls._enter_context_fn = None
        cls._last_calibration_dataset = None
        for uninstaller in cls._backend_uninstallers:
            try:
                uninstaller()
            except Exception as exc:
                logging.warning(
                    "[PipelineGraphCollector] Backend uninstall failed: %s", exc
                )

        # Close any open transition regions in reverse-nesting order:
        # innermost etrecord first, then edge, then quantization. Whichever
        # of edge / quantization is currently active gets closed; the other
        # is already None.
        if cls._etrecord_stack is not None:
            try:
                cls._etrecord_stack.close()
            except Exception as exc:
                logging.warning(
                    "[PipelineGraphCollector] Failed to close etrecord region: %s",
                    exc,
                )
            cls._etrecord_stack = None
        if cls._edge_stack is not None:
            try:
                cls._edge_stack.close()
            except Exception as exc:
                logging.warning(
                    "[PipelineGraphCollector] Failed to close edge region: %s",
                    exc,
                )
            cls._edge_stack = None
        if cls._quantization_stack is not None:
            try:
                cls._quantization_stack.close()
            except Exception as exc:
                logging.warning(
                    "[PipelineGraphCollector] Failed to close quantization region: %s",
                    exc,
                )
            cls._quantization_stack = None

        cls._installed = False
        logging.info("[PipelineGraphCollector] Uninstalled all patches")
