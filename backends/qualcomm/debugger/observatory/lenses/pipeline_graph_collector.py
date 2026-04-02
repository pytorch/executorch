# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline Graph Collector Lens — auto-collects graphs at compilation stages.

This lens installs monkey-patches on framework-level functions to transparently
capture graph artifacts at each stage of the export → quantize → lower pipeline.
All patches are installed on session start and removed on session end.

Collection points (in pipeline order):
  1. torch.export.export        → "Exported Float" (ExportedProgram)
  2. prepare_pt2e               → "Annotated Model" (GraphModule with observers)
  3. convert_pt2e (input)       → "Calibrated Model" (post-calibration, pre-convert)
  4. convert_pt2e (output)      → "Quantized Model" (GraphModule with Q/DQ ops)
  5. to_edge_transform_and_lower → "Edge" / "Transformed Edge"
  6. ETRecord.add_*             → "ETRecord Exported/…", "ETRecord Edge/…", etc.

Patching strategy:
  - Framework-level patches (torchao, executorch.exir) work for ALL backends.
  - ETRecord patches fire when generate_etrecord=True (forced by the
    to_edge_transform_and_lower patch).
  - All originals are saved in _originals and restored on session end.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ..interfaces import AnalysisResult, Lens, ObservationContext, RecordDigest


class PipelineGraphCollectorLens(Lens):
    """Unified graph collector — owns all graph-related monkey-patches."""

    _installed: bool = False
    _originals: Dict[str, Any] = {}
    _collect_fn: Optional[Callable[[str, Any], None]] = None
    _last_export_inputs: Optional[tuple] = None

    @classmethod
    def get_name(cls) -> str:
        return "pipeline_graph_collector"

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> None:
        if cls._installed:
            return

        from ..observatory import Observatory

        cls._collect_fn = Observatory.collect
        cls._install_export_patch()
        cls._install_quantizer_patches()
        cls._install_edge_lower_patch()
        cls._install_etrecord_patches()
        cls._installed = True

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> None:
        cls._uninstall_all()

    @classmethod
    def clear(cls) -> None:
        cls._uninstall_all()
        cls._last_export_inputs = None

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        return None

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return None

    @staticmethod
    def analyze(records: List[RecordDigest], config: Dict[str, Any]) -> AnalysisResult:
        return AnalysisResult()

    # ------------------------------------------------------------------
    # Patch: torch.export.export
    # Captures the ExportedProgram produced from the float model.
    # ------------------------------------------------------------------

    @classmethod
    def _install_export_patch(cls) -> None:
        try:
            import torch.export as export_module

            original = export_module.export
            cls._originals["torch.export.export"] = original

            def patched_export(*args, **kwargs):
                result = original(*args, **kwargs)
                # from_node is absent on the raw export output; it is populated by
                # run_decompositions({}), which re-traces the graph through an Interpreter.
                # We work on a copy so the original result returned to the caller is not mutated.
                collect_target = result
                try:
                    from executorch.exir.passes import DebugHandleGeneratorPass
                    collect_target = result.run_decompositions({})
                    pass_result = DebugHandleGeneratorPass()(collect_target.graph_module)
                    breakpoint()
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] DebugHandleGeneratorPass skipped: %s",
                        exc,
                    )
                try:
                    # torch.export.export(mod, args, ...) — args[1] is the input tuple
                    if len(args) >= 2:
                        cls._last_export_inputs = args[1]

                    cls._collect_fn("Exported Float", collect_target)
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] collect skipped (Exported Float): %s",
                        exc,
                    )
                return result

            export_module.export = patched_export
            logging.info("[PipelineGraphCollector] Installed patch: torch.export.export")
        except Exception as exc:
            logging.warning(
                "[PipelineGraphCollector] Failed to patch torch.export.export: %s", exc
            )

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
    # Forces generate_etrecord=True and collects edge programs.
    # ------------------------------------------------------------------

    @classmethod
    def _install_edge_lower_patch(cls) -> None:
        try:
            import executorch.exir.program._program as program_module
            import executorch.exir as exir_module 

            for i, module in enumerate([program_module, exir_module]):
                original = module.to_edge_transform_and_lower
                cls._originals[f"to_edge_transform_and_lower_{i}"] = original

                def patched_to_edge_transform_and_lower(*args, **kwargs):
                    kwargs["generate_etrecord"] = True
                    result = original(*args, **kwargs)
                    try:
                        cls._collect_fn("Edge", result.exported_program())
                    except Exception as exc:
                        logging.debug(
                            "[PipelineGraphCollector] collect skipped (Edge): %s", exc
                        )
                    return result

                module.to_edge_transform_and_lower = (
                    patched_to_edge_transform_and_lower
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
                if key == "torch.export.export":
                    import torch.export as export_module

                    export_module.export = original
                elif key == "prepare_pt2e":
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
            except Exception as exc:
                logging.warning(
                    "[PipelineGraphCollector] Failed to restore %s: %s", key, exc
                )

        cls._originals.clear()
        cls._collect_fn = None
        cls._last_export_inputs = None
        cls._installed = False
        logging.info("[PipelineGraphCollector] Uninstalled all patches")
