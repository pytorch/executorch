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
  - Backend-specific patches are installed after framework-level patches to avoid
    early module-import alias freezing.
  - Contract: a backend-specific patch that emits "Exported Float" must also
    populate `_last_calibration_dataset` so AccuracyLens can auto-configure.
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
    # Cross-lens contract for AccuracyLens fallback dataset.
    _last_calibration_dataset: Optional[list] = None

    @classmethod
    def get_name(cls) -> str:
        return "pipeline_graph_collector"

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> None:
        if cls._installed:
            return

        from ..observatory import Observatory

        cls._collect_fn = Observatory.collect
        # Install backend-agnostic patches first.
        cls._install_quantizer_patches()
        cls._install_edge_lower_patch()
        cls._install_etrecord_patches()
        # Install backend-specific patches last (QNN currently).
        cls._install_backend_specific_patches()
        cls._installed = True

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> None:
        cls._uninstall_all()

    @classmethod
    def clear(cls) -> None:
        cls._uninstall_all()
        cls._last_calibration_dataset = None

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
    # Backend-specific patches
    # ------------------------------------------------------------------

    @classmethod
    def _install_backend_specific_patches(cls) -> None:
        # QNN backend specific hooks.
        cls._install_qnn_ptq_calibrate_patch()
        # XNNPACK backend specific hooks.
        cls._install_xnnpack_quantize_patch()

    # ------------------------------------------------------------------
    # QNN Patch: ptq_calibrate
    # Captures the float ExportedProgram with from_node metadata populated.
    # Fires after quantization calibration, when from_node is available.
    # ------------------------------------------------------------------

    @classmethod
    def _install_qnn_ptq_calibrate_patch(cls) -> None:
        try:
            import executorch.examples.qualcomm.utils as qnn_utils_module

            original = qnn_utils_module.ptq_calibrate
            cls._originals["qnn.ptq_calibrate"] = original

            def patched_ptq_calibrate(captured_model, quantizer, dataset):
                # Store dataset for AccuracyLens fallback
                cls._set_accuracy_fallback_dataset(
                    dataset, source="qnn.ptq_calibrate"
                )

                # Re-export with dataset[0] to get from_node metadata
                collect_target = captured_model
                try:
                    sample = cls._last_calibration_dataset[0] if cls._last_calibration_dataset else None
                    if sample is not None:
                        import torch
                        ep = torch.export.export(captured_model, sample, strict=False)
                        collect_target = ep.run_decompositions({})
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] from_node re-export skipped: %s", exc
                    )

                try:
                    cls._collect_fn("Exported Float", collect_target)
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] collect skipped (Exported Float): %s", exc
                    )

                return original(captured_model, quantizer, dataset)

            qnn_utils_module.ptq_calibrate = patched_ptq_calibrate
            logging.info("[PipelineGraphCollector] Installed QNN patch: ptq_calibrate")
        except Exception as exc:
            logging.warning(
                "[PipelineGraphCollector] Failed to patch QNN ptq_calibrate: %s", exc
            )

    # ------------------------------------------------------------------
    # XNNPACK Patch: quantize
    # Captures the float ExportedProgram with from_node metadata populated.
    # Also stores example inputs as fallback dataset for AccuracyLens.
    # ------------------------------------------------------------------

    @classmethod
    def _install_xnnpack_quantize_patch(cls) -> None:
        try:
            import executorch.examples.xnnpack.quantization.utils as xnnpack_qutils

            original = xnnpack_qutils.quantize
            cls._originals["xnnpack.quantize"] = original

            def patched_quantize(model, example_inputs, quant_type=None):
                # Store a single-sample fallback dataset for AccuracyLens.
                sample = None
                try:
                    if isinstance(example_inputs, (tuple, list)):
                        sample = tuple(example_inputs)
                    else:
                        sample = (example_inputs,)
                    cls._set_accuracy_fallback_dataset(
                        [sample], source="xnnpack.quantize"
                    )
                except Exception:
                    pass

                # Re-export before quantization to collect an "Exported Float" record
                # with from_node metadata populated.
                collect_target = model
                try:
                    import torch

                    if sample is not None:
                        ep = torch.export.export(model, sample, strict=False)
                        collect_target = ep.run_decompositions({})
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] XNNPACK from_node re-export skipped: %s",
                        exc,
                    )

                try:
                    cls._collect_fn("Exported Float", collect_target)
                except Exception as exc:
                    logging.debug(
                        "[PipelineGraphCollector] collect skipped (Exported Float): %s",
                        exc,
                    )

                if quant_type is None:
                    return original(model, example_inputs)
                return original(model, example_inputs, quant_type)

            xnnpack_qutils.quantize = patched_quantize
            logging.info("[PipelineGraphCollector] Installed XNNPACK patch: quantize")
        except Exception as exc:
            logging.warning(
                "[PipelineGraphCollector] Failed to patch XNNPACK quantize: %s", exc
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
                if key in ("ptq_calibrate", "qnn.ptq_calibrate"):
                    import executorch.examples.qualcomm.utils as qnn_utils_module
                    qnn_utils_module.ptq_calibrate = original
                elif key == "xnnpack.quantize":
                    import executorch.examples.xnnpack.quantization.utils as xnnpack_qutils

                    xnnpack_qutils.quantize = original
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
        cls._last_calibration_dataset = None
        cls._installed = False
        logging.info("[PipelineGraphCollector] Uninstalled all patches")
