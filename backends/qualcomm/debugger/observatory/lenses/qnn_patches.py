# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QNN backend patches for PipelineGraphCollectorLens.

Installs a monkey-patch on executorch.examples.qualcomm.utils.ptq_calibrate
to capture the float ExportedProgram with from_node metadata populated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
        PipelineGraphCollectorLens,
    )


def install_qnn_patches(cls: type[PipelineGraphCollectorLens]) -> None:
    """Install QNN ptq_calibrate patch on the PipelineGraphCollectorLens."""
    try:
        import executorch.backends.qualcomm.export_utils as qnn_utils_module

        original = qnn_utils_module._ptq_calibrate
        cls._originals["qnn._ptq_calibrate"] = original

        def patched_ptq_calibrate(captured_model, quantizer, dataset):
            cls._set_accuracy_fallback_dataset(
                dataset, source="qnn.ptq_calibrate"
            )

            collect_target = captured_model
            try:
                sample = (
                    cls._last_calibration_dataset[0]
                    if cls._last_calibration_dataset
                    else None
                )
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
                    "[PipelineGraphCollector] collect skipped (Exported Float): %s",
                    exc,
                )
            return original(captured_model, quantizer, dataset)

        qnn_utils_module._ptq_calibrate = patched_ptq_calibrate
        logging.info("[PipelineGraphCollector] Installed QNN patch: _ptq_calibrate")

        def _uninstall():
            try:
                qnn_utils_module._ptq_calibrate = original
            except Exception:
                pass

        cls._backend_uninstallers.append(_uninstall)
    except Exception as exc:
        logging.warning(
            "[PipelineGraphCollector] Failed to patch QNN ptq_calibrate: %s", exc
        )
