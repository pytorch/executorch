# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XNNPACK backend patches for PipelineGraphCollectorLens.

Installs a monkey-patch on executorch.examples.xnnpack.quantization.utils.quantize
to capture the float ExportedProgram with from_node metadata populated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
        PipelineGraphCollectorLens,
    )


def install_xnnpack_patches(cls: type[PipelineGraphCollectorLens]) -> None:
    """Install XNNPACK quantize patch on the PipelineGraphCollectorLens."""
    try:
        import executorch.examples.xnnpack.quantization.utils as xnnpack_qutils

        original = xnnpack_qutils.quantize
        cls._originals["xnnpack.quantize"] = original

        def patched_quantize(model, example_inputs, quant_type=None):
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

        def _uninstall():
            try:
                xnnpack_qutils.quantize = original
            except Exception:
                pass

        cls._backend_uninstallers.append(_uninstall)
    except Exception as exc:
        logging.warning(
            "[PipelineGraphCollector] Failed to patch XNNPACK quantize: %s", exc
        )
