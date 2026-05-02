# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XNNPACK backend patches for PipelineGraphCollectorLens.

Installs a monkey-patch on XNNPACK quantization helpers (both
``examples`` and ``executorch.examples`` import paths) to capture the
float ExportedProgram with from_node metadata populated.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
        PipelineGraphCollectorLens,
    )


MODULE_CANDIDATES = (
    "examples.xnnpack.quantization.utils",
    "executorch.examples.xnnpack.quantization.utils",
)


def _install_patch_for_module(
    cls: type[PipelineGraphCollectorLens], module, alias: str
) -> bool:
    try:
        original = module.quantize
    except AttributeError:
        logging.debug(
            "[PipelineGraphCollector] XNNPACK patch skipped; no quantize in %s",
            alias,
        )
        return False

    key = f"xnnpack.quantize[{alias}]"
    if key in cls._originals:
        return True

    cls._originals[key] = original

    def patched_quantize(model, example_inputs, quant_type=None):
        sample = None
        try:
            if isinstance(example_inputs, (tuple, list)):
                sample = tuple(example_inputs)
            else:
                sample = (example_inputs,)
            cls._set_accuracy_fallback_dataset([sample], source=key)
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

    module.quantize = patched_quantize
    logging.info(
        "[PipelineGraphCollector] Installed XNNPACK patch: quantize (%s)", alias
    )

    def _uninstall():
        try:
            module.quantize = original
        except Exception:
            pass

    cls._backend_uninstallers.append(_uninstall)
    return True


def install_xnnpack_patches(cls: type[PipelineGraphCollectorLens]) -> None:
    """Install XNNPACK quantize patch on the PipelineGraphCollectorLens."""

    patched = False
    seen_modules: set[int] = set()

    for alias in MODULE_CANDIDATES:
        try:
            module = importlib.import_module(alias)
        except ImportError:
            continue

        module_id = id(module)
        if module_id in seen_modules:
            continue
        seen_modules.add(module_id)

        try:
            patched |= _install_patch_for_module(cls, module, alias)
        except Exception as exc:
            logging.warning(
                "[PipelineGraphCollector] Failed to patch XNNPACK quantize (%s): %s",
                alias,
                exc,
            )

    if not patched:
        logging.warning(
            "[PipelineGraphCollector] Failed to patch XNNPACK quantize: no candidate module found"
        )
