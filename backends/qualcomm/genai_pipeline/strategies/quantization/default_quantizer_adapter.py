# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)


class DefaultQuantizerAdapter:
    """Default adapter delegating to real ExecuTorch/QNN quantization APIs.

    Wraps ``torchao.quantization.pt2e.prepare_pt2e`` and
    ``torchao.quantization.pt2e.convert_pt2e`` for production use.
    """

    def export_model(
        self,
        model: Any,
        sample_input: Any,
    ) -> Any:
        """Export the model using ``torch.export.export``.

        Args:
            model: The nn.Module to export.
            sample_input: Sample input tuple for tracing.

        Returns:
            The exported module (``ExportedProgram.module()``).
        """
        logger.debug("Exporting model via torch.export.export")
        return torch.export.export(model, sample_input, strict=True).module()

    def prepare_pt2e(
        self,
        model: Any,
        quantizer: Any,
    ) -> Any:
        """Prepare the model for PT2E quantization.

        Args:
            model: The exported model.
            quantizer: The configured quantizer.

        Returns:
            The annotated model with observers inserted.
        """
        from torchao.quantization.pt2e.quantize_pt2e import (
            prepare_pt2e as _prepare_pt2e,
        )

        logger.debug("Preparing model for PT2E quantization")
        return _prepare_pt2e(model, quantizer)

    def init_encodings(
        self,
        module: Any,
        example_inputs: Any,
    ) -> Any:
        """Initialize a graph's observers with one dummy forward.

        Args:
            module: The annotated graph module.
            example_inputs: This graph's positional example-input tuple.

        Returns:
            The module after the dummy forward.
        """
        logger.debug("Initializing encodings with a dummy forward")
        with torch.no_grad():
            module(*example_inputs)
        return module

    def calibrate(
        self,
        model: Any,
        calibration_data: Iterable[Any],
        **kwargs: Any,
    ) -> Any:
        """Run calibration data through the annotated model.

        One forward pass per sample. The default adapter supports both the
        legacy single-module shape and the strategy's component-map shape:
        ``{component: module}`` is paired with ``{component: iterable}``, and
        components without calibration data are left untouched. Adapters needing
        a stateful procedure -- e.g. autoregressive LLM calibration, where each
        step's input depends on the previous step's output and the KV cache
        mutates across steps -- should override this method.

        Args:
            model: The annotated model with observers, or a component-keyed map
                of selected calibration graph modules.
            calibration_data: Any ``Iterable[Tuple[Tensor, ...]]`` for a single
                module, or a component-keyed map of such iterables. Plain lists
                and ``DataLoader`` instances are both accepted.
            **kwargs: Extra adapter-specific options. Ignored by the default
                adapter; model-specific adapters may read keys such as
                ``inference``.

        Returns:
            The calibrated model or component map.
        """
        logger.debug("Running calibration")
        if isinstance(model, dict):
            with torch.no_grad():
                for component, module in model.items():
                    component_data = calibration_data.get(component, ())
                    for data in component_data:
                        module(*data)
            return model

        with torch.no_grad():
            for data in calibration_data:
                model(*data)
        return model

    def convert_pt2e(
        self,
        model: Any,
    ) -> Any:
        """Convert the calibrated model to a quantized model.

        Args:
            model: The calibrated model.

        Returns:
            The quantized model.
        """
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e as _convert_pt2e,
        )

        logger.debug("Converting PT2E model to quantized form")
        return _convert_pt2e(model)
