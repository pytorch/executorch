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

    Wraps ``export_utils.make_quantizer``, ``torchao.quantization.pt2e.prepare_pt2e``,
    and ``torchao.quantization.pt2e.convert_pt2e`` for production use.
    """

    def make_quantizer(
        self,
        quant_dtype: Any,
        backend: Any,
        soc_model: Any,
        **kwargs: Any,
    ) -> Any:
        """Create a QNN quantizer via ``export_utils.make_quantizer``.

        Args:
            quant_dtype: Quantization data type.
            backend: QNN backend type enum.
            soc_model: Target SoC (string name like "SM8750" or QcomChipset enum).
            **kwargs: Forwarded to ``make_quantizer``.

        Returns:
            A configured ``QnnQuantizer`` instance.
        """
        from executorch.backends.qualcomm.export_utils import (
            make_quantizer as _make_quantizer,
        )

        # export_utils.make_quantizer expects soc_model as a string for
        # getattr(QcomChipset, soc_model) lookup. Normalize enum → string.
        soc_model_str = soc_model.name if hasattr(soc_model, "name") else str(soc_model)

        logger.debug(
            "Creating quantizer: dtype=%s, backend=%s, soc=%s",
            quant_dtype,
            backend,
            soc_model_str,
        )
        return _make_quantizer(
            quant_dtype=quant_dtype,
            backend=backend,
            soc_model=soc_model_str,
            **kwargs,
        )

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
        return torch.export.export(model, sample_input, strict=False).module()

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

    def calibrate(
        self,
        model: Any,
        calibration_data: Iterable[Any],
    ) -> Any:
        """Run calibration data through the annotated model.

        One forward pass per sample. Adapters needing a stateful procedure --
        e.g. autoregressive LLM calibration, where each step's input depends on
        the previous step's output and the KV cache mutates across steps --
        should override this method rather than passing a callable as data.

        Args:
            model: The annotated model with observers.
            calibration_data: Any ``Iterable[Tuple[Tensor, ...]]``, including a
                plain list or a ``DataLoader``.

        Returns:
            The calibrated model.
        """
        logger.debug("Running calibration")
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
