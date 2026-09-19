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
        quant_dtype: Any = None,
        backend: Any = None,
        soc_model: Any = None,
        quant_recipe: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Create a QNN quantizer via ``export_utils.make_quantizer``.

        ``quant_dtype`` defaults to ``None`` and is only forwarded when set, so
        ``export_utils.make_quantizer`` remains the single owner of the default
        (``QuantDtype.use_8a8w``) rather than this wrapper duplicating it.

        ``quant_recipe`` is **not** an argument of
        ``export_utils.make_quantizer``; a recipe is applied to the constructed
        quantizer via ``QnnQuantizer.set_recipe``, so it is consumed here and
        never forwarded.

        Args:
            quant_dtype: Quantization data type. ``None`` leaves the default to
                ``export_utils.make_quantizer``.
            backend: QNN backend type enum.
            soc_model: Target SoC (string name like "SM8750" or QcomChipset enum).
            quant_recipe: Optional recipe applied via ``set_recipe`` after the
                quantizer is constructed.
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
            "Creating quantizer: dtype=%s, backend=%s, soc=%s, recipe=%s",
            quant_dtype,
            backend,
            soc_model_str,
            quant_recipe,
        )
        make_quantizer_kwargs = {
            "backend": backend,
            "soc_model": soc_model_str,
            **kwargs,
        }
        if quant_dtype is not None:
            make_quantizer_kwargs["quant_dtype"] = quant_dtype

        quantizer = _make_quantizer(**make_quantizer_kwargs)

        if quant_recipe is not None:
            logger.debug("Applying quantization recipe via set_recipe")
            quantizer.set_recipe(quant_recipe)

        return quantizer

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
