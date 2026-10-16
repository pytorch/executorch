# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLMQuantizerAdapter: quantization adapter for LLMs.

The strategy selects the recipe and dtype for each component. This adapter
creates one quantizer from those scalar options and applies the supplied recipe.

Most methods operate on a **single graph module**. ``calibrate`` receives the
single-level ``{component: module}`` map selected by the strategy so it can drive
the text decoder. Fanning out over graph variants remains the strategy's job.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


class LLMQuantizerAdapter:
    """Quantizer adapter for LLM models.

    Handles LLM-specific calibration and encoding propagation.
    """

    def export_model(
        self,
        module: Any,
        example_inputs: Any,
    ) -> Any:
        """Export a single graph module using torch.export.export.

        Args:
            module: The decoder module to export.
            example_inputs: Positional example inputs for this graph.

        Returns:
            The exported module.
        """
        return torch.export.export(module, example_inputs, strict=True).module()

    def prepare_pt2e(
        self,
        module: Any,
        quantizer: Any,
    ) -> Any:
        """Prepare a single exported module for PT2E quantization.

        Args:
            module: The exported module.
            quantizer: The QnnQuantizer instance.

        Returns:
            The annotated module with observers inserted.
        """
        from torchao.quantization.pt2e.quantize_pt2e import (
            prepare_pt2e as _prepare_pt2e,
        )

        return _prepare_pt2e(module, quantizer)

    def init_encodings(
        self,
        module: Any,
        example_inputs: Any,
    ) -> Any:
        """Initialize a deployed graph's observers with one dummy forward.

        Args:
            module: The annotated deployed graph module.
            example_inputs: This graph's positional example-input tuple.

        Returns:
            The module after the dummy forward.
        """
        with torch.no_grad():
            module(*example_inputs)
        return module

    def calibrate(
        self,
        modules: Any,
        calibration_data: Iterable[Any],
        **kwargs: Any,
    ):
        """Run true PTQ calibration over the selected decoder.

        ``modules`` is the single-level ``{component: module}`` map created by
        the strategy. It has no graph axis because it contains only the selected
        calibration graph for each component.

        Args:
            modules: ``{ARTIFACT_TEXT_DECODER: decoder_module}``.
            calibration_data: ``{component: DataLoader}`` corpus batches.
            **kwargs: Extra adapter-specific options. Expects ``inference``, the
                ``ModelInference`` bound to the calibration graph.

        Returns:
            ``None``. The decoder module is calibrated in place.
        """
        import torch

        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )
        from torch.utils.data import DataLoader

        inference = kwargs["inference"]
        text_dataloader = calibration_data.get(ARTIFACT_TEXT_DECODER)
        if not isinstance(text_dataloader, DataLoader):
            raise ValueError(
                "Calibration requires a corpus-backed DataLoader for "
                f"{ARTIFACT_TEXT_DECODER};"
            )

        decoder_module = modules[ARTIFACT_TEXT_DECODER]
        with torch.no_grad():
            for batch in text_dataloader:
                inference.predict_step(
                    decoder_module,
                    input_ids=batch["input_ids"],
                    attn_mask=batch["attention_mask"],
                )

    def convert_pt2e(
        self,
        module: Any,
    ) -> Any:
        """Convert a single calibrated module to a quantized module.

        Cross-graph encoding override and the recording of quantized logits /
        KV-cache attributes are performed by the strategy after this returns.

        Args:
            module: The calibrated module.

        Returns:
            The quantized module.
        """
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e as _convert_pt2e,
        )

        return _convert_pt2e(module)
