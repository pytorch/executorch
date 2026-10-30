# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantizer adapter for multimodal models.

Most methods operate on a **single graph module**. ``calibrate`` receives the
single-level ``{component: module}`` map selected by the strategy so it can
coordinate encoder, embedding, and decoder calibration. Graph fan-out and graph
selection remain the strategy's job, including choosing each component's recipe
and dtype before creating its quantizer.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


class MLLMQuantizerAdapter:
    """Quantizer adapter for multimodal models."""

    def export_model(
        self,
        module: Any,
        example_inputs: Any,
    ) -> Any:
        """Export a single graph module using torch.export.export.

        Args:
            module: The component module to export.
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
    ) -> Any:
        """Calibrate selected components in a ``{component: module}`` map.

        The graph axis has already been resolved by the strategy. This method
        coordinates the encoder, embedding, and decoder modules for one
        multimodal calibration path.

        Args:
            modules: Selected encoder, embedding, and decoder modules.
            calibration_data: Per-component calibration batches.
            **kwargs: Extra adapter-specific options. Expects ``inference``, the
                model-specific calibration driver.
        """

        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_AUDIO_ENCODER,
            ARTIFACT_TEXT_DECODER,
            ARTIFACT_TOK_EMBEDDING,
            ARTIFACT_VISION_ENCODER,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.utils import (
            safe_dataloader_iter,
        )

        inference = kwargs["inference"]
        audio_dataloader = calibration_data.get(ARTIFACT_AUDIO_ENCODER)
        vision_dataloader = calibration_data.get(ARTIFACT_VISION_ENCODER)
        text_dataloader = calibration_data[ARTIFACT_TEXT_DECODER]

        audio_encoder = modules.get(ARTIFACT_AUDIO_ENCODER)
        vision_encoder = modules.get(ARTIFACT_VISION_ENCODER)
        tok_embedding = modules.get(ARTIFACT_TOK_EMBEDDING)
        text_decoder = modules.get(ARTIFACT_TEXT_DECODER)
        encoder = audio_encoder or vision_encoder

        for _, (audio_batch, vision_batch, text_batch) in enumerate(
            zip(
                safe_dataloader_iter(audio_dataloader),
                safe_dataloader_iter(vision_dataloader),
                text_dataloader,
            )
        ):
            encoder_inputs = (audio_batch or vision_batch or {}).get("inputs")
            inference.predict_step(
                text_decoder,
                input_ids=text_batch["input_ids"],
                attn_mask=text_batch["attention_mask"],
                tok_embedding=tok_embedding,
                encoder_module=encoder,
                encoder_inputs=encoder_inputs,
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
