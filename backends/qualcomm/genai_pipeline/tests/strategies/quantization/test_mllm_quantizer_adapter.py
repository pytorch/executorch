# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.mllm_quantizer_adapter import (
    MLLMQuantizerAdapter,
)


def _fake_quantize_pt2e(**functions):
    module = types.ModuleType("torchao.quantization.pt2e.quantize_pt2e")
    for name, function in functions.items():
        setattr(module, name, function)
    return module


def _fake_llama_utils():
    module = types.ModuleType("executorch.examples.qualcomm.oss_scripts.llama.utils")
    module.safe_dataloader_iter = lambda dataloader: (
        itertools.chain(dataloader, itertools.repeat([]))
        if dataloader is not None
        else itertools.repeat([])
    )
    return module


class TestMLLMQuantizerAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MLLMQuantizerAdapter()

    def test_export_model_returns_exported_module(self):
        module = MagicMock(name="encoder")
        example_inputs = (MagicMock(name="inputs"),)
        exported_module = MagicMock(name="exported_module")
        exported_program = MagicMock(name="exported_program")
        exported_program.module.return_value = exported_module

        with patch("torch.export.export", return_value=exported_program) as export:
            result = self.adapter.export_model(module, example_inputs)

        self.assertIs(result, exported_module)
        export.assert_called_once_with(module, example_inputs, strict=True)

    def test_prepare_pt2e_delegates_to_torchao(self):
        module = MagicMock(name="exported_module")
        quantizer = MagicMock(name="quantizer")
        prepared_module = MagicMock(name="prepared_module")
        prepare_pt2e = MagicMock(return_value=prepared_module)

        with patch.dict(
            sys.modules,
            {
                "torchao.quantization.pt2e.quantize_pt2e": _fake_quantize_pt2e(
                    prepare_pt2e=prepare_pt2e
                )
            },
        ):
            result = self.adapter.prepare_pt2e(module, quantizer)

        self.assertIs(result, prepared_module)
        prepare_pt2e.assert_called_once_with(module, quantizer)

    def test_init_encodings_runs_one_forward(self):
        module = MagicMock(name="prepared_encoder")
        example_inputs = (MagicMock(name="inputs"),)

        result = self.adapter.init_encodings(module, example_inputs)

        self.assertIs(result, module)
        module.assert_called_once_with(*example_inputs)

    def test_calibrate_prefers_audio_encoder_and_inputs(self):
        audio_encoder = MagicMock(name="audio_encoder")
        vision_encoder = MagicMock(name="vision_encoder")
        tok_embedding = MagicMock(name="tok_embedding")
        text_decoder = MagicMock(name="text_decoder")
        inference = MagicMock(name="inference")
        audio_inputs = MagicMock(name="audio_inputs")
        vision_inputs = MagicMock(name="vision_inputs")
        text_batch = {
            "input_ids": MagicMock(name="input_ids"),
            "attention_mask": MagicMock(name="attention_mask"),
        }

        with patch.dict(
            sys.modules,
            {
                "executorch.examples.qualcomm.oss_scripts.llama.utils": _fake_llama_utils()
            },
        ):
            self.adapter.calibrate(
                {
                    ARTIFACT_AUDIO_ENCODER: audio_encoder,
                    ARTIFACT_VISION_ENCODER: vision_encoder,
                    ARTIFACT_TOK_EMBEDDING: tok_embedding,
                    ARTIFACT_TEXT_DECODER: text_decoder,
                },
                {
                    ARTIFACT_AUDIO_ENCODER: [{"inputs": audio_inputs}],
                    ARTIFACT_VISION_ENCODER: [{"inputs": vision_inputs}],
                    ARTIFACT_TEXT_DECODER: [text_batch],
                },
                inference=inference,
            )

        inference.predict_step.assert_called_once_with(
            text_decoder,
            input_ids=text_batch["input_ids"],
            attn_mask=text_batch["attention_mask"],
            tok_embedding=tok_embedding,
            encoder_module=audio_encoder,
            encoder_inputs=audio_inputs,
        )

    def test_calibrate_uses_vision_when_audio_is_absent(self):
        vision_encoder = MagicMock(name="vision_encoder")
        text_decoder = MagicMock(name="text_decoder")
        inference = MagicMock(name="inference")
        vision_inputs = MagicMock(name="vision_inputs")
        text_batch = {
            "input_ids": MagicMock(name="input_ids"),
            "attention_mask": MagicMock(name="attention_mask"),
        }

        with patch.dict(
            sys.modules,
            {
                "executorch.examples.qualcomm.oss_scripts.llama.utils": _fake_llama_utils()
            },
        ):
            self.adapter.calibrate(
                {
                    ARTIFACT_VISION_ENCODER: vision_encoder,
                    ARTIFACT_TEXT_DECODER: text_decoder,
                },
                {
                    ARTIFACT_AUDIO_ENCODER: None,
                    ARTIFACT_VISION_ENCODER: [{"inputs": vision_inputs}],
                    ARTIFACT_TEXT_DECODER: [text_batch],
                },
                inference=inference,
            )

        inference.predict_step.assert_called_once_with(
            text_decoder,
            input_ids=text_batch["input_ids"],
            attn_mask=text_batch["attention_mask"],
            tok_embedding=None,
            encoder_module=vision_encoder,
            encoder_inputs=vision_inputs,
        )

    def test_convert_pt2e_delegates_to_torchao(self):
        module = MagicMock(name="prepared_module")
        converted_module = MagicMock(name="converted_module")
        convert_pt2e = MagicMock(return_value=converted_module)

        with patch.dict(
            sys.modules,
            {
                "torchao.quantization.pt2e.quantize_pt2e": _fake_quantize_pt2e(
                    convert_pt2e=convert_pt2e
                )
            },
        ):
            result = self.adapter.convert_pt2e(module)

        self.assertIs(result, converted_module)
        convert_pt2e.assert_called_once_with(module)


if __name__ == "__main__":
    unittest.main()
