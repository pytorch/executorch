# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.llm_quantizer_adapter import (
    LLMQuantizerAdapter,
)
from torch.utils.data import DataLoader


def _fake_quantize_pt2e(**functions):
    module = types.ModuleType("torchao.quantization.pt2e.quantize_pt2e")
    for name, function in functions.items():
        setattr(module, name, function)
    return module


class TestLLMQuantizerAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = LLMQuantizerAdapter()

    def test_export_model_returns_exported_module(self):
        module = MagicMock(name="decoder")
        example_inputs = (MagicMock(name="tokens"),)
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
        module = MagicMock(name="prepared_decoder")
        example_inputs = (MagicMock(name="tokens"), MagicMock(name="attn_mask"))

        result = self.adapter.init_encodings(module, example_inputs)

        self.assertIs(result, module)
        module.assert_called_once_with(*example_inputs)

    def test_calibrate_drives_decoder_with_corpus_batches(self):
        decoder = MagicMock(name="decoder")
        inference = MagicMock(name="inference")
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        dataloader = DataLoader([batch], batch_size=None)

        result = self.adapter.calibrate(
            {"text_decoder": decoder},
            {"text_decoder": dataloader},
            inference=inference,
        )

        self.assertIsNone(result)
        inference.predict_step.assert_called_once_with(
            decoder,
            input_ids=batch["input_ids"],
            attn_mask=batch["attention_mask"],
        )

    def test_calibrate_rejects_non_dataloader_text_data(self):
        with self.assertRaisesRegex(ValueError, "corpus-backed DataLoader"):
            self.adapter.calibrate(
                {"text_decoder": MagicMock(name="decoder")},
                {"text_decoder": []},
                inference=MagicMock(name="inference"),
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
