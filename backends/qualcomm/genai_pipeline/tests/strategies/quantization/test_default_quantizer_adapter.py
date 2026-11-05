# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import torch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_VISION_ENCODER,
    DECODE_QDQ_FILENAME,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.default_quantizer_adapter import (
    DefaultQuantizerAdapter,
)


class TestCalibrate(unittest.TestCase):

    def setUp(self):
        self.adapter = DefaultQuantizerAdapter()

    def test_runs_one_forward_pass_per_sample(self):
        model = MagicMock()
        data = [(torch.zeros(1, 2),), (torch.ones(1, 2),)]
        self.adapter.calibrate(model, data)
        self.assertEqual(model.call_count, len(data))

    def test_unpacks_tuple_into_positional_args(self):
        model = MagicMock()
        input_ids, attention_mask = torch.zeros(1, 2), torch.ones(1, 2)
        self.adapter.calibrate(model, [(input_ids, attention_mask)])
        model.assert_called_once_with(input_ids, attention_mask)

    def test_returns_the_model(self):
        model = MagicMock()
        self.assertIs(self.adapter.calibrate(model, []), model)

    def test_accepts_a_dataloader(self):
        # A DataLoader with a collate_fn that yields the sample tuple unchanged
        # is consumed directly, with no re-wrapping by the caller.
        model = MagicMock()
        input_ids, attention_mask = torch.zeros(1, 2), torch.ones(1, 2)
        dataloader = torch.utils.data.DataLoader(
            [(input_ids, attention_mask)],
            batch_size=1,
            collate_fn=lambda batch: batch[0],
        )
        self.adapter.calibrate(model, dataloader)
        model.assert_called_once_with(input_ids, attention_mask)

    def test_empty_dataset_is_a_no_op(self):
        model = MagicMock()
        self.adapter.calibrate(model, [])
        model.assert_not_called()

    def test_component_map_calibrates_components_with_data(self):
        text_model = MagicMock()
        vision_model = MagicMock()
        input_ids, attention_mask = torch.zeros(1, 2), torch.ones(1, 2)

        result = self.adapter.calibrate(
            {
                ARTIFACT_TEXT_DECODER: text_model,
                ARTIFACT_VISION_ENCODER: vision_model,
            },
            {
                ARTIFACT_TEXT_DECODER: [(input_ids, attention_mask)],
            },
        )

        self.assertIs(result[ARTIFACT_TEXT_DECODER], text_model)
        text_model.assert_called_once_with(input_ids, attention_mask)
        vision_model.assert_not_called()

    def test_save_quantized_module_exports_strict_qdq_program(self):
        model = MagicMock(name="quantized_model")
        example_inputs = (torch.zeros(1, 2),)
        exported_program = MagicMock(name="exported_program")

        with TemporaryDirectory() as artifact_dir, patch(
            "torch.export.export", return_value=exported_program
        ) as export, patch("torch.export.save") as save:
            from executorch.backends.qualcomm.genai_pipeline.quant_utilities import (
                save_quantized_module,
            )

            result = save_quantized_module(
                model,
                example_inputs,
                artifact_dir,
            )

            expected_path = Path(artifact_dir) / DECODE_QDQ_FILENAME
            export.assert_called_once_with(model, example_inputs, strict=True)
            save.assert_called_once_with(exported_program, expected_path)
            self.assertEqual(result, expected_path)


if __name__ == "__main__":
    unittest.main()
