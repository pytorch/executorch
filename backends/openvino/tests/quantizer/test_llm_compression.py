import unittest
from unittest.mock import Mock, patch

import torch
from executorch.backends.openvino.quantizer import OpenVINOQuantizer, QuantizationMode

from executorch.backends.openvino.quantizer.llm_compression import (
    apply_nncf_data_aware_compression,
    get_calibration_data,
    transform_fn,
)
from executorch.extension.llm.export.builder import LLMEdgeManager
from synthetic_test_models import ExportLlamaTestModel  # type: ignore[import-not-found]


class TestWeightsOnlyQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = ExportLlamaTestModel(vocab_size=5, hidden_size=2, num_layers=1)
        cls.model.eval()

        cls.max_seq_len = 128
        cls.example_inputs = (
            torch.tensor([[1]], dtype=torch.long),
            {"input_pos": torch.tensor([0], dtype=torch.long)},
        )

        cls.compression_configs = [
            {
                "name": "awq_only",
                "awq": True,
                "scale_estimation": False,
            },
            {
                "name": "scale_estimation_only",
                "awq": False,
                "scale_estimation": True,
            },
            {
                "name": "awq_and_scale_estimation",
                "awq": True,
                "scale_estimation": True,
            },
            {
                "name": "no_calibration",
                "awq": False,
                "scale_estimation": False,
            },
        ]

        cls.calibration_data = "The quick brown fox jumps over the lazy dog."

        cls.reference_scales = {
            "awq_only": {
                "symmetric_weights_decompressor_embed_weight._scale": torch.tensor(
                    [[-0.042084], [-0.029312], [0.140381], [-0.276123], [-0.057709]],
                    dtype=torch.float16,
                ),
                "symmetric_weights_decompressor_layers_0_weight._scale": torch.tensor(
                    [[0.040710], [-0.058624]], dtype=torch.float16
                ),
                "relu/awq_mul._scale_value": torch.tensor([[[1.0, 1.0]]]),
                "symmetric_weights_decompressor_lm_head_weight_updated_constant0._scale": torch.tensor(
                    [[0.053131], [0.087280], [-0.079834], [-0.068237], [-0.054626]],
                    dtype=torch.float16,
                ),
            },
            "scale_estimation_only": {
                "symmetric_weights_decompressor_embed_weight._scale": torch.tensor(
                    [[-0.042084], [-0.029312], [0.140381], [-0.276123], [-0.057709]],
                    dtype=torch.float16,
                ),
                "symmetric_weights_decompressor_layers_0_weight._scale": torch.tensor(
                    [[0.040710], [-0.057709]], dtype=torch.float16
                ),
                "symmetric_weights_decompressor_lm_head_weight._scale": torch.tensor(
                    [[0.0], [0.0], [-0.0], [-0.0], [-0.0]], dtype=torch.float16
                ),
            },
            "awq_and_scale_estimation": {
                "symmetric_weights_decompressor_embed_weight._scale": torch.tensor(
                    [[-0.042084], [-0.029312], [0.140381], [-0.276123], [-0.057709]],
                    dtype=torch.float16,
                ),
                "symmetric_weights_decompressor_layers_0_weight._scale": torch.tensor(
                    [[0.040710], [-0.057709]], dtype=torch.float16
                ),
                "relu/awq_mul._scale_value": torch.tensor([[[1.0, 1.0]]]),
                "symmetric_weights_decompressor_lm_head_weight_updated_constant0._scale": torch.tensor(
                    [[0.0], [0.0], [-0.0], [-0.0], [-0.0]], dtype=torch.float16
                ),
            },
            "no_calibration": {
                "symmetric_weights_decompressor_embed_weight._scale": torch.tensor(
                    [[-0.042084], [-0.029312], [0.140381], [-0.276123], [-0.057709]],
                    dtype=torch.float16,
                ),
                "symmetric_weights_decompressor_layers_0_weight._scale": torch.tensor(
                    [[0.040710], [-0.058624]], dtype=torch.float16
                ),
                "symmetric_weights_decompressor_lm_head_weight._scale": torch.tensor(
                    [[0.053131], [0.087280], [-0.079834], [-0.068237], [-0.054626]],
                    dtype=torch.float16,
                ),
            },
        }

    def _create_builder(self, config_name, calibration_data=None):
        builder_kwargs = {
            "model": self.model,
            "modelname": f"tinyllama_{config_name}",
            "max_seq_len": self.max_seq_len,
            "use_kv_cache": True,
            "example_inputs": self.example_inputs,
            "example_kwarg_inputs": None,
        }

        if calibration_data:
            builder_kwargs.update(
                {
                    "calibration_seq_length": 32,
                    "calibration_data": calibration_data,
                    "tokenizer_path": "dummy_path",
                }
            )

        return LLMEdgeManager(**builder_kwargs)

    def _extract_scales_from_model(self, model):
        extracted_scales = {}
        state_dict = dict(model.state_dict())
        for name, _ in state_dict.items():
            if "_scale" in name.lower():
                extracted_scales[name] = state_dict[name]
        return extracted_scales

    def _compare_scales(self, extracted_scales, reference_scales):
        for name, reference_value in reference_scales.items():
            self.assertIn(name, extracted_scales, f"Scale {name} not found in model")
            extracted_value = extracted_scales[name]
            self.assertTrue(
                torch.allclose(extracted_value, reference_value),
                f"Scale {name} mismatch {extracted_value}",
            )

    @patch("executorch.backends.openvino.quantizer.llm_compression.get_tokenizer")
    @patch(
        "executorch.backends.openvino.quantizer.llm_compression.get_calibration_data"
    )
    def test_compression_flow_with_mocked_calibration(
        self, mock_get_calibration_data, mock_get_tokenizer
    ):
        mock_calibration_data = [(i, i) for i in range(5)]
        mock_get_calibration_data.return_value = mock_calibration_data

        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer

        for config in self.compression_configs:
            with self.subTest(phase="compression_config", config=config["name"]):
                calibration_data = (
                    self.calibration_data
                    if config["awq"] or config["scale_estimation"]
                    else None
                )

                builder = self._create_builder(
                    config["name"], calibration_data=calibration_data
                )
                builder.export()

                test_input = torch.tensor([[4]], dtype=torch.long)
                test_pos = torch.tensor([0], dtype=torch.long)
                # Quantize weights for all layers(including embedding and lm_head which would by default be in INT8)
                # to Per-Channel INT4 Symmetric
                quantizer = OpenVINOQuantizer(
                    mode=QuantizationMode.INT4WO_SYM, group_size=-1, all_layers=True
                )
                builder = apply_nncf_data_aware_compression(
                    builder,
                    quantizer=quantizer,
                    awq=config["awq"],
                    scale_estimation=config["scale_estimation"],
                )
                # Run the model to check it is performant
                builder.pre_autograd_graph_module(test_input, {"input_pos": test_pos})
                extracted_scales = self._extract_scales_from_model(
                    builder.pre_autograd_graph_module
                )
                self._compare_scales(
                    extracted_scales,
                    self.reference_scales[config["name"]],
                )

    def test_scale_estimation_requires_calibration_params(self):
        builder = self._create_builder(
            "missing_calibration_data", calibration_data=None
        )
        builder.export()

        quantizer = OpenVINOQuantizer(
            mode=QuantizationMode.INT4WO_SYM, group_size=-1, all_layers=True
        )

        with self.assertRaises(ValueError) as cm:
            apply_nncf_data_aware_compression(
                builder,
                quantizer=quantizer,
                awq=False,
                scale_estimation=True,
            )

        err = str(cm.exception)
        self.assertIn("Missing required calibration parameter(s)", err)
        self.assertIn("calibration_data", err)
        self.assertIn("calibration_seq_length", err)
        self.assertIn("tokenizer_path", err)


class TestCalibrationDataGeneration(unittest.TestCase):

    def test_get_calibration_data_with_mock_module(self):
        mock_tokenizer = Mock()
        mock_tokenizer.eos_id = 2
        mock_tokenizer.encode = Mock(return_value=[1, 5, 6])

        mock_module = Mock()
        mock_module.return_value = torch.tensor([[[0.1, 0.2, 0.9, 0.0]]])

        result = get_calibration_data(
            mock_module, mock_tokenizer, "test prompt", max_len=10
        )

        positions = [item[0] for item in result]
        self.assertEqual(positions, list(range(len(positions))))

    def test_transform_fn(self):
        token_pos_map = (5, 10)
        result = transform_fn(token_pos_map)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        token, input_pos_dict = result
        self.assertEqual(token.shape, torch.Size([1, 1]))
        self.assertEqual(token, torch.tensor([[10]]))
        self.assertIn("input_pos", input_pos_dict)
        self.assertEqual(input_pos_dict["input_pos"], torch.tensor([5]))
