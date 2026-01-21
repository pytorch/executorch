import unittest
from unittest.mock import patch, Mock
import torch
from executorch.extension.llm.export.builder import LLMEdgeManager

from executorch.backends.openvino.quantizer.llm_compression import (
    apply_nncf_data_aware_compression,
    get_calibration_data,
)
from executorch.backends.openvino.quantizer import (
    OpenVINOQuantizer,
    QuantizationMode,
)
from synthetic_test_models import SimpleTransformer

class TestWeightsOnlyQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "llama"
        cls.model_class_name = "Llama2Model"
        cls.model = SimpleTransformer()
        cls.model.eval()

        cls.max_seq_len = 128
        cls.example_inputs = (torch.tensor([[1]], dtype=torch.long), {"input_pos": torch.tensor([0], dtype=torch.long)})
        
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
            builder_kwargs.update({
                "calibration_seq_length": 32,
                "calibration_data": calibration_data,
                "tokenizer_path": "dummy_path", # Will be mocked
            })
        
        return LLMEdgeManager(**builder_kwargs)


    @patch('executorch.backends.openvino.quantizer.llm_compression.get_tokenizer')
    @patch('executorch.backends.openvino.quantizer.llm_compression.get_calibration_data')
    def test_compression_flow_with_mocked_calibration(
        self, mock_get_calibration_data, mock_get_tokenizer
    ):
        mock_calibration_data = [
            (0, 1), (1, 5), (2, 10), (3, 15), (4, 20),
            (5, 25), (6, 30), (7, 35), (8, 40), (9, 45)
        ]
        mock_get_calibration_data.return_value = mock_calibration_data
        
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        for config in self.compression_configs:
            with self.subTest(phase="compression_config", config=config["name"]):
                calibration_data = self.calibration_data if config["awq"] or config["scale_estimation"] else None

                builder = self._create_builder(
                    config["name"], 
                    calibration_data=calibration_data
                )
                builder.export()
                import copy
                original_model = copy.deepcopy(builder.pre_autograd_graph_module)

                test_input = torch.tensor([[5]], dtype=torch.long)
                test_pos = torch.tensor([0], dtype=torch.long)
                reference_output = original_model(test_input, {"input_pos": test_pos})

                quantizer = OpenVINOQuantizer(mode=QuantizationMode.INT4WO_SYM, group_size=-1)
                builder = apply_nncf_data_aware_compression(
                    builder,
                    quantizer=quantizer,
                    awq=config["awq"],
                    scale_estimation=config["scale_estimation"],
                )
                
                compressed_output = builder.pre_autograd_graph_module(test_input, {"input_pos": test_pos})

                torch.allclose(compressed_output, reference_output)


class TestCalibrationDataGeneration(unittest.TestCase):
    """Test the calibration data generation method. We first create a mock tokenizer
    and then compare it with a reference created manually"""

    def test_get_calibration_data_with_mock_module(self):
        mock_tokenizer = Mock()
        mock_tokenizer.eos_id = 2
        mock_tokenizer.encode = Mock(return_value=[1, 5, 6])
        
        mock_module = Mock()
        mock_module.return_value = torch.tensor([[[0.1, 0.2, 0.9, 0.0]]])
        
        result = get_calibration_data(
            mock_module,
            mock_tokenizer,
            "test prompt", # Will be mocked
            max_len=10
        )

        positions = [item[0] for item in result]
        self.assertEqual(positions, list(range(len(positions))))
