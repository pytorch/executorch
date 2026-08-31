# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import create_autospec, MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.executorch_quantization_strategy import (
    ExecuTorchQuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantizer_adapter import (
    QuantizerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


def _make_mock_adapter():
    """Create a mock adapter with all methods returning sensible defaults.

    Autospec'd against ``QuantizerAdapter`` rather than a bare ``MagicMock``: a
    bare mock accepts any call whatsoever, so a strategy call that no real
    adapter could satisfy -- omitting an argument the Protocol declares as
    required -- would pass here and only fail in production. The autospec binds
    every call to the Protocol signature, keeping Protocol, adapter and strategy
    in step.

    Its limit is worth stating: ``make_quantizer`` takes ``**kwargs`` by design,
    to forward ``extra_options`` verbatim, so no signature check can reject a
    keyword the *underlying* API does not accept. That the forwarded keywords
    are ones ``export_utils.make_quantizer`` actually takes is only observable
    against the real adapter, and belongs in the integration tests.
    """
    adapter = create_autospec(QuantizerAdapter, instance=True)
    adapter.export_model.return_value = MagicMock(name="exported_model")
    adapter.make_quantizer.return_value = MagicMock(name="quantizer")
    adapter.prepare_pt2e.return_value = MagicMock(name="annotated_model")
    adapter.calibrate.return_value = MagicMock(name="calibrated_model")
    adapter.convert_pt2e.return_value = MagicMock(name="quantized_model")
    return adapter


def _make_valid_input_config(**overrides):
    """Create a valid QuantizationInputConfig with defaults."""
    defaults = {
        "soc_model": MagicMock(name="SM8750"),
        "backend_type": MagicMock(name="kHtpBackend"),
        "model_module": MagicMock(name="test_model"),
        "example_inputs": (MagicMock(name="example_input"),),
        "calibration_data": [(MagicMock(),)],
    }
    defaults.update(overrides)
    return QuantizationInputConfig(**defaults)


class TestExecuTorchQuantizationStrategy(unittest.TestCase):

    def test_is_quantization_strategy(self):
        """Strategy inherits from QuantizationStrategy ABC."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        self.assertIsInstance(strategy, QuantizationStrategy)

    def test_default_adapter_created_when_none_provided(self):
        """When no adapter is provided, DefaultQuantizerAdapter is created."""
        with patch(
            "executorch.backends.qualcomm.genai_pipeline.strategies.quantization."
            "default_quantizer_adapter.DefaultQuantizerAdapter"
        ) as mock_cls:
            strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=None)
            mock_cls.assert_called_once()
            self.assertIs(strategy.adapter, mock_cls.return_value)

    def test_custom_adapter_injected(self):
        """Custom adapter is used when provided via constructor."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        self.assertIs(strategy.adapter, adapter)

    def test_invoke_happy_path(self):
        """Full quantization pipeline runs successfully end-to-end."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertIsInstance(result, QuantizationOutputConfig)
        self.assertIs(result.quantized_model, adapter.convert_pt2e.return_value)

    def test_invoke_calls_adapter_in_correct_order(self):
        """Adapter methods are called in the correct PT2E sequence."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)

        strategy.invoke(make_test_context(), _make_valid_input_config())

        # Verify call order: export → make_quantizer → prepare → calibrate → convert
        self.assertEqual(
            [c[0] for c in adapter.method_calls],
            [
                "export_model",
                "make_quantizer",
                "prepare_pt2e",
                "calibrate",
                "convert_pt2e",
            ],
        )

    def test_invoke_exports_with_example_inputs_not_calibration_data(self):
        """export_model receives ``example_inputs``, never a calibration sample.

        The export signature comes from the model (zero-initialized KV caches,
        fixed AR length); a calibration sample has neither, so sourcing it from
        the dataset would export the wrong graph.
        """
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        model = MagicMock(name="model")
        example_inputs = (MagicMock(name="tokens"), MagicMock(name="attn_mask"))
        calibration_sample = (MagicMock(name="calibration_sample"),)
        input_config = _make_valid_input_config(
            model_module=model,
            example_inputs=example_inputs,
            calibration_data=[calibration_sample],
        )

        strategy.invoke(make_test_context(), input_config)

        adapter.export_model.assert_called_once_with(model, example_inputs)

    def test_invoke_passes_calibration_data_through_untouched(self):
        """``calibration_data`` reaches ``calibrate`` as the very same object.

        Nothing peeks at, indexes or copies it, so a single-use generator keeps
        every sample and a DataLoader keeps streaming instead of being pulled
        into memory.
        """
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        samples = [(MagicMock(name=f"sample{i}"),) for i in range(3)]
        dataset = (sample for sample in samples)
        input_config = _make_valid_input_config(calibration_data=dataset)

        strategy.invoke(make_test_context(), input_config)

        self.assertIs(adapter.calibrate.call_args[0][1], dataset)
        # Untouched by the strategy, so all three samples are still available.
        self.assertEqual(list(adapter.calibrate.call_args[0][1]), samples)

    def test_invoke_missing_example_inputs_raises_stage_error(self):
        """StageError raised when example_inputs is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        input_config = _make_valid_input_config(example_inputs=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("example_inputs", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "quantization")
        adapter.export_model.assert_not_called()

    def test_invoke_passes_correct_args_to_make_quantizer(self):
        """make_quantizer receives backend_type and soc_model (no quant_dtype when not set)."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        soc = MagicMock(name="soc")
        backend = MagicMock(name="backend")
        input_config = _make_valid_input_config(soc_model=soc, backend_type=backend)

        strategy.invoke(make_test_context(), input_config)

        # quant_dtype is NOT passed when not explicitly set in extra_options, so
        # the default owned by export_utils.make_quantizer applies.
        adapter.make_quantizer.assert_called_once_with(
            backend=backend,
            soc_model=soc,
        )

    def test_invoke_passes_quant_dtype_from_extra_options(self):
        """quant_dtype extracted from extra_options and forwarded."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        quant_dtype = MagicMock(name="quant_dtype")
        input_config = _make_valid_input_config(
            extra_options={"quant_dtype": quant_dtype}
        )

        strategy.invoke(make_test_context(), input_config)

        adapter.make_quantizer.assert_called_once_with(
            quant_dtype=quant_dtype,
            backend=input_config.backend_type,
            soc_model=input_config.soc_model,
        )

    def test_invoke_passes_quant_recipe_from_config(self):
        """quant_recipe on the config reaches the adapter as a declared kwarg.

        The adapter consumes it (``QnnQuantizer.set_recipe``) rather than
        forwarding it to ``export_utils.make_quantizer``, which takes no such
        argument.
        """
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        recipe = MagicMock(name="quant_recipe")
        input_config = _make_valid_input_config(quant_recipe=recipe)

        strategy.invoke(make_test_context(), input_config)

        _, kwargs = adapter.make_quantizer.call_args
        self.assertIs(kwargs["quant_recipe"], recipe)

    def test_invoke_training_data_does_not_switch_to_qat(self):
        """``training_data`` is accepted but this strategy still performs PTQ."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        input_config = _make_valid_input_config(training_data=[(MagicMock(),)])

        with self.assertLogs(
            "executorch.backends.qualcomm.genai_pipeline.strategies.quantization."
            "executorch_quantization_strategy",
            level="WARNING",
        ):
            result = strategy.invoke(make_test_context(), input_config)

        # Still the plain PTQ sequence.
        adapter.calibrate.assert_called_once()
        self.assertIs(result.quantized_model, adapter.convert_pt2e.return_value)

    def test_invoke_missing_model_raises_stage_error(self):
        """StageError raised when model_module is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        input_config = _make_valid_input_config(model_module=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("model_module", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "quantization")

    def test_invoke_none_calibration_data_raises_stage_error(self):
        """StageError raised when calibration_data is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        input_config = _make_valid_input_config(calibration_data=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("calibration_data", str(cm.exception))

    def test_invoke_adapter_exception_wrapped_in_stage_error(self):
        """Exceptions from the adapter are wrapped in StageError."""
        adapter = _make_mock_adapter()
        adapter.export_model.side_effect = RuntimeError("export failed")
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertEqual(cm.exception.stage_name, "quantization")
        self.assertIsInstance(cm.exception.original_exception, RuntimeError)
        self.assertIn("export failed", str(cm.exception))

    def test_invoke_stage_error_not_double_wrapped(self):
        """StageError from adapter is re-raised directly, not wrapped again."""
        adapter = _make_mock_adapter()
        original_error = StageError(stage_name="quantization", message="inner error")
        adapter.export_model.side_effect = original_error
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertIs(cm.exception, original_error)

    def test_invoke_missing_soc_model_raises_stage_error(self):
        """StageError raised when soc_model is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        input_config = _make_valid_input_config(soc_model=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("soc_model", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "quantization")

    def test_invoke_missing_backend_type_raises_stage_error(self):
        """StageError raised when backend_type is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        input_config = _make_valid_input_config(backend_type=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("backend_type", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "quantization")

    def test_invoke_extra_options_forwarded_to_make_quantizer(self):
        """Extra options (minus quant_dtype) forwarded as kwargs."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchQuantizationStrategy(quantizer_adapter=adapter)
        input_config = _make_valid_input_config(
            extra_options={
                "quant_dtype": "test_dtype",
                "per_channel_conv": True,
                "act_symmetric": True,
            }
        )

        strategy.invoke(make_test_context(), input_config)

        adapter.make_quantizer.assert_called_once_with(
            quant_dtype="test_dtype",
            backend=input_config.backend_type,
            soc_model=input_config.soc_model,
            per_channel_conv=True,
            act_symmetric=True,
        )


if __name__ == "__main__":
    unittest.main()
