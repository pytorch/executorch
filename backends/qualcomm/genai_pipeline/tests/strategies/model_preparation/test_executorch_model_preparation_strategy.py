# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.default_calibration_data_adapter import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEQ_LENGTH,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.executorch_model_preparation_strategy import (
    ExecuTorchModelPreparationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.model_preparation_strategy import (
    ModelPreparationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


def _make_mock_adapter():
    """Create a mock model loader adapter with sensible defaults."""
    adapter = MagicMock()
    adapter.load_model.return_value = MagicMock(name="model_module")
    tokenizer = MagicMock(name="tokenizer")
    # Default to "no chat template" so tests opt in explicitly.
    tokenizer.chat_template = None
    adapter.load_tokenizer.return_value = tokenizer
    adapter.get_example_inputs.return_value = (MagicMock(name="example_input"),)
    adapter.export_tokenizer.return_value = Path("/tmp/tokenizer/tokenizer.json")
    return adapter


def _make_mock_calibration_adapter():
    """Create a mock calibration data adapter with sensible defaults."""
    adapter = MagicMock()
    adapter.generate_calibration_data.return_value = [(MagicMock(),)]
    return adapter


def _make_strategy(loader=None, calibration=None):
    """Build the strategy with both adapters mocked by default."""
    return ExecuTorchModelPreparationStrategy(
        model_loader_adapter=loader if loader is not None else _make_mock_adapter(),
        calibration_data_adapter=(
            calibration if calibration is not None else _make_mock_calibration_adapter()
        ),
    )


def _make_valid_input_config(**overrides):
    """Create a valid ModelPreparationInputConfig with defaults."""
    defaults = {
        "model_name": "test_model",
        "soc_model": "SM8750",
    }
    defaults.update(overrides)
    return ModelPreparationInputConfig(**defaults)


class TestExecuTorchModelPreparationStrategy(unittest.TestCase):

    def test_is_model_preparation_strategy(self):
        """Strategy inherits from ModelPreparationStrategy ABC."""
        self.assertIsInstance(_make_strategy(), ModelPreparationStrategy)

    def test_default_adapters_created_when_none_provided(self):
        """Both adapters fall back to their default implementations."""
        with patch(
            "executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation."
            "default_model_loader_adapter.DefaultModelLoaderAdapter"
        ) as mock_loader_cls, patch(
            "executorch.backends.qualcomm.genai_pipeline.datasets."
            "default_calibration_data_adapter.DefaultCalibrationDataAdapter"
        ) as mock_calib_cls:
            strategy = ExecuTorchModelPreparationStrategy()

            mock_loader_cls.assert_called_once()
            mock_calib_cls.assert_called_once()
            self.assertIs(strategy.adapter, mock_loader_cls.return_value)
            self.assertIs(
                strategy.calibration_data_adapter, mock_calib_cls.return_value
            )

    def test_custom_adapters_injected(self):
        """Both adapters are used when provided via the constructor."""
        loader = _make_mock_adapter()
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(loader, calibration)
        self.assertIs(strategy.adapter, loader)
        self.assertIs(strategy.calibration_data_adapter, calibration)

    def test_invoke_happy_path(self):
        """Full model preparation pipeline runs successfully end-to-end."""
        loader = _make_mock_adapter()
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(loader, calibration)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertIsInstance(result, ModelPreparationOutputConfig)
        self.assertIs(result.model_module, loader.load_model.return_value)
        self.assertIs(result.tokenizer, loader.load_tokenizer.return_value)
        self.assertEqual(
            result.calibration_data,
            calibration.generate_calibration_data.return_value,
        )

    def test_invoke_calls_loader_in_correct_order(self):
        """The loader is driven in order: load_model → load_tokenizer → example inputs."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)

        strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertEqual(
            [c[0] for c in loader.method_calls],
            ["load_model", "load_tokenizer", "get_example_inputs"],
        )

    def test_invoke_example_inputs_derived_from_the_loaded_model(self):
        """``example_inputs`` come from the model, not from the calibration data.

        The exported graph's positional signature (zero-initialized KV caches,
        fixed AR length) is a property of the model; the calibration dataset is
        in fact derived *from* it, so the dependency must not be inverted.
        """
        loader = _make_mock_adapter()
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(loader, calibration)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        loader.get_example_inputs.assert_called_once_with(
            model=loader.load_model.return_value,
            extra_options=None,
        )
        self.assertIs(result.example_inputs, loader.get_example_inputs.return_value)

    def test_invoke_example_input_options_forwarded(self):
        """example_input_options from extra_options reach get_example_inputs."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        example_opts = {"ar_len": 128}
        input_config = _make_valid_input_config(
            extra_options={"example_input_options": example_opts}
        )

        strategy.invoke(make_test_context(), input_config)

        loader.get_example_inputs.assert_called_once_with(
            model=loader.load_model.return_value,
            extra_options=example_opts,
        )

    def test_invoke_generates_calibration_data_from_dataset_adapter(self):
        """Calibration data comes from the dataset adapter, using the loaded tokenizer."""
        loader = _make_mock_adapter()
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(loader, calibration)

        strategy.invoke(make_test_context(), _make_valid_input_config())

        calibration.generate_calibration_data.assert_called_once_with(
            tokenizer=loader.load_tokenizer.return_value,
            num_samples=DEFAULT_NUM_SAMPLES,
            seq_length=DEFAULT_SEQ_LENGTH,
            extra_options=None,
        )

    def test_invoke_passes_model_name_to_load_model(self):
        """load_model receives model_name from input config."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)

        strategy.invoke(
            make_test_context(), _make_valid_input_config(model_name="llama3_2-1b")
        )

        loader.load_model.assert_called_once_with(
            model_name="llama3_2-1b",
            extra_options=None,
        )

    def test_invoke_passes_model_name_to_load_tokenizer(self):
        """load_tokenizer receives model_name from input config."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)

        strategy.invoke(
            make_test_context(), _make_valid_input_config(model_name="llama3_2-1b")
        )

        loader.load_tokenizer.assert_called_once_with(
            model_name="llama3_2-1b",
            extra_options=None,
        )

    def test_invoke_custom_calibration_params_from_extra_options(self):
        """Calibration params from extra_options override the defaults."""
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(calibration=calibration)
        calibration_options = {"dataset": "wikitext"}
        input_config = _make_valid_input_config(
            extra_options={
                "num_calibration_samples": 64,
                "calibration_seq_length": 256,
                "calibration_options": calibration_options,
            }
        )

        strategy.invoke(make_test_context(), input_config)

        _, kwargs = calibration.generate_calibration_data.call_args
        self.assertEqual(kwargs["num_samples"], 64)
        self.assertEqual(kwargs["seq_length"], 256)
        self.assertEqual(kwargs["extra_options"], calibration_options)

    def test_invoke_no_tokenizer_export_by_default(self):
        """export_tokenizer is NOT called when export_tokenizer option is absent."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        loader.export_tokenizer.assert_not_called()
        self.assertIsNone(result.runtime_tokenizer_path)

    def test_invoke_exports_tokenizer_when_requested(self):
        """export_tokenizer is called when export_tokenizer=True in extra_options."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        input_config = _make_valid_input_config(
            extra_options={"export_tokenizer": True}
        )
        context = make_test_context(artifact_dir="/my/artifacts")

        result = strategy.invoke(context, input_config)

        loader.export_tokenizer.assert_called_once_with(
            tokenizer=loader.load_tokenizer.return_value,
            output_dir=Path("/my/artifacts") / "tokenizer",
            extra_options=None,
        )
        # The adapter returns the tokenizer file itself, not its directory.
        self.assertEqual(
            result.runtime_tokenizer_path, Path("/tmp/tokenizer/tokenizer.json")
        )

    def test_invoke_extracts_chat_template_from_tokenizer(self):
        """A chat template on the tokenizer is carried into the output config."""
        loader = _make_mock_adapter()
        loader.load_tokenizer.return_value.chat_template = "{{ messages }}"
        strategy = _make_strategy(loader)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertEqual(result.chat_template, "{{ messages }}")

    def test_invoke_chat_template_from_extra_options_when_tokenizer_has_none(self):
        """extra_options supplies the chat template only as a fallback."""
        strategy = _make_strategy()
        input_config = _make_valid_input_config(
            extra_options={"chat_template": "fallback"}
        )

        result = strategy.invoke(make_test_context(), input_config)

        self.assertEqual(result.chat_template, "fallback")

    def test_invoke_chat_template_prefers_tokenizer_over_extra_options(self):
        """With both present the tokenizer wins: extra_options is only a fallback.

        Pins the precedence itself -- the single-source tests above would still
        pass if the two branches were swapped.
        """
        loader = _make_mock_adapter()
        loader.load_tokenizer.return_value.chat_template = "tokenizer_template"
        strategy = _make_strategy(loader)
        input_config = _make_valid_input_config(
            extra_options={"chat_template": "fallback"}
        )

        result = strategy.invoke(make_test_context(), input_config)

        self.assertEqual(result.chat_template, "tokenizer_template")

    def test_invoke_missing_model_name_raises_stage_error(self):
        """StageError raised when model_name is empty."""
        strategy = _make_strategy()

        with self.assertRaises(StageError) as cm:
            strategy.invoke(
                make_test_context(), _make_valid_input_config(model_name="")
            )
        self.assertIn("model_name", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "model_preparation")

    def test_invoke_missing_soc_model_raises_stage_error(self):
        """StageError raised when soc_model is empty."""
        strategy = _make_strategy()

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config(soc_model=""))
        self.assertIn("soc_model", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "model_preparation")

    def test_invoke_adapter_exception_wrapped_in_stage_error(self):
        """Exceptions from the adapter are wrapped in StageError."""
        loader = _make_mock_adapter()
        loader.load_model.side_effect = RuntimeError("model load failed")
        strategy = _make_strategy(loader)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertEqual(cm.exception.stage_name, "model_preparation")
        self.assertIsInstance(cm.exception.original_exception, RuntimeError)
        self.assertIn("model load failed", str(cm.exception))

    def test_invoke_calibration_adapter_exception_wrapped_in_stage_error(self):
        """Failures in the dataset adapter surface as a model_preparation StageError."""
        calibration = _make_mock_calibration_adapter()
        calibration.generate_calibration_data.side_effect = ValueError("no dataset")
        strategy = _make_strategy(calibration=calibration)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertEqual(cm.exception.stage_name, "model_preparation")
        self.assertIsInstance(cm.exception.original_exception, ValueError)

    def test_invoke_stage_error_not_double_wrapped(self):
        """StageError from adapter is re-raised directly."""
        loader = _make_mock_adapter()
        original_error = StageError(
            stage_name="model_preparation", message="inner error"
        )
        loader.load_model.side_effect = original_error
        strategy = _make_strategy(loader)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertIs(cm.exception, original_error)

    def test_invoke_model_options_forwarded(self):
        """model_options from extra_options are forwarded to load_model."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        model_opts = {"torch_dtype": "float16"}
        input_config = _make_valid_input_config(
            extra_options={"model_options": model_opts}
        )

        strategy.invoke(make_test_context(), input_config)

        loader.load_model.assert_called_once_with(
            model_name="test_model",
            extra_options=model_opts,
        )


if __name__ == "__main__":
    unittest.main()
