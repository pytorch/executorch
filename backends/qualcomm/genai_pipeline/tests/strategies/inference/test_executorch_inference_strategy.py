# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.device_runner_adapter import (
    InferenceResult,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.executorch_inference_strategy import (
    ExecuTorchInferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


def _make_mock_adapter():
    """Create a mock device runner adapter with sensible defaults."""
    adapter = MagicMock()
    adapter.push_artifacts.return_value = None
    adapter.execute.return_value = InferenceResult(
        output_data=["Hello, world!"],
        performance_metrics={"tokens_per_sec": 42.0},
        etdump=None,
    )
    adapter.pull_results.return_value = [Path("/tmp/output/result.bin")]
    return adapter


def _make_valid_input_config(**overrides):
    """Create a valid InferenceInputConfig with defaults."""
    defaults = {
        "soc_model": MagicMock(name="SM8750"),
        "artifact_paths": [Path("/tmp/test.pte")],
    }
    defaults.update(overrides)
    return InferenceInputConfig(**defaults)


class TestExecuTorchInferenceStrategy(unittest.TestCase):

    def test_is_inference_strategy(self):
        """Strategy inherits from InferenceStrategy ABC."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        self.assertIsInstance(strategy, InferenceStrategy)

    def test_custom_adapter_injected(self):
        """Custom adapter is used when provided via constructor."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        self.assertIs(strategy.adapter, adapter)

    def test_no_adapter_defaults_to_none(self):
        """When no adapter is provided, adapter property is None."""
        strategy = ExecuTorchInferenceStrategy()
        self.assertIsNone(strategy.adapter)

    def test_invoke_happy_path(self):
        """Full inference pipeline runs successfully end-to-end."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        context = make_test_context()
        input_config = _make_valid_input_config()

        result = strategy.invoke(context, input_config)

        self.assertIsInstance(result, InferenceOutputConfig)
        self.assertEqual(result.inference_results, ["Hello, world!"])
        self.assertEqual(result.performance_metrics["tokens_per_sec"], 42.0)

    def test_invoke_calls_adapter_in_correct_order(self):
        """Adapter methods are called in order: push → execute → pull."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)

        strategy.invoke(make_test_context(), _make_valid_input_config())

        expected_calls = ["push_artifacts", "execute", "pull_results"]
        actual_method_calls = [c[0] for c in adapter.method_calls]
        self.assertEqual(actual_method_calls, expected_calls)

    def test_invoke_passes_artifact_paths_to_push(self):
        """push_artifacts receives the artifact paths from input config."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        artifact_paths = [Path("/tmp/a.pte"), Path("/tmp/b.pte")]
        input_config = _make_valid_input_config(artifact_paths=artifact_paths)

        strategy.invoke(make_test_context(), input_config)

        adapter.push_artifacts.assert_called_once_with(
            artifact_paths=artifact_paths,
        )

    def test_invoke_passes_inference_options_to_execute(self):
        """execute receives inference_options from input config."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        opts = {"method_index": 1, "iteration": 5}
        input_config = _make_valid_input_config(inference_options=opts)

        strategy.invoke(make_test_context(), input_config)

        adapter.execute.assert_called_once_with(inference_options=opts)

    def test_invoke_pull_results_uses_context_artifact_dir(self):
        """pull_results output_dir is based on context.artifact_dir."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        context = make_test_context(artifact_dir="/custom/dir")

        strategy.invoke(context, _make_valid_input_config())

        adapter.pull_results.assert_called_once_with(
            output_dir=Path("/custom/dir") / "inference_output",
        )

    def test_invoke_populates_performance_metrics(self):
        """Performance metrics from adapter result are in output config."""
        adapter = _make_mock_adapter()
        adapter.execute.return_value = InferenceResult(
            output_data=None,
            performance_metrics={"ttft_ms": 100, "tokens_per_sec": 50},
        )
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertEqual(result.performance_metrics["ttft_ms"], 100)
        self.assertEqual(result.performance_metrics["tokens_per_sec"], 50)

    def test_invoke_populates_etdump(self):
        """etdump from adapter result is forwarded to output config."""
        adapter = _make_mock_adapter()
        mock_etdump = MagicMock(name="etdump")
        adapter.execute.return_value = InferenceResult(
            output_data=None,
            performance_metrics={},
            etdump=mock_etdump,
        )
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertIs(result.etdump, mock_etdump)

    def test_invoke_no_adapter_raises_stage_error(self):
        """StageError raised when no adapter is configured."""
        strategy = ExecuTorchInferenceStrategy()
        input_config = _make_valid_input_config()

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("device_runner_adapter", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "inference")

    def test_invoke_missing_artifacts_raises_stage_error(self):
        """StageError raised when artifact_paths is empty."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        input_config = _make_valid_input_config(artifact_paths=[])

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("artifact_paths", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "inference")

    def test_invoke_none_artifacts_raises_stage_error(self):
        """StageError raised when artifact_paths is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)
        input_config = _make_valid_input_config(artifact_paths=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("artifact_paths", str(cm.exception))

    def test_invoke_adapter_exception_wrapped_in_stage_error(self):
        """Exceptions from the adapter are wrapped in StageError."""
        adapter = _make_mock_adapter()
        adapter.push_artifacts.side_effect = RuntimeError("push failed")
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertEqual(cm.exception.stage_name, "inference")
        self.assertIsInstance(cm.exception.original_exception, RuntimeError)
        self.assertIn("push failed", str(cm.exception))

    def test_invoke_stage_error_not_double_wrapped(self):
        """StageError from adapter is re-raised directly."""
        adapter = _make_mock_adapter()
        original_error = StageError(stage_name="inference", message="inner error")
        adapter.execute.side_effect = original_error
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertIs(cm.exception, original_error)

    def test_invoke_uses_pulled_files_when_output_data_is_none(self):
        """When execute returns output_data=None, pulled file paths are used."""
        adapter = _make_mock_adapter()
        adapter.execute.return_value = InferenceResult(
            output_data=None,
            performance_metrics={"tokens_per_sec": 30.0},
            etdump=None,
        )
        adapter.pull_results.return_value = [
            Path("/tmp/out/result_0.bin"),
            Path("/tmp/out/result_1.bin"),
        ]
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertEqual(
            result.inference_results,
            ["/tmp/out/result_0.bin", "/tmp/out/result_1.bin"],
        )

    def test_invoke_prefers_output_data_over_pulled_files(self):
        """When execute returns non-None output_data, pulled files are not used."""
        adapter = _make_mock_adapter()
        adapter.execute.return_value = InferenceResult(
            output_data=["generated text"],
            performance_metrics={},
            etdump=None,
        )
        adapter.pull_results.return_value = [Path("/tmp/out/result.bin")]
        strategy = ExecuTorchInferenceStrategy(device_runner_adapter=adapter)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertEqual(result.inference_results, ["generated text"])


if __name__ == "__main__":
    unittest.main()
