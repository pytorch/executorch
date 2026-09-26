# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilationResult,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.executorch_compilation_strategy import (
    ExecuTorchCompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


def _make_mock_adapter():
    """Create a mock compiler adapter returning a valid CompilationResult."""
    adapter = MagicMock()
    adapter.compile_model.return_value = CompilationResult(
        artifact_paths=[Path("/tmp/test_model.pte")],
        etrecord=None,
    )
    return adapter


def _make_valid_input_config(**overrides):
    """Create a valid CompilationInputConfig with defaults."""
    defaults = {
        "soc_model": MagicMock(name="SM8750"),
        "backend_type": MagicMock(name="kHtpBackend"),
        "model": MagicMock(name="test_model"),
        "example_inputs": (MagicMock(name="example_input"),),
        "artifact_dir": Path("/tmp/artifacts"),
    }
    defaults.update(overrides)
    return CompilationInputConfig(**defaults)


class TestExecuTorchCompilationStrategy(unittest.TestCase):

    def test_is_compilation_strategy(self):
        """Strategy inherits from CompilationStrategy ABC."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        self.assertIsInstance(strategy, CompilationStrategy)

    def test_default_adapter_created_when_none_provided(self):
        """When no adapter is provided, DefaultCompilerAdapter is created."""
        with patch(
            "executorch.backends.qualcomm.genai_pipeline.strategies.compilation."
            "default_compiler_adapter.DefaultCompilerAdapter"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            strategy = ExecuTorchCompilationStrategy(compiler_adapter=None)
            mock_cls.assert_called_once()
            self.assertIsNotNone(strategy.adapter)

    def test_custom_adapter_injected(self):
        """Custom adapter is used when provided via constructor."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        self.assertIs(strategy.adapter, adapter)

    def test_invoke_happy_path(self):
        """Full compilation pipeline runs successfully end-to-end."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        context = make_test_context()
        input_config = _make_valid_input_config()

        result = strategy.invoke(context, input_config)

        self.assertIsInstance(result, CompilationOutputConfig)
        self.assertEqual(result.artifact_paths, [Path("/tmp/test_model.pte")])

    def test_invoke_passes_correct_args_to_adapter(self):
        """compile_model receives model, specs, artifact_dir, etc."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        model = MagicMock(name="model")
        soc = MagicMock(name="soc")
        backend = MagicMock(name="backend")
        specs = MagicMock(name="specs")
        example_inputs = (MagicMock(name="example_input"),)
        input_config = _make_valid_input_config(
            model=model,
            soc_model=soc,
            backend_type=backend,
            compile_specs=specs,
            example_inputs=example_inputs,
            artifact_dir=Path("/tmp/out"),
        )
        context = make_test_context()

        strategy.invoke(context, input_config)

        # example_inputs is an explicit parameter, not an extra_options key;
        # extra_options is None when context has no compilation-relevant keys.
        adapter.compile_model.assert_called_once_with(
            model=model,
            example_inputs=example_inputs,
            compile_specs=specs,
            artifact_dir=Path("/tmp/out"),
            file_name=context.model_name,
            soc_model=soc,
            backend_type=backend,
            extra_options=None,
        )

    def test_invoke_filters_extra_options_for_compilation(self):
        """Only compilation-relevant keys from context.extra_options are forwarded."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        skip_ops = {"aten.slice.Tensor"}
        context = make_test_context(
            extra_options={
                "generate_etrecord": True,
                "skip_node_op_set": skip_ops,
                # example_inputs travels on the input config, so a stray
                # context option of that name must not reach the adapter.
                "example_inputs": MagicMock(name="stale_inputs"),
                "unrelated_option": "should_not_pass",
            }
        )

        strategy.invoke(context, _make_valid_input_config())

        _, kwargs = adapter.compile_model.call_args
        self.assertEqual(
            kwargs["extra_options"],
            {"generate_etrecord": True, "skip_node_op_set": skip_ops},
        )

    def test_invoke_missing_example_inputs_raises_stage_error(self):
        """StageError raised when example_inputs is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        input_config = _make_valid_input_config(example_inputs=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("example_inputs", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "compilation")

    def test_invoke_missing_soc_model_raises_stage_error(self):
        """StageError raised when soc_model is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        input_config = _make_valid_input_config(soc_model=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("soc_model", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "compilation")

    def test_invoke_missing_backend_type_raises_stage_error(self):
        """StageError raised when backend_type is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        input_config = _make_valid_input_config(backend_type=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("backend_type", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "compilation")

    def test_invoke_uses_context_model_name_as_file_name(self):
        """file_name passed to adapter comes from context.model_name."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        context = make_test_context(model_name="my_model")

        strategy.invoke(context, _make_valid_input_config())

        _, kwargs = adapter.compile_model.call_args
        self.assertEqual(kwargs["file_name"], "my_model")

    def test_invoke_missing_model_raises_stage_error(self):
        """StageError raised when model is None."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)
        input_config = _make_valid_input_config(model=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("model", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "compilation")

    def test_invoke_adapter_exception_wrapped_in_stage_error(self):
        """Exceptions from the adapter are wrapped in StageError."""
        adapter = _make_mock_adapter()
        adapter.compile_model.side_effect = RuntimeError("compile failed")
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertEqual(cm.exception.stage_name, "compilation")
        self.assertIsInstance(cm.exception.original_exception, RuntimeError)
        self.assertIn("compile failed", str(cm.exception))

    def test_invoke_stage_error_not_double_wrapped(self):
        """StageError from adapter is re-raised directly."""
        adapter = _make_mock_adapter()
        original_error = StageError(stage_name="compilation", message="inner error")
        adapter.compile_model.side_effect = original_error
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertIs(cm.exception, original_error)

    def test_invoke_returns_etrecord_when_present(self):
        """etrecord from CompilationResult is forwarded to output config."""
        adapter = _make_mock_adapter()
        mock_etrecord = MagicMock(name="etrecord")
        adapter.compile_model.return_value = CompilationResult(
            artifact_paths=[Path("/tmp/test.pte")],
            etrecord=mock_etrecord,
        )
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertIs(result.etrecord, mock_etrecord)

    def test_invoke_multiple_artifacts(self):
        """Multiple artifact paths from adapter are forwarded correctly."""
        adapter = _make_mock_adapter()
        adapter.compile_model.return_value = CompilationResult(
            artifact_paths=[Path("/tmp/prefill.pte"), Path("/tmp/decode.pte")],
            etrecord=None,
        )
        strategy = ExecuTorchCompilationStrategy(compiler_adapter=adapter)

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertEqual(len(result.artifact_paths), 2)
        self.assertEqual(result.artifact_paths[0], Path("/tmp/prefill.pte"))
        self.assertEqual(result.artifact_paths[1], Path("/tmp/decode.pte"))


if __name__ == "__main__":
    unittest.main()
