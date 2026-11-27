# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ALL_ARTIFACT_KEYS,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
    pte_file_name,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    EvaluationScore,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ExecuteOutput,
    ForwardOutput,
    ForwardRequest,
    GenerateOutput,
    GenerateRequest,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.executorch_inference_strategy import (
    ExecuTorchInferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
    TEST_MODEL_NAME,
    TEST_PTE_PATH,
    TEST_SOC_CHIPSET,
)

_CONTROL_ARGS_PATCH = (
    "executorch.backends.qualcomm.genai_pipeline.control_args."
    "ControlArgs.from_pipeline_context"
)


class _RecordingEvaluator:
    """Evaluator that feeds fixed requests and reports a fixed score."""

    def __init__(self, method, requests=(), score=None):
        self.method = method
        self.requests = list(requests)
        self.score = score or EvaluationScore()
        self.predictions = []
        self.calls = []

    def evaluate(self, model, control_args, input_config, metadata):
        """Feed every request to the model and return the fixed score."""
        self.calls.append((control_args, input_config, metadata))
        self.predictions = [model(request) for request in self.requests]
        return self.score


def _make_mock_adapter(prediction=None):
    """Create a mock adapter whose pull_results returns ``prediction``."""
    adapter = MagicMock()
    adapter.get_metadata.return_value = MagicMock(name="metadata")
    adapter.pull_results.return_value = prediction or GenerateOutput(
        texts=["Hello, world!"], tokens_per_sec=10.0
    )
    return adapter


def _make_valid_input_config(artifact_dir, **overrides):
    """Create a valid InferenceInputConfig rooted at ``artifact_dir``."""
    defaults = {
        "soc_model": TEST_SOC_CHIPSET,
        "artifact_dir": Path(artifact_dir),
        "pte_paths": {ARTIFACT_TEXT_DECODER: TEST_PTE_PATH},
    }
    defaults.update(overrides)
    return InferenceInputConfig(**defaults)


def _write_ptes(directory, names):
    """Create empty PTE files named ``names`` under ``directory``."""
    for name in names:
        (Path(directory) / name).write_bytes(b"")


class _TempDirTestCase(unittest.TestCase):
    """Test case providing a temporary artifact directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestExecuTorchInferenceStrategyConstruction(unittest.TestCase):

    def test_is_inference_strategy(self):
        """The strategy implements the InferenceStrategy interface."""
        self.assertIsInstance(ExecuTorchInferenceStrategy(), InferenceStrategy)

    def test_custom_adapter_and_evaluators_injected(self):
        """The injected adapter and evaluators are exposed unchanged."""
        adapter = _make_mock_adapter()
        evaluator = _RecordingEvaluator("prompt_eval")
        strategy = ExecuTorchInferenceStrategy(
            device_runner_adapter=adapter, evaluators=[evaluator]
        )

        self.assertIs(strategy.adapter, adapter)
        self.assertEqual(strategy.evaluators, (evaluator,))

    def test_defaults_to_no_adapter_and_no_evaluators(self):
        """A registry-built strategy has neither adapter nor evaluators."""
        strategy = ExecuTorchInferenceStrategy()

        self.assertIsNone(strategy.adapter)
        self.assertEqual(strategy.evaluators, ())


@patch(_CONTROL_ARGS_PATCH)
class TestExecuTorchInferenceStrategyInvoke(_TempDirTestCase):

    def _invoke(self, adapter, evaluators, context=None, **config_overrides):
        """Invoke the strategy over a valid input config in the temp directory."""
        strategy = ExecuTorchInferenceStrategy(
            device_runner_adapter=adapter, evaluators=evaluators
        )
        return strategy.invoke(
            context or make_test_context(),
            _make_valid_input_config(self.tmp_dir, **config_overrides),
        )

    def test_invoke_returns_results_keyed_by_evaluator_method(self, _):
        """Each evaluator's result is keyed by its method."""
        evaluators = [_RecordingEvaluator("prompt_eval"), _RecordingEvaluator("x")]

        result = self._invoke(_make_mock_adapter(), evaluators)

        self.assertIsInstance(result, InferenceOutputConfig)
        self.assertEqual(sorted(result.results), ["prompt_eval", "x"])

    def test_invoke_pushes_artifacts_once_for_all_evaluators(self, _):
        """Artifacts are pushed once, however many evaluators run."""
        adapter = _make_mock_adapter()

        self._invoke(adapter, [_RecordingEvaluator("a"), _RecordingEvaluator("b")])

        adapter.push_artifacts.assert_called_once()

    def test_invoke_passes_shared_control_args_and_metadata(self, control_args):
        """The adapter and every evaluator share one ControlArgs, built once."""
        adapter = _make_mock_adapter()
        evaluators = [_RecordingEvaluator("a"), _RecordingEvaluator("b")]

        self._invoke(adapter, evaluators)

        control_args.assert_called_once()
        self.assertIs(
            adapter.push_artifacts.call_args.kwargs["control_args"],
            control_args.return_value,
        )
        for evaluator in evaluators:
            args, input_config, metadata = evaluator.calls[0]
            self.assertIs(args, control_args.return_value)
            self.assertEqual(input_config.artifact_dir, self.tmp_dir)
            self.assertIs(metadata, adapter.get_metadata.return_value)

    def test_deployed_model_pushes_executes_and_pulls_each_request(self, _):
        """Each request runs push_inputs, execute, then pull_results in order."""
        adapter = _make_mock_adapter()
        request = GenerateRequest(prompts=["hi"])

        self._invoke(adapter, [_RecordingEvaluator("m", requests=[request])])

        calls = [
            name
            for name, *_ in adapter.method_calls
            if name in {"push_inputs", "execute", "pull_results"}
        ]
        self.assertEqual(calls, ["push_inputs", "execute", "pull_results"])
        adapter.push_inputs.assert_called_once_with(request)
        adapter.execute.assert_called_once_with(request)
        adapter.pull_results.assert_called_once_with(
            request, self.tmp_dir / "inference_output" / "m"
        )

    def test_deployed_model_returns_the_pulled_prediction(self, _):
        """The evaluator receives the prediction pull_results returns."""
        adapter = _make_mock_adapter()
        evaluator = _RecordingEvaluator("m", requests=[GenerateRequest(["hi"])])

        self._invoke(adapter, [evaluator])

        self.assertEqual(evaluator.predictions, [adapter.pull_results.return_value])

    def test_invoke_reports_generations_and_metrics_from_score(self, _):
        """The score's generations and metrics become the method's result."""
        score = EvaluationScore(generations=["out"], metrics={"sqnr": 20.0})

        result = self._invoke(
            _make_mock_adapter(), [_RecordingEvaluator("m", score=score)]
        )

        method_result = result.results["m"]
        self.assertEqual(method_result.generations, ["out"])
        self.assertEqual(method_result.eval_results, {"sqnr": 20.0})

    def test_invoke_reports_no_generations_when_score_has_none(self, _):
        """An empty generation list is reported as None."""
        result = self._invoke(_make_mock_adapter(), [_RecordingEvaluator("m")])

        self.assertIsNone(result.results["m"].generations)

    def test_invoke_reports_mean_throughput_over_predictions(self, _):
        """Throughput averages tokens_per_sec over generate and forward predictions."""
        adapter = _make_mock_adapter()
        adapter.pull_results.side_effect = [
            GenerateOutput(texts=["a"], tokens_per_sec=10.0),
            ForwardOutput(logits=MagicMock(), tokens_per_sec=30.0),
        ]
        requests = [GenerateRequest(["a"]), GenerateRequest(["b"])]

        result = self._invoke(adapter, [_RecordingEvaluator("m", requests=requests)])

        self.assertEqual(
            result.results["m"].performance_metrics,
            {"num_requests": 2, "mean_tokens_per_sec": 20.0},
        )

    def test_invoke_frees_each_prediction_the_evaluator_drops(self, _):
        """Only throughput and files are kept, so lm_eval can free each logits tensor."""
        logits_refs = []

        def pull_results(*_):
            logits = torch.zeros(1, 2, 4)
            logits_refs.append(weakref.ref(logits))
            return ForwardOutput(logits=logits, tokens_per_sec=5.0)

        adapter = _make_mock_adapter()
        adapter.pull_results.side_effect = pull_results

        class _DroppingEvaluator:
            method = "tasks_eval"

            def evaluate(self, model, *_):
                model(ForwardRequest(token_ids=[1]))
                self.freed = logits_refs[0]() is None
                return EvaluationScore()

        evaluator = _DroppingEvaluator()
        self._invoke(adapter, [evaluator])

        self.assertTrue(evaluator.freed)

    def test_invoke_reports_zero_throughput_without_timed_predictions(self, _):
        """Predictions without tokens_per_sec yield a zero mean throughput."""
        adapter = _make_mock_adapter(prediction=ExecuteOutput())
        evaluator = _RecordingEvaluator("m", requests=[MagicMock()])

        result = self._invoke(adapter, [evaluator])

        self.assertEqual(
            result.results["m"].performance_metrics,
            {"num_requests": 1, "mean_tokens_per_sec": 0.0},
        )

    def test_invoke_writes_results_and_performance_json(self, _):
        """results.json holds the metrics and performance.json the throughput."""
        score = EvaluationScore(metrics={"wiki_ppl": 12.5})

        result = self._invoke(
            _make_mock_adapter(), [_RecordingEvaluator("m", score=score)]
        )

        method_dir = self.tmp_dir / "inference_output" / "m"
        self.assertEqual(
            json.loads((method_dir / "results.json").read_text()),
            {"wiki_ppl": 12.5},
        )
        self.assertEqual(
            json.loads((method_dir / "performance.json").read_text()),
            {"num_requests": 0, "mean_tokens_per_sec": 0.0},
        )
        self.assertEqual(
            result.results["m"].result_files,
            [method_dir / "results.json", method_dir / "performance.json"],
        )

    def test_invoke_skips_results_json_without_metrics(self, _):
        """No results.json is written when the score has no metrics."""
        result = self._invoke(_make_mock_adapter(), [_RecordingEvaluator("m")])

        method_dir = self.tmp_dir / "inference_output" / "m"
        self.assertFalse((method_dir / "results.json").exists())
        self.assertEqual(
            result.results["m"].result_files, [method_dir / "performance.json"]
        )

    def test_invoke_lists_each_pulled_runner_file_once(self, _):
        """Runner files repeated across predictions are listed once."""
        runner_file = self.tmp_dir / "outputs.txt"
        adapter = _make_mock_adapter(
            prediction=GenerateOutput(
                texts=["a"], tokens_per_sec=1.0, files=[runner_file]
            )
        )
        requests = [GenerateRequest(["a"]), GenerateRequest(["b"])]

        result = self._invoke(adapter, [_RecordingEvaluator("m", requests=requests)])

        files = result.results["m"].result_files
        self.assertEqual(files.count(runner_file), 1)
        self.assertEqual(files[0], runner_file)

    def test_invoke_no_adapter_raises_stage_error(self, _):
        """invoke requires an adapter."""
        with self.assertRaises(StageError) as ctx:
            self._invoke(None, [_RecordingEvaluator("m")])

        self.assertIn("device_runner_adapter is required", str(ctx.exception))

    def test_invoke_no_evaluators_raises_stage_error(self, _):
        """invoke requires at least one evaluator."""
        with self.assertRaises(StageError) as ctx:
            self._invoke(_make_mock_adapter(), [])

        self.assertIn("evaluators are required", str(ctx.exception))

    def test_invoke_empty_pte_paths_raises_stage_error(self, _):
        """An empty compilation output leaves nothing to run."""
        with self.assertRaises(StageError) as ctx:
            self._invoke(_make_mock_adapter(), [_RecordingEvaluator("m")], pte_paths={})

        self.assertIn("pte_paths holds no PTE artifacts", str(ctx.exception))

    def test_invoke_adapter_exception_wrapped_in_stage_error(self, _):
        """Adapter failures surface as a StageError carrying the original."""
        adapter = _make_mock_adapter()
        adapter.push_artifacts.side_effect = RuntimeError("adb failed")

        with self.assertRaises(StageError) as ctx:
            self._invoke(adapter, [_RecordingEvaluator("m")])

        self.assertIn("Inference failed", str(ctx.exception))
        self.assertIsInstance(ctx.exception.original_exception, RuntimeError)

    def test_invoke_stage_error_not_double_wrapped(self, _):
        """A StageError raised inside invoke propagates unchanged."""
        adapter = _make_mock_adapter()
        original = StageError(stage_name="inference", message="inner")
        adapter.push_artifacts.side_effect = original

        with self.assertRaises(StageError) as ctx:
            self._invoke(adapter, [_RecordingEvaluator("m")])

        self.assertIs(ctx.exception, original)


class TestResolvePtePaths(_TempDirTestCase):

    def _resolve(self, pre_gen_pte, **config_overrides):
        """Resolve pte_paths with compilation skipped."""
        config_overrides.setdefault("pte_paths", None)
        return ExecuTorchInferenceStrategy()._resolve_pte_paths(
            make_test_context(),
            ControlArgs(pre_gen_pte=pre_gen_pte),
            _make_valid_input_config(self.tmp_dir, **config_overrides),
        )

    def test_keeps_input_config_with_compiled_pte_paths(self):
        """Compiled pte_paths are used as is, even with --pre-gen-pte."""
        input_config = _make_valid_input_config(self.tmp_dir)

        resolved = ExecuTorchInferenceStrategy()._resolve_pte_paths(
            make_test_context(),
            ControlArgs(pre_gen_pte=str(self.tmp_dir)),
            input_config,
        )

        self.assertIs(resolved, input_config)

    def test_resolves_component_ptes_by_pte_file_name(self):
        """Component PTEs are found under the names compilation writes."""
        keys = [ARTIFACT_TEXT_DECODER, ARTIFACT_TOK_EMBEDDING, ARTIFACT_VISION_ENCODER]
        _write_ptes(
            self.tmp_dir, [f"{pte_file_name(TEST_MODEL_NAME, key)}.pte" for key in keys]
        )

        resolved = self._resolve(str(self.tmp_dir))

        self.assertEqual(
            resolved.pte_paths,
            {
                key: self.tmp_dir / f"{pte_file_name(TEST_MODEL_NAME, key)}.pte"
                for key in keys
            },
        )

    def test_omits_absent_components(self):
        """Only the components present in the directory are resolved."""
        _write_ptes(
            self.tmp_dir,
            [f"{pte_file_name(TEST_MODEL_NAME, ARTIFACT_TEXT_DECODER)}.pte"],
        )

        resolved = self._resolve(str(self.tmp_dir))

        self.assertEqual(list(resolved.pte_paths), [ARTIFACT_TEXT_DECODER])

    def test_resolves_single_graph_pte_as_text_decoder(self):
        """A single-graph {model}.pte is used as the text decoder."""
        _write_ptes(self.tmp_dir, [f"{pte_file_name(TEST_MODEL_NAME)}.pte"])

        resolved = self._resolve(str(self.tmp_dir))

        self.assertEqual(
            resolved.pte_paths,
            {ARTIFACT_TEXT_DECODER: self.tmp_dir / f"{TEST_MODEL_NAME}.pte"},
        )

    def test_component_ptes_win_over_single_graph_pte(self):
        """With both layouts present, the component PTEs are used."""
        decoder = f"{pte_file_name(TEST_MODEL_NAME, ARTIFACT_TEXT_DECODER)}.pte"
        _write_ptes(self.tmp_dir, [decoder, f"{TEST_MODEL_NAME}.pte"])

        resolved = self._resolve(str(self.tmp_dir))

        self.assertEqual(
            resolved.pte_paths, {ARTIFACT_TEXT_DECODER: self.tmp_dir / decoder}
        )

    def test_empty_directory_raises_stage_error_listing_expected_ptes(self):
        """The error names every PTE file it looked for."""
        with self.assertRaises(StageError) as ctx:
            self._resolve(str(self.tmp_dir))

        message = str(ctx.exception)
        self.assertIn(f"{TEST_MODEL_NAME}.pte", message)
        for key in ALL_ARTIFACT_KEYS:
            self.assertIn(f"{pte_file_name(TEST_MODEL_NAME, key)}.pte", message)

    def test_skipped_compilation_without_pre_gen_pte_raises_stage_error(self):
        """Without compiled PTEs, --pre-gen-pte is required."""
        with self.assertRaises(StageError) as ctx:
            self._resolve(None)

        self.assertIn("--pre-gen-pte is not set", str(ctx.exception))

    def test_invoke_runs_on_pre_generated_ptes(self):
        """invoke deploys the PTEs it resolved from --pre-gen-pte."""
        decoder = f"{pte_file_name(TEST_MODEL_NAME, ARTIFACT_TEXT_DECODER)}.pte"
        _write_ptes(self.tmp_dir, [decoder])
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(
            device_runner_adapter=adapter, evaluators=[_RecordingEvaluator("m")]
        )

        strategy.invoke(
            make_test_context(extra_options={"pre_gen_pte": str(self.tmp_dir)}),
            _make_valid_input_config(self.tmp_dir, pte_paths=None),
        )

        pushed = adapter.push_artifacts.call_args.kwargs["input_config"]
        self.assertEqual(
            pushed.pte_paths, {ARTIFACT_TEXT_DECODER: self.tmp_dir / decoder}
        )


class TestResolveRuntimeTokenizerPath(_TempDirTestCase):

    def _resolve(self, extra_options, runtime_tokenizer_path):
        """Resolve the runtime tokenizer of a config carrying model preparation's path."""
        return ExecuTorchInferenceStrategy()._resolve_runtime_tokenizer_path(
            make_test_context(extra_options=extra_options),
            _make_valid_input_config(
                self.tmp_dir, runtime_tokenizer_path=runtime_tokenizer_path
            ),
        )

    def test_override_wins_over_model_preparation(self):
        """extra_options['runtime_tokenizer_path'] replaces the prepared path."""
        resolved = self._resolve(
            {"runtime_tokenizer_path": "/tmp/override.bin"},
            Path("/tmp/prep/tokenizer.json"),
        )

        self.assertEqual(resolved.runtime_tokenizer_path, Path("/tmp/override.bin"))

    def test_keeps_model_preparation_path_without_override(self):
        """Without an override the input config is returned unchanged."""
        input_config = _make_valid_input_config(
            self.tmp_dir, runtime_tokenizer_path=Path("/tmp/prep/tokenizer.json")
        )

        resolved = ExecuTorchInferenceStrategy()._resolve_runtime_tokenizer_path(
            make_test_context(), input_config
        )

        self.assertIs(resolved, input_config)

    def test_does_not_validate_the_path(self):
        """Neither a missing tokenizer nor a nonexistent file is an error here."""
        self.assertIsNone(self._resolve({}, None).runtime_tokenizer_path)
        self.assertEqual(
            self._resolve(
                {"runtime_tokenizer_path": "/nonexistent/tokenizer.bin"}, None
            ).runtime_tokenizer_path,
            Path("/nonexistent/tokenizer.bin"),
        )

    def test_invoke_hands_the_override_to_the_adapter(self):
        """invoke settles the tokenizer before pushing artifacts."""
        adapter = _make_mock_adapter()
        strategy = ExecuTorchInferenceStrategy(
            device_runner_adapter=adapter, evaluators=[_RecordingEvaluator("m")]
        )

        with patch(_CONTROL_ARGS_PATCH):
            strategy.invoke(
                make_test_context(
                    extra_options={"runtime_tokenizer_path": "/tmp/override.bin"}
                ),
                _make_valid_input_config(
                    self.tmp_dir, runtime_tokenizer_path=Path("/tmp/prep.json")
                ),
            )

        pushed = adapter.push_artifacts.call_args.kwargs["input_config"]
        self.assertEqual(pushed.runtime_tokenizer_path, Path("/tmp/override.bin"))


if __name__ == "__main__":
    unittest.main()
