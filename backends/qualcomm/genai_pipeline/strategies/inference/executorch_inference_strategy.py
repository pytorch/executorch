# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ALL_ARTIFACT_KEYS,
    ARTIFACT_TEXT_DECODER,
    pte_file_name,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
    InferenceResult,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ForwardOutput,
    GenerateOutput,
    RunnerOutput,
    RunnerRequest,
)
from executorch.backends.qualcomm.genai_pipeline.inference.utils import (
    get_inference_control_args,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.device_runner_adapter import (
    DeviceRunnerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
        DeployedModel,
        EvaluationScore,
        Evaluator,
    )

logger = logging.getLogger(__name__)

_STAGE_NAME = "inference"


class ExecuTorchInferenceStrategy(InferenceStrategy):
    """ExecuTorch-based inference using QNN runtime on device.

    Delegates to a ``DeviceRunnerAdapter`` for all external API calls and to
    ``Evaluator``s for which inputs to feed and how to score the predictions,
    enabling dependency injection for testability. The strategy itself measures
    the runner's throughput and writes each method's result files.

    The inference flow:
    1. Resolve the PTE paths, from ``--pre-gen-pte`` when compilation was
       skipped, settle the runtime tokenizer, and validate the input
       configuration
    2. Push artifacts to device, once for every evaluation method
    3. For each evaluator, wrap the adapter as a ``DeployedModel`` and let the
       evaluator feed it inputs and score the predictions
    4. Build each method's result from the evaluator's score and the recorded
       predictions' throughput

    The strategy prepares no inputs and decodes no outputs: the evaluator
    decides what to feed the model and the adapter how to run it and decode the
    result (see ``Evaluator`` and ``runner_io``).

    Args:
        device_runner_adapter: Injectable adapter for device inference operations.
            Optional so the stage registry can build the strategy without
            arguments, but required by ``invoke``: device configuration has no
            default.
        evaluators: Evaluators for the requested evaluation methods. Optional
            for the same reason; ``invoke`` requires at least one.
    """

    def __init__(
        self,
        device_runner_adapter: Optional[DeviceRunnerAdapter] = None,
        evaluators: Sequence[Evaluator] = (),
    ) -> None:
        """Initialize the strategy with its adapter and evaluators.

        Args:
            device_runner_adapter: Adapter owning the runner lifecycle.
            evaluators: Evaluators to run sequentially.
        """
        self._adapter = device_runner_adapter
        self._evaluators = tuple(evaluators)

    @property
    def adapter(self) -> Optional[DeviceRunnerAdapter]:
        """The device runner adapter used by this strategy."""
        return self._adapter

    @property
    def evaluators(self) -> tuple[Evaluator, ...]:
        """The evaluators run by this strategy."""
        return self._evaluators

    def invoke(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
    ) -> InferenceOutputConfig:
        """Run every evaluator's inputs on the deployed artifacts.

        Args:
            context: Pipeline context with model identity, target SoC, and
                evaluation options.
            input_config: Inference configuration with compiled artifacts and
                runtime tokenizer information.

        Returns:
            Inference output containing results keyed by evaluation method.

        Raises:
            StageError: If artifacts are missing or inference fails.
        """
        input_config = self._resolve_pte_paths(context, input_config)
        input_config = self._resolve_runtime_tokenizer_path(context, input_config)
        self._validate_input(input_config)

        logger.info(
            "Starting inference for model '%s' on SoC=%s",
            context.model_name,
            input_config.soc_model,
        )

        try:
            # Step 1: Push artifacts to device once for every evaluation method.
            logger.debug("Pushing artifacts to device")
            self._adapter.push_artifacts(context=context, input_config=input_config)
            metadata = self._adapter.get_metadata()
            # Built once and shared by every evaluator.
            control_args = get_inference_control_args(context, input_config)

            results = {}
            output_dir = input_config.artifact_dir / "inference_output"
            for evaluator in self._evaluators:
                method_output_dir = output_dir / evaluator.method
                predictions: List[RunnerOutput] = []
                # Steps 2-3: the evaluator runs its inputs on device.
                logger.debug("Running %s on device", evaluator.method)
                score = evaluator.evaluate(
                    self._deployed_model(method_output_dir, predictions),
                    control_args,
                    input_config,
                    metadata,
                )
                results[evaluator.method] = self._build_result(
                    score, predictions, method_output_dir
                )

            logger.info("Inference completed successfully")
            return InferenceOutputConfig(results=results)

        except StageError:
            raise
        except Exception as e:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="Inference failed",
                original_exception=e,
            ) from e

    def _resolve_pte_paths(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
    ) -> InferenceInputConfig:
        """Resolve ``pte_paths`` from ``--pre-gen-pte`` when compilation was skipped.

        Each pre-generated PTE is named by ``pte_file_name``: one
        ``{model_name}_{artifact_key}.pte`` per component of a multi-graph
        model, or ``{model_name}.pte`` for a single-graph model, which serves
        as the text decoder. Artifacts absent from the directory are omitted,
        as in the compilation output, so the adapter decides which ones it
        requires.

        Args:
            context: Pipeline context supplying the model name and
                ``pre_gen_pte``.
            input_config: The inference input configuration.

        Returns:
            ``input_config`` itself when it already carries ``pte_paths``, else a
            copy with them resolved, shared by the adapter and the evaluators.

        Raises:
            StageError: If compilation was skipped without ``--pre-gen-pte``,
                or the directory holds none of the expected PTEs.
        """
        if input_config.pte_paths is not None:
            return input_config

        pre_gen_pte = context.extra_options.get("pre_gen_pte")
        if not pre_gen_pte:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="pte_paths is required for inference: compilation was "
                "skipped and --pre-gen-pte is not set",
            )

        pre_gen_dir = Path(pre_gen_pte)
        candidates = {
            key: pre_gen_dir / f"{pte_file_name(context.model_name, key)}.pte"
            for key in sorted(ALL_ARTIFACT_KEYS)
        }
        pte_paths = {key: path for key, path in candidates.items() if path.is_file()}
        single_graph_pte = pre_gen_dir / f"{pte_file_name(context.model_name)}.pte"
        if not pte_paths and single_graph_pte.is_file():
            pte_paths = {ARTIFACT_TEXT_DECODER: single_graph_pte}
        if not pte_paths:
            expected = [single_graph_pte, *candidates.values()]
            raise StageError(
                stage_name=_STAGE_NAME,
                message=f"--pre-gen-pte {pre_gen_dir} holds none of the expected "
                f"PTEs: {sorted(path.name for path in expected)}",
            )
        logger.info(
            "Compilation was skipped, so inference uses the pre-generated PTEs "
            "in %s: %s",
            pre_gen_dir,
            sorted(pte_paths),
        )
        return replace(input_config, pte_paths=pte_paths)

    def _resolve_runtime_tokenizer_path(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
    ) -> InferenceInputConfig:
        """Settle the runtime tokenizer every adapter and evaluator reads.

        ``extra_options['runtime_tokenizer_path']`` -- the user's explicit path,
        or the CLI's resolution when model preparation was skipped -- wins over
        the one model preparation reported. Whether a tokenizer is required is
        left to the adapter, since not every runner loads one.

        Args:
            context: Pipeline context supplying the optional override.
            input_config: The inference input configuration.

        Returns:
            ``input_config`` itself when there is no override, else a copy
            carrying it.
        """
        override = context.extra_options.get("runtime_tokenizer_path")
        if override is None:
            return input_config
        return replace(input_config, runtime_tokenizer_path=Path(override))

    def _deployed_model(
        self,
        output_dir: Path,
        predictions: List[RunnerOutput],
    ) -> "DeployedModel":
        """Wrap the adapter as the model an evaluator feeds inputs to.

        Each call pushes one input, executes it on the deployed runner, pulls
        its prediction into ``output_dir``, and records it in ``predictions``
        for the throughput. Calls are synchronous, so the prediction is decoded
        before the next input overwrites the runner's output files.

        Args:
            output_dir: The method's output directory for the runner files.
            predictions: Where every prediction is recorded, in order.

        Returns:
            The deployed model.
        """

        def model(model_input: RunnerRequest) -> RunnerOutput:
            self._adapter.push_inputs(model_input)
            self._adapter.execute(model_input)
            prediction = self._adapter.pull_results(model_input, output_dir)
            predictions.append(prediction)
            return prediction

        return model

    @staticmethod
    def _build_result(
        score: "EvaluationScore",
        predictions: Sequence[RunnerOutput],
        output_dir: Path,
    ) -> InferenceResult:
        """Combine a method's score with the runner's throughput and write both.

        Writes ``results.json`` when the score has metrics and always writes
        ``performance.json``, next to the runner files the predictions brought
        back.

        Args:
            score: The evaluator's verdict.
            predictions: Every prediction the method received, in order.
            output_dir: The method's output directory.

        Returns:
            The method's result: generations, metrics, throughput, and the
            runner and JSON files.
        """
        speeds = [
            prediction.tokens_per_sec
            for prediction in predictions
            if isinstance(prediction, (GenerateOutput, ForwardOutput))
        ]
        performance_metrics = {
            "num_requests": len(predictions),
            "mean_tokens_per_sec": sum(speeds) / len(speeds) if speeds else 0.0,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        # Each request overwrites the same runner files, so keep each path once.
        result_files = list(
            dict.fromkeys(
                file for prediction in predictions for file in prediction.files
            )
        )
        if score.metrics:
            results_path = output_dir / "results.json"
            results_path.write_text(json.dumps(score.metrics, indent=2, default=str))
            result_files.append(results_path)
        performance_path = output_dir / "performance.json"
        performance_path.write_text(json.dumps(performance_metrics, indent=2))
        result_files.append(performance_path)

        return InferenceResult(
            generations=score.generations or None,
            performance_metrics=performance_metrics,
            eval_results=score.metrics,
            result_files=result_files,
        )

    def _validate_input(self, input_config: InferenceInputConfig) -> None:
        """Validate required fields in the input configuration.

        Args:
            input_config: The inference input configuration.

        Raises:
            StageError: If required fields are missing or adapter is not set.
        """
        if self._adapter is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="device_runner_adapter is required for inference. "
                "Provide a DeviceRunnerAdapter via the constructor.",
            )
        if not self._evaluators:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="evaluators are required for inference. "
                "Provide at least one Evaluator via the constructor.",
            )
        if not input_config.pte_paths:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="pte_paths holds no PTE artifacts for inference",
            )
