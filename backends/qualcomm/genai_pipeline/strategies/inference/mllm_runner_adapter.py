# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``qnn_multimodal_runner`` adapter for multimodal LLMs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.inference.platforms import (
    RunnerPlatform,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_constants import (
    MODALITY_INPUT_FLAG_MAP,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    GenerateOutput,
    GenerateRequest,
)
from executorch.backends.qualcomm.genai_pipeline.inference.utils import (
    build_generate_args,
    build_runner_args,
    get_inference_control_args,
    get_pte_paths,
    get_runtime_tokenizer_path,
    preprocess_encoder_inputs,
    read_generate_output,
    RUNNER_INFERENCE_SPEED,
    RUNNER_OUTPUTS,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext


class MLLMRunnerAdapter:
    """Multimodal adapter running ``qnn_multimodal_runner`` on a ``RunnerPlatform``.

    Provides the decoder, token-embedding, and modality-encoder PTEs together
    with the preprocessed modality inputs, then runs one ``GenerateRequest`` per
    ``execute``. Where the runner runs is the platform's concern.

    Args:
        platform: Where the runner runs.
    """

    _RUNNER = "examples/qualcomm/oss_scripts/llama/qnn_multimodal_runner"
    # Generation cap on a platform that limits generation, as the legacy flow
    # uses for multimodal models on the x86 emulator.
    _LIMITED_GENERATE_SEQ_LEN = 320

    def __init__(self, platform: RunnerPlatform) -> None:
        self._platform = platform
        self._pte_paths: Optional[Dict[str, Path]] = None
        self._modality: Optional[str] = None
        self._runtime_tokenizer_path: Optional[Path] = None
        self._input_list_path: Optional[Path] = None
        self._runner_input_list: Optional[str] = None
        self._control_args: Optional[ControlArgs] = None

    def push_artifacts(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
        input_data: Optional[List[Any]] = None,
        extra_files: Optional[List[str]] = None,
    ) -> None:
        """Provide the MLLM artifacts and prepared modality inputs to the runner.

        Args:
            context: Pipeline context that supplies the model name and control
                args.
            input_config: Inference configuration that supplies the PTE paths,
                artifact directory, tokenizer, image/audio paths, and
                control-arg overrides.
            input_data: Unused. Source media is prepared by this adapter.
            extra_files: Optional additional files for the runner.
        """
        from executorch.backends.qualcomm.export_utils import generate_inputs
        from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
            get_modality,
            get_model_config,
            validate_modality_inputs,
        )

        self._input_list_path = None
        model_config = get_model_config(context.model_name)
        self._modality = get_modality(context.model_name)
        self._pte_paths = get_pte_paths(
            input_config.pte_paths,
            (ARTIFACT_TEXT_DECODER, ARTIFACT_TOK_EMBEDDING, self._modality),
        )
        runtime_tokenizer_path = get_runtime_tokenizer_path(input_config)

        self._control_args = get_inference_control_args(context, input_config)

        inference_options = input_config.inference_options
        image_paths = inference_options.get("image_path", [])
        audio_paths = inference_options.get("audio_path", [])
        validate_modality_inputs(context.model_name, image_paths, audio_paths)
        modality_inputs = preprocess_encoder_inputs(
            model_config,
            context.model_name,
            audio_paths,
            image_paths,
        )
        inputs = modality_inputs.get(self._modality)
        if not inputs:
            raise ValueError(f"No inputs produced for {self._modality}")

        host_artifact_dir = input_config.artifact_dir
        host_artifact_dir.mkdir(parents=True, exist_ok=True)
        input_list_file, input_files = generate_inputs(
            str(host_artifact_dir),
            input_list_filename=f"{self._modality}_input_list.txt",
            inputs=inputs,
            prefix_input_filename=self._modality,
        )
        if input_list_file is None:
            raise RuntimeError(f"Failed to create input list for {self._modality}")
        self._runtime_tokenizer_path = runtime_tokenizer_path
        self._input_list_path = Path(input_list_file)

        # Where a host-side runner writes its outputs, as in the legacy flow.
        runner_output_dir = input_config.artifact_dir / "outputs"
        runner_output_dir.mkdir(parents=True, exist_ok=True)
        self._platform.setup(
            self._RUNNER,
            [
                *self._pte_paths.values(),
                *(Path(file) for file in extra_files or []),
                runtime_tokenizer_path,
                self._input_list_path,
                *(Path(file) for file in input_files),
            ],
            runner_output_dir,
        )
        self._runner_input_list = self._platform.runner_input_list(
            self._input_list_path
        )

    def get_metadata(self) -> None:
        """Return ``None``: the multimodal flow serves no request that needs it."""
        return None

    def push_inputs(self, request: GenerateRequest) -> None:
        """Nothing to push: the modality inputs are deployed with the artifacts
        and the prompts travel as runner flags.

        Raises:
            TypeError: If the request is not a ``GenerateRequest``.
        """
        if not isinstance(request, GenerateRequest):
            raise TypeError(
                f"Unsupported request for {type(self).__name__}: {request!r}"
            )

    def execute(self, request: GenerateRequest) -> None:
        """Run the multimodal runner on the request's prompts.

        Args:
            request: The prompts to generate from.

        Raises:
            TypeError: If the request is not a ``GenerateRequest``.
        """
        if not isinstance(request, GenerateRequest):
            raise TypeError(
                f"Unsupported request for {type(self).__name__}: {request!r}"
            )
        if self._control_args is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )

        control_args = self._control_args
        platform = self._platform
        args = [
            *build_runner_args(control_args, platform, self._runtime_tokenizer_path),
            f"--decoder_path {platform.runner_path(self._pte_paths[ARTIFACT_TEXT_DECODER])}",
            f"--encoder_path {platform.runner_path(self._pte_paths[self._modality])}",
            f"--tok_embedding_path {platform.runner_path(self._pte_paths[ARTIFACT_TOK_EMBEDDING])}",
            f"--{MODALITY_INPUT_FLAG_MAP[self._modality]} "
            f"{self._runner_input_list}",
            *build_generate_args(
                control_args,
                request.prompts,
                platform,
                self._LIMITED_GENERATE_SEQ_LEN,
            ),
        ]
        platform.execute(args)

    def pull_results(
        self,
        request: GenerateRequest,
        output_dir: Path,
    ) -> GenerateOutput:
        """Collect the generated text and speed of the request ``execute`` just ran.

        Args:
            request: The request ``execute`` just ran.
            output_dir: Local directory to store the collected files.

        Returns:
            The generated text and decode speed.
        """
        if self._control_args is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs_path = output_dir / RUNNER_OUTPUTS
        inference_speed_path = output_dir / RUNNER_INFERENCE_SPEED
        self._platform.pull(RUNNER_OUTPUTS, outputs_path)
        self._platform.pull(RUNNER_INFERENCE_SPEED, inference_speed_path)
        return read_generate_output(outputs_path, inference_speed_path)
