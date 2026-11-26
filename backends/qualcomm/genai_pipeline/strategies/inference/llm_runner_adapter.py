# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``qnn_llama_runner`` adapter for text LLMs."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.inference.platforms import (
    RunnerPlatform,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_constants import (
    EVAL_MODE,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ForwardOutput,
    ForwardRequest,
    GenerateOutput,
    GenerateRequest,
    RunnerMetadata,
)
from executorch.backends.qualcomm.genai_pipeline.inference.utils import (
    build_generate_args,
    build_runner_args,
    get_pte_paths,
    get_runtime_tokenizer_path,
    post_process_logits,
    read_generate_output,
    retrieve_info_from_pte,
    RUNNER_INFERENCE_SPEED,
    RUNNER_INPUT_TOKENS,
    RUNNER_LOGITS,
    RUNNER_OUTPUTS,
)


class LLMRunnerAdapter:
    """Text-LLM adapter running ``qnn_llama_runner`` on a ``RunnerPlatform``.

    Provides the text-decoder PTE and runtime tokenizer once, then runs one
    ``GenerateRequest`` or ``ForwardRequest`` per ``execute``. Runner-level
    settings come from ``ControlArgs`` at deploy time; each request carries only
    its own inputs. Where the runner runs is the platform's concern.

    Args:
        platform: Where the runner runs.
    """

    RUNNER = "examples/qualcomm/oss_scripts/llama/qnn_llama_runner"
    # Generation cap on a platform that limits generation, as the legacy flow
    # uses for text models on the x86 emulator.
    _LIMITED_GENERATE_SEQ_LEN = 32

    def __init__(self, platform: RunnerPlatform) -> None:
        self._platform = platform
        self._pte_path: Optional[Path] = None
        self._runtime_tokenizer_path: Optional[Path] = None
        self._control_args: Optional[ControlArgs] = None
        self._metadata: Optional[RunnerMetadata] = None
        self._artifact_dir: Optional[Path] = None

    def push_artifacts(
        self,
        control_args: ControlArgs,
        input_config: InferenceInputConfig,
        input_data: Optional[List[Any]] = None,
        extra_files: Optional[List[str]] = None,
    ) -> None:
        """Provide the text-decoder artifact and runtime tokenizer to the runner.

        Args:
            control_args: Control args supplying the runner settings.
            input_config: Inference configuration that supplies the PTE paths,
                artifact directory, and tokenizer paths.
            input_data: Unused; requests carry their own inputs.
            extra_files: Optional additional files for the runner.
        """
        self._pte_path = get_pte_paths(input_config.pte_paths)[ARTIFACT_TEXT_DECODER]
        self._runtime_tokenizer_path = get_runtime_tokenizer_path(input_config)
        self._control_args = control_args
        self._metadata = retrieve_info_from_pte(self._pte_path)
        self._artifact_dir = input_config.artifact_dir
        # Where a host-side runner writes its outputs, as in the legacy flow.
        runner_output_dir = input_config.artifact_dir / "outputs"
        runner_output_dir.mkdir(parents=True, exist_ok=True)
        self._platform.setup(
            self.RUNNER,
            [
                self._pte_path,
                *(Path(file) for file in extra_files or []),
                self._runtime_tokenizer_path,
            ],
            runner_output_dir,
        )

    def get_metadata(self) -> RunnerMetadata:
        """Return the deployed text decoder's metadata.

        Raises:
            RuntimeError: If ``push_artifacts`` has not been called.
        """
        if self._metadata is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )
        return self._metadata

    def push_inputs(self, request: GenerateRequest | ForwardRequest) -> None:
        """Write a forward request's tokens and make them readable by the runner.

        A generation request needs nothing: its prompts travel as runner flags.

        Args:
            request: The prompts to generate from, or the tokens to forward.

        Raises:
            TypeError: If the request type is unsupported.
            ValueError: If a ``ForwardRequest`` does not fit the PTE context.
        """
        if self._metadata is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )
        if isinstance(request, GenerateRequest):
            return
        if not isinstance(request, ForwardRequest):
            raise TypeError(
                f"Unsupported request for {type(self).__name__}: {request!r}"
            )

        max_context_len = self._metadata.max_context_len
        if len(request.token_ids) >= max_context_len:
            raise ValueError(
                f"token length must be smaller than PTE max context length "
                f"({max_context_len}), got {len(request.token_ids)}"
            )
        token_path = self._token_path()
        np.asarray(request.token_ids, dtype=np.uint64).tofile(token_path)
        self._platform.add_file(token_path)

    def execute(self, request: GenerateRequest | ForwardRequest) -> None:
        """Run one generate or forward request whose inputs are pushed.

        Args:
            request: The prompts to generate from, or the tokens to forward.

        Raises:
            TypeError: If the request type is unsupported.
        """
        if self._control_args is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )

        control_args = self._control_args
        platform = self._platform
        args = [
            *build_runner_args(control_args, platform, self._runtime_tokenizer_path),
            f"--model_path {shlex.quote(platform.runner_path(self._pte_path))}",
        ]
        if isinstance(request, GenerateRequest):
            args.extend(
                build_generate_args(
                    control_args,
                    request.prompts,
                    platform,
                    self._LIMITED_GENERATE_SEQ_LEN,
                )
            )
        elif isinstance(request, ForwardRequest):
            args.extend(
                (
                    f"--eval_mode {EVAL_MODE['kv']}",
                    "--temperature 0",
                    f"--seq_len {request.seq_len or len(request.token_ids) + 1}",
                    f"--dump_logits_path {shlex.quote(f'{platform.output_dir()}/{RUNNER_LOGITS}')}",
                    f"--tokenized_prompt {shlex.quote(platform.runner_path(self._token_path()))}",
                )
            )
        else:
            raise TypeError(
                f"Unsupported request for {type(self).__name__}: {request!r}"
            )

        platform.execute(args)

    def _token_path(self) -> Path:
        # In the artifact directory, as the legacy flow and the multimodal
        # adapter's input lists are.
        return self._artifact_dir / RUNNER_INPUT_TOKENS

    def pull_results(
        self,
        request: GenerateRequest | ForwardRequest,
        output_dir: Path,
    ) -> GenerateOutput | ForwardOutput:
        """Collect the generated text or logits, and the speed, of ``request``.

        Args:
            request: The request ``execute`` just ran.
            output_dir: Local directory to store the collected files.

        Returns:
            A ``GenerateOutput`` for a ``GenerateRequest``, or a ``ForwardOutput``
            with the dequantized logits over the request's tokens for a
            ``ForwardRequest``.
        """
        if self._control_args is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        files = {
            filename: output_dir / filename
            for filename in (
                RUNNER_INFERENCE_SPEED,
                (
                    RUNNER_OUTPUTS
                    if isinstance(request, GenerateRequest)
                    else RUNNER_LOGITS
                ),
            )
        }
        for filename, host_path in files.items():
            self._platform.pull(filename, host_path)

        if isinstance(request, GenerateRequest):
            return read_generate_output(
                files[RUNNER_OUTPUTS], files[RUNNER_INFERENCE_SPEED]
            )
        return ForwardOutput(
            logits=post_process_logits(
                files[RUNNER_LOGITS], self._metadata, len(request.token_ids)
            ),
            tokens_per_sec=float(files[RUNNER_INFERENCE_SPEED].read_text()),
            files=list(files.values()),
        )
