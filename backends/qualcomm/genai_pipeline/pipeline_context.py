# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PipelineContext:
    """Immutable context holding all user-provided inputs for the pipeline.

    Attributes:
        model_name: Model identifier (e.g., "llama3_2-1b_instruct").
        soc_model: Target SoC (e.g., "SM8750").
        prompt: User prompt(s) for text generation.
        artifact_dir: Directory for storing compiled artifacts.
        extra_options: Additional options passed through to stages.
    """

    model_name: str
    soc_model: str
    prompt: List[str]
    artifact_dir: str = "./genai_artifacts"
    extra_options: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def builder() -> "PipelineContextBuilder":
        return PipelineContextBuilder()


class PipelineContextBuilder:
    """Builder for constructing a PipelineContext with validation.

    Example usage:

        context = (
            PipelineContext.builder()
            .with_model("llama3_2-1b_instruct")
            .with_soc("SM8750")
            .with_prompt("What is the capital of France?")
            .build()
        )
    """

    def __init__(self) -> None:
        self._model_name: Optional[str] = None
        self._soc_model: Optional[str] = None
        self._prompt: Optional[List[str]] = None
        self._artifact_dir: str = "./genai_artifacts"
        self._extra_options: Dict[str, Any] = {}

    def with_model(self, model_name: str) -> "PipelineContextBuilder":
        self._model_name = model_name
        return self

    def with_soc(self, soc_model: str) -> "PipelineContextBuilder":
        self._soc_model = soc_model
        return self

    def with_prompt(self, prompt: str | List[str]) -> "PipelineContextBuilder":
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompt cannot be empty or blank")
            self._prompt = [prompt]
        else:
            filtered = [p for p in prompt if isinstance(p, str) and p.strip()]
            if not filtered:
                raise ValueError(
                    "prompt list cannot be empty (all entries are blank or list is empty)"
                )
            self._prompt = filtered
        return self

    def with_artifact_dir(self, artifact_dir: str) -> "PipelineContextBuilder":
        self._artifact_dir = artifact_dir
        return self

    def with_extra_options(
        self, extra_options: Dict[str, Any]
    ) -> "PipelineContextBuilder":
        self._extra_options = dict(extra_options)
        return self

    def build(self) -> PipelineContext:
        """Build and validate the PipelineContext.

        Returns:
            A validated PipelineContext instance.

        Raises:
            ValueError: If required fields are missing.
        """
        missing = []
        if self._model_name is None:
            missing.append("model_name (use .with_model())")
        if self._soc_model is None:
            missing.append("soc_model (use .with_soc())")
        if not self._prompt:
            missing.append("prompt (use .with_prompt())")

        if missing:
            raise ValueError(
                f"Cannot build PipelineContext, missing required fields: "
                f"{', '.join(missing)}"
            )

        return PipelineContext(
            model_name=self._model_name,
            soc_model=self._soc_model,
            prompt=self._prompt,
            artifact_dir=self._artifact_dir,
            extra_options=self._extra_options,
        )
