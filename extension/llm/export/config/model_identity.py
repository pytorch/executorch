# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ModelFamily(str, Enum):
    """Model architecture used to select the eager implementation."""

    llama2 = "llama2"
    llama3 = "llama3"
    llama3_1 = "llama3_1"
    llama3_2 = "llama3_2"
    llama3_2_vision = "llama3_2_vision"
    qwen2_5 = "qwen2_5"
    qwen3 = "qwen3"
    qwen3_5 = "qwen3_5"
    phi_4_mini = "phi_4_mini"
    smollm2 = "smollm2"
    lfm2 = "lfm2"
    lfm2_5 = "lfm2_5"
    spark_x2_5 = "spark_x2_5"


class ModelType(str, Enum):
    """Legacy model_class values."""

    stories110m = "stories110m"
    llama2 = "llama2"
    llama3 = "llama3"
    llama3_1 = "llama3_1"
    llama3_2 = "llama3_2"
    llama3_2_vision = "llama3_2_vision"
    static_llama = "static_llama"
    qwen2_5_0_5b = "qwen2_5_0_5b"
    qwen2_5_1_5b = "qwen2_5_1_5b"
    qwen2_5_coder_32b = "qwen2_5_coder_32b"
    qwen3_0_6b = "qwen3_0_6b"
    qwen3_1_7b = "qwen3_1_7b"
    qwen3_4b = "qwen3_4b"
    qwen3_5_0_8b = "qwen3_5_0_8b"
    qwen3_5_2b = "qwen3_5_2b"
    qwen3_5_4b = "qwen3_5_4b"
    phi_4_mini = "phi_4_mini"
    smollm2 = "smollm2"
    smollm2_360m = "smollm2_360m"
    lfm2_350m = "lfm2_350m"
    lfm2_700m = "lfm2_700m"
    lfm2_1_2b = "lfm2_1_2b"
    lfm2_5_350m = "lfm2_5_350m"
    lfm2_5_1_2b = "lfm2_5_1_2b"
    spark_x2_5_1_7b = "spark_x2_5_1_7b"
    spark_x2_5_4b = "spark_x2_5_4b"
    spark_x2_6_1_7b = "spark_x2_6_1_7b"
    spark_x2_6_4b = "spark_x2_6_4b"


@dataclass(frozen=True)
class ModelSpec:
    family: ModelFamily
    hf_repo_id: str


@dataclass(frozen=True)
class ModelIdentity:
    """Resolved architecture and optional registered model id."""

    family: ModelFamily
    model_id: Optional[str]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "qwen2_5_0_5b": ModelSpec(ModelFamily.qwen2_5, "Qwen/Qwen2.5-0.5B"),
    "qwen2_5_1_5b": ModelSpec(ModelFamily.qwen2_5, "Qwen/Qwen2.5-1.5B"),
    "qwen2_5_coder_32b": ModelSpec(
        ModelFamily.qwen2_5, "Qwen/Qwen2.5-Coder-32B-Instruct"
    ),
    "qwen3_0_6b": ModelSpec(ModelFamily.qwen3, "Qwen/Qwen3-0.6B"),
    "qwen3_1_7b": ModelSpec(ModelFamily.qwen3, "Qwen/Qwen3-1.7B"),
    "qwen3_4b": ModelSpec(ModelFamily.qwen3, "Qwen/Qwen3-4B"),
    "qwen3_5_0_8b": ModelSpec(ModelFamily.qwen3_5, "Qwen/Qwen3.5-0.8B"),
    "qwen3_5_2b": ModelSpec(ModelFamily.qwen3_5, "Qwen/Qwen3.5-2B"),
    "qwen3_5_4b": ModelSpec(ModelFamily.qwen3_5, "Qwen/Qwen3.5-4B"),
    "phi_4_mini": ModelSpec(ModelFamily.phi_4_mini, "microsoft/Phi-4-mini-instruct"),
    "smollm2": ModelSpec(ModelFamily.smollm2, "HuggingFaceTB/SmolLM2-135M"),
    "smollm2_360m": ModelSpec(ModelFamily.smollm2, "HuggingFaceTB/SmolLM2-360M"),
    "lfm2_350m": ModelSpec(ModelFamily.lfm2, "LiquidAI/LFM2-350M"),
    "lfm2_700m": ModelSpec(ModelFamily.lfm2, "LiquidAI/LFM2-700M"),
    "lfm2_1_2b": ModelSpec(ModelFamily.lfm2, "LiquidAI/LFM2-1.2B"),
    "lfm2_5_350m": ModelSpec(ModelFamily.lfm2_5, "LiquidAI/LFM2.5-350M"),
    "lfm2_5_1_2b": ModelSpec(ModelFamily.lfm2_5, "LiquidAI/LFM2.5-1.2B-Instruct"),
    "spark_x2_5_1_7b": ModelSpec(ModelFamily.spark_x2_5, "XHToken/Spark-X2.5-1.7B"),
    "spark_x2_5_4b": ModelSpec(ModelFamily.spark_x2_5, "XHToken/Spark-X2.5-4B"),
    "spark_x2_6_1_7b": ModelSpec(ModelFamily.spark_x2_5, "XHToken/Spark-X2.6-1.7B"),
    "spark_x2_6_4b": ModelSpec(ModelFamily.spark_x2_5, "XHToken/Spark-X2.6-4B"),
}

LEGACY_MODEL_CLASS_TO_ID: Dict[str, str] = {
    model_type.value: model_type.value
    for model_type in ModelType
    if model_type.value in MODEL_SPECS
}

_LEGACY_FAMILY_ONLY_MODEL_CLASSES = {
    ModelType.stories110m: ModelFamily.llama2,
    ModelType.llama2: ModelFamily.llama2,
    ModelType.llama3: ModelFamily.llama3,
    ModelType.llama3_1: ModelFamily.llama3_1,
    ModelType.llama3_2: ModelFamily.llama3_2,
    ModelType.llama3_2_vision: ModelFamily.llama3_2_vision,
    ModelType.static_llama: ModelFamily.llama3_2,
}

_unmapped_legacy_model_types = {
    model_type.value
    for model_type in ModelType
    if model_type.value not in MODEL_SPECS
    and model_type not in _LEGACY_FAMILY_ONLY_MODEL_CLASSES
}
if _unmapped_legacy_model_types:
    raise RuntimeError(
        "Missing legacy model mappings for: " f"{sorted(_unmapped_legacy_model_types)}"
    )

LEGACY_MODEL_CLASS_TO_FAMILY: Dict[str, ModelFamily] = {
    model_type.value: (
        MODEL_SPECS[model_type.value].family
        if model_type.value in MODEL_SPECS
        else _LEGACY_FAMILY_ONLY_MODEL_CLASSES[model_type]
    )
    for model_type in ModelType
}


def _get_legacy_model_type(model_class: Any) -> str:
    return model_class.value if isinstance(model_class, Enum) else str(model_class)


def normalize_model_identity(
    model_class: Any = None,
    model_family: Any = None,
    model_id: Optional[str] = None,
) -> ModelIdentity:
    """Validate selectors and return their effective model identity."""

    if model_class is not None:
        if model_family is not None or model_id is not None:
            raise ValueError(
                "base.model_class cannot be used with base.model_family or "
                "base.model_id"
            )

        legacy_model_class = _get_legacy_model_type(model_class)
        try:
            family = LEGACY_MODEL_CLASS_TO_FAMILY[legacy_model_class]
        except KeyError as error:
            raise ValueError(f"Unknown legacy model_class {model_class!r}") from error
        return ModelIdentity(
            family,
            LEGACY_MODEL_CLASS_TO_ID.get(legacy_model_class),
        )

    if model_family is not None and model_id is not None:
        raise ValueError("base.model_family and base.model_id are mutually exclusive")

    family = ModelFamily(model_family) if model_family is not None else None
    if model_id is None:
        family = family or ModelFamily.llama3
        return ModelIdentity(family, None)

    model_spec = MODEL_SPECS.get(model_id)
    if model_spec is not None:
        return ModelIdentity(model_spec.family, model_id)

    raise ValueError(
        f"Unknown model_id {model_id!r}. Use base.model_family with "
        "base.checkpoint for a custom local model."
    )


def resolve_model_identity(
    base_config: Any,
) -> ModelIdentity:
    """Resolve the checked family and model id for a base config."""

    return normalize_model_identity(
        model_class=base_config.model_class,
        model_family=base_config.model_family,
        model_id=base_config.model_id,
    )
