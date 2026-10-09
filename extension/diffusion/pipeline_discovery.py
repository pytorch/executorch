# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Discover pipeline specifications and model metadata."""

from collections.abc import Mapping
from typing import Any

from executorch.exir._warnings import experimental

from .pipeline_specs import (
    DiffusionModelInfo,
    DiffusionPipelineSpec,
    HF_PIPELINE_SPECS,
    ResolvedDiffusionPipeline,
)


def _component_slots(index: Mapping[str, Any], prefix: str) -> tuple[str, ...]:
    slots = []
    for name, component in index.items():
        if not (name == prefix or name.startswith(f"{prefix}_")):
            continue
        if (
            isinstance(component, (list, tuple))
            and len(component) >= 2
            and component[1] is not None
        ):
            slots.append(name)
    return tuple(sorted(slots, key=lambda name: (name != prefix, name)))


@experimental("This API is experimental and may change without notice.")
def pipeline_spec_from_model_index(
    index: Mapping[str, Any],
) -> DiffusionPipelineSpec:
    """Resolve a pipeline specification from a Diffusers model index."""
    pipeline_name = index.get("_class_name")
    try:
        pipeline_spec_class = HF_PIPELINE_SPECS[str(pipeline_name)]
    except KeyError as error:
        raise ValueError(
            f"unsupported diffusion pipeline {pipeline_name!r}; supported: "
            f"{sorted(HF_PIPELINE_SPECS)}"
        ) from error
    tokenizer_slots = _component_slots(index, "tokenizer")
    text_encoder_slots = _component_slots(index, "text_encoder")
    defaults = pipeline_spec_class()
    if tokenizer_slots != defaults.tokenizer_slots:
        raise ValueError(
            f"{pipeline_name} requires tokenizer components "
            f"{defaults.tokenizer_slots}, found {tokenizer_slots}"
        )
    if text_encoder_slots != defaults.text_encoder_slots:
        raise ValueError(
            f"{pipeline_name} requires text encoder components "
            f"{defaults.text_encoder_slots}, found {text_encoder_slots}"
        )
    if len(defaults.default_max_sequence_lengths) != len(text_encoder_slots):
        raise ValueError(
            f"{pipeline_name} must define one maximum sequence length per "
            "text encoder"
        )
    if len(defaults.text_encoder_input_names) != len(text_encoder_slots):
        raise ValueError(f"{pipeline_name} must define inputs for every text encoder")
    return defaults


@experimental("This API is experimental and may change without notice.")
def resolve_huggingface_pipeline(
    model_id: str, revision: str | None = None
) -> ResolvedDiffusionPipeline:
    """Resolve pipeline behavior and model facts from Hugging Face."""
    try:
        import diffusers
    except ImportError as error:
        raise ImportError("Hugging Face export requires diffusers") from error
    index = diffusers.DiffusionPipeline.load_config(model_id, revision=revision)
    pipeline_spec = pipeline_spec_from_model_index(index)
    scheduler_entry = index.get("scheduler")
    if not isinstance(scheduler_entry, (list, tuple)) or len(scheduler_entry) < 2:
        raise ValueError("model index has no valid scheduler component")
    scheduler_name = str(scheduler_entry[1])
    scheduler_class = getattr(diffusers, scheduler_name, None)
    if scheduler_class is None or not hasattr(scheduler_class, "load_config"):
        raise ValueError(f"unsupported scheduler class {scheduler_name!r}")
    scheduler_config = scheduler_class.load_config(
        model_id, subfolder="scheduler", revision=revision
    )
    prediction_type = scheduler_config.get("prediction_type")
    if not isinstance(prediction_type, str):
        prediction_type = None
    zero_negative_prompt = index.get("force_zeros_for_empty_prompt")
    if not isinstance(zero_negative_prompt, bool):
        zero_negative_prompt = pipeline_spec.default_zero_negative_prompt
    return ResolvedDiffusionPipeline(
        spec=pipeline_spec,
        model_info=DiffusionModelInfo(
            scheduler_name=scheduler_name,
            scheduler_prediction_type=prediction_type,
            max_sequence_lengths=pipeline_spec.default_max_sequence_lengths,
            zero_negative_prompt=zero_negative_prompt,
        ),
    )
