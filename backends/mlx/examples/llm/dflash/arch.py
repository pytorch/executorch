# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve HuggingFace architecture classes from a checkpoint's ``model_type``.

The DFlash draft stack subclasses the target architecture's own attention and
decoder layer, so those classes are looked up through the same auto-mapping
machinery ``AutoModelForCausalLM`` uses instead of being imported by name.
"""

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Type

from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING,
    model_type_to_module_name,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING


@dataclass(frozen=True)
class HFArchitecture:
    model_type: str
    config_cls: Type[Any]
    attention_cls: Type[Any]
    decoder_layer_cls: Type[Any]
    rotary_embedding_cls: Type[Any]
    apply_rotary_pos_emb: Callable[..., Any]
    eager_attention_forward: Callable[..., Any]


def _resolve_config_cls(model_type: str) -> Type[Any]:
    try:
        return CONFIG_MAPPING[model_type]
    except KeyError as e:
        raise ValueError(
            f"Unknown model_type {model_type!r}: transformers does not register a "
            f"config for it. Check the draft checkpoint's config.json."
        ) from e


def _resolve_modeling_module(model_type: str):
    module_name = model_type_to_module_name(model_type)
    return importlib.import_module(
        f".{module_name}.modeling_{module_name}", "transformers.models"
    )


def _resolve_class_prefix(model_type: str, config_cls: Type[Any]) -> str:
    """Derive the architecture's class-name prefix from its causal-LM class.

    Taken from MODEL_FOR_CAUSAL_LM_MAPPING rather than the config class name
    because the two disagree for some families (Gemma3TextConfig vs Gemma3Attention).
    """
    try:
        causal_lm_cls = MODEL_FOR_CAUSAL_LM_MAPPING[config_cls]
    except KeyError as e:
        raise ValueError(
            f"model_type {model_type!r} has no registered causal-LM class, so its "
            f"module class names cannot be derived. DFlash needs a text decoder "
            f"architecture."
        ) from e
    name = causal_lm_cls.__name__
    if not name.endswith("ForCausalLM"):
        raise ValueError(
            f"Cannot derive a class prefix for model_type {model_type!r} from "
            f"{name!r}: expected a name ending in 'ForCausalLM'."
        )
    return name[: -len("ForCausalLM")]


def _getattr_or_raise(module, name: str, model_type: str):
    attr = getattr(module, name, None)
    if attr is None:
        raise ValueError(
            f"model_type {model_type!r} is not supported by DFlash: "
            f"{module.__name__} does not define {name!r}."
        )
    return attr


@lru_cache(maxsize=None)
def resolve_arch(model_type: str) -> HFArchitecture:
    config_cls = _resolve_config_cls(model_type)
    module = _resolve_modeling_module(model_type)
    prefix = _resolve_class_prefix(model_type, config_cls)

    return HFArchitecture(
        model_type=model_type,
        config_cls=config_cls,
        attention_cls=_getattr_or_raise(module, f"{prefix}Attention", model_type),
        decoder_layer_cls=_getattr_or_raise(
            module, f"{prefix}DecoderLayer", model_type
        ),
        rotary_embedding_cls=_getattr_or_raise(
            module, f"{prefix}RotaryEmbedding", model_type
        ),
        apply_rotary_pos_emb=_getattr_or_raise(
            module, "apply_rotary_pos_emb", model_type
        ),
        eager_attention_forward=_getattr_or_raise(
            module, "eager_attention_forward", model_type
        ),
    )
