# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-specific module paths and tagging rules used during lowering.

Sharding splits a graph on module paths, and the decoder's quantized-IO tagger
has a few model-specific rules. Both differ between models, so they live in one
registry here instead of in model-name conditionals in ``pass_policy``. A model
registers a layout only when it differs from the default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_GEMMA4_SELF_DECODER_PREFIX = "model.self_decoder.layers"
_GEMMA4_CROSS_DECODER_PREFIX = "model.cross_decoder.layers"


@dataclass(frozen=True)
class ModelLayout:
    """Where a model's layers live, and which extra tensors its decoder tags.

    Attributes:
        decoder_layer_prefixes: Module-path prefixes of the decoder's layers.
            The sharding pattern matches these.
        encoder_layer_prefixes: Module-path prefixes of an encoder's layers.
        layer_offset_meta: ``(prefix, meta_key)`` pairs, for a decoder whose
            layers span several modules that each count from zero. The layer
            indices under ``prefix`` are offset by ``meta[meta_key]``. Other
            prefixes start at 0. Empty for one layer sequence.
        extra_head_dim_meta: Meta key of a second head dim, whose KV-cache and
            RoPE shapes are also tagged. ``None`` when all layers share one.
        per_layer_input_traces: Source snippets. A node whose stack trace holds
            one is tagged with ``io_type``.
        donor_kv_prefixes: Source-name prefixes of the donor K/V tensors that
            shared layers consume. A matching node is tagged with ``io_type``.
    """

    decoder_layer_prefixes: Tuple[str, ...] = ("layers",)
    encoder_layer_prefixes: Tuple[str, ...] = (
        "encoder.layers",
        "vision_tower.encoder.layer",
        "vision_model.encoder.layers",
    )
    layer_offset_meta: Tuple[Tuple[str, str], ...] = ()
    extra_head_dim_meta: Optional[str] = None
    per_layer_input_traces: Tuple[str, ...] = ()
    donor_kv_prefixes: Tuple[str, ...] = ()


_DEFAULT_LAYOUT = ModelLayout()

# The tagging rules match variable names in the model's ``forward``. Renaming
# them there silently drops the tag.
_LAYOUTS: Dict[str, ModelLayout] = {
    "gemma4-e2b": ModelLayout(
        decoder_layer_prefixes=(
            _GEMMA4_CROSS_DECODER_PREFIX,
            _GEMMA4_SELF_DECODER_PREFIX,
        ),
        layer_offset_meta=((_GEMMA4_CROSS_DECODER_PREFIX, "get_n_self_layers"),),
        extra_head_dim_meta="get_global_head_dim",
        per_layer_input_traces=("per_layer_inputs[n_self:]", "per_layer_inputs[i]"),
        donor_kv_prefixes=("full_k", "full_v"),
    ),
}


def get_model_layout(model_name: Optional[str]) -> ModelLayout:
    """Look up a model's layout.

    Args:
        model_name: The model's name, as in ``ControlArgs.decoder_model``.

    Returns:
        The registered layout, or the default for an unregistered model.
    """
    layout = _LAYOUTS.get(model_name) if model_name else None
    if layout is None:
        logger.debug("No layout registered for model '%s'; using defaults", model_name)
        return _DEFAULT_LAYOUT

    return layout


def register_model_layout(model_name: str, layout: ModelLayout) -> None:
    """Register the layout of a model that differs from the default.

    Args:
        model_name: The model's name, as in ``ControlArgs.decoder_model``.
        layout: The model's layout.

    Raises:
        ValueError: If ``model_name`` already has a different layout.
    """
    existing = _LAYOUTS.get(model_name)
    if existing is not None and existing != layout:
        raise ValueError(f"model '{model_name}' already has a different layout")

    _LAYOUTS[model_name] = layout
