# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from executorch.examples.models.gemma4.convert_weights import convert_weights
from executorch.examples.models.llama.model import Llama2Model


# Mirrors qwen3_5_moe's `Qwen35MoEConfig.from_hf_config(...)` factory style:
# a single class with a variant selector instead of one subclass per size.
_CONFIG_DIR = Path(__file__).parent / "config"
_VARIANT_CONFIGS = {
    "e2b": _CONFIG_DIR / "e2b_config.json",
    "e4b": _CONFIG_DIR / "e4b_config.json",
}


def config_path_for_variant(variant: str) -> Path:
    """Return the config JSON path for a Gemma 4 variant."""
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown Gemma 4 variant {variant!r}; "
            f"expected one of {sorted(_VARIANT_CONFIGS)}"
        )
    return _VARIANT_CONFIGS[variant]


class Gemma4Model(Llama2Model):
    """Gemma 4 multimodal LLM (E2B / E4B) on the shared Llama transformer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "Gemma4Model",
    "config_path_for_variant",
    "convert_weights",
]
