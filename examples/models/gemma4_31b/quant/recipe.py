# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization recipe: declares what to quantize and how.

A ``QuantRecipe`` is an ordered list of ``QuantRule`` objects matched against
weight FQNs. First match wins. The recipe says nothing about packing format,
tensor subclass, or target backend.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class QuantConfig:
    """Per-weight quantization parameters."""

    bits: int  # storage width: 4 or 8 (6-bit formats like Q6_K are widened to 8)
    group_size: int  # 32, 64, 128
    symmetric: bool  # True = no zero point
    method: str  # "min_max" | "hqq" | "gguf_q4_k" | "gguf_q6_k"


@dataclass
class QuantRule:
    """A single recipe rule: regex pattern + config + optional layer filter."""

    pattern: str  # regex matched against weight FQN
    config: Optional[QuantConfig]  # None = skip (leave unquantized)
    layers: Optional[set[int]] = field(default=None, repr=False)  # None = all layers


@dataclass
class QuantRecipe:
    """Ordered list of rules. First match wins."""

    rules: list[QuantRule]

    def get_config(self, fqn: str) -> Optional[QuantConfig]:
        """Return the ``QuantConfig`` for a weight FQN, or ``None`` to skip."""
        layer_idx = self._extract_layer_idx(fqn)
        for rule in self.rules:
            if rule.layers is not None:
                if layer_idx is None or layer_idx not in rule.layers:
                    continue
            if re.fullmatch(rule.pattern, fqn):
                return rule.config
        return None

    @staticmethod
    def _extract_layer_idx(fqn: str) -> Optional[int]:
        m = re.search(r"layers\.(\d+)\.", fqn)
        return int(m.group(1)) if m else None
