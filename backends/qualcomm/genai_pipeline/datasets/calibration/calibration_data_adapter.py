# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Calibration data adapter protocol.

A purpose adapter assembles calibration data from raw dataset loaders. Its output
is always component-keyed so quantization strategies can consume one contract for
LLM and MLLM models.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class CalibrationDataAdapter(Protocol):
    """Protocol for assembling calibration data for quantization."""

    def generate_calibration_data(
        self,
        tokenizer: Any,
        example_inputs: Optional[Dict[str, Any]] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Iterable[Any]]:
        """Return ``{component: iterable}`` calibration inputs.

        Args:
            tokenizer: Tokenizer used by dataset loaders.
            example_inputs: Model calibration-graph signatures used by adapters
                that construct collators. Adapters without collators may ignore it.
            extra_options: Optional data-source settings. ``max_context_len`` is
                the shared sequence-length source of truth for dataset loaders
                and collators.

        Returns:
            A map from artifact component keys to calibration iterables.

        Note:
            Sizing such as sample count, batch size, and seed is fixed when the
            adapter is constructed rather than passed to this method.
        """
        ...
