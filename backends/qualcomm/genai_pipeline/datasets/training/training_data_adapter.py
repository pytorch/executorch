# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training data adapter protocol.

A purpose adapter assembling QAT training data from raw dataset loaders. Its
output is always component-keyed so strategy code can consume one LLM/MLLM
contract.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class TrainingDataAdapter(Protocol):
    """Protocol for assembling QAT training data."""

    def generate_training_data(
        self,
        tokenizer: Any,
        num_samples: int = ...,
        seq_length: int = ...,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Iterable[Any]]:
        """Return ``{component: iterable}`` training data."""
        ...
