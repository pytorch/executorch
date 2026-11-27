# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class InferenceResult:
    """Final result for one evaluation method.

    Attributes:
        generations: Text the runner generated, or ``None`` when the
            method generates no text.
        performance_metrics: The runner's throughput over the method's
            requests: ``num_requests`` and ``mean_tokens_per_sec``.
        eval_results: The evaluator's metrics, such as SQNR or perplexity.
        result_files: The pulled runner files plus the ``results.json`` and
            ``performance.json`` written for the method.
    """

    generations: Optional[List[str]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    eval_results: Dict[str, Any] = field(default_factory=dict)
    result_files: List[Path] = field(default_factory=list)


@dataclass
class InferenceOutputConfig:
    """Output produced by the inference stage.

    Attributes:
        results: Final results keyed by evaluation method, such as
            ``prompt_eval`` and ``tasks_eval``.
    """

    results: Dict[str, InferenceResult] = field(default_factory=dict)
