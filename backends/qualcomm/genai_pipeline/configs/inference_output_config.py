# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.devtools.etdump.schema_flatcc import ETDump


@dataclass
class InferenceOutputConfig:
    """Output produced by the inference stage.

    Attributes:
        inference_results: Generated text output(s) from the model, or the paths
            of the pulled result files when the adapter writes results to disk
            rather than returning them. Stays ``List[str]`` rather than widening
            to ``List[Any]``: ``InferenceResult.output_data`` is already decoded
            text by contract, so it needs no conversion, and coercing it in the
            strategy would risk stringifying raw token ids.
        performance_metrics: Performance data (e.g., TTFT, tokens/sec).
        eval_results: Evaluation metric results (e.g., SQNR, perplexity).
        etdump: Optional ETDump for debugging. ExecuTorch engine only.
    """

    inference_results: Optional[List[str]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    eval_results: Dict[str, Any] = field(default_factory=dict)
    etdump: Optional["ETDump"] = None
