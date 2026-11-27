# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict

from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle


@dataclass
class QuantizationOutputConfig:
    """Output produced by the quantization stage.

    Attributes:
        graphs: Component- and graph-keyed quantized modules with their export
            inputs, metadata, and optional quantized IO dtypes. The compilation
            stage consumes the bundles and bakes metadata-derived KV-cache and
            logits quantization attributes into the ``.pte``.
    """

    graphs: Dict[str, Dict[str, GraphBundle]] = None
