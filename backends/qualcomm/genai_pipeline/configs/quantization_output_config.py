# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any


@dataclass
class QuantizationOutputConfig:
    """Output produced by the quantization stage.

    Attributes:
        quantized_model: The quantized nn.Module or path to saved QDQ model.
    """

    quantized_model: Any = None
