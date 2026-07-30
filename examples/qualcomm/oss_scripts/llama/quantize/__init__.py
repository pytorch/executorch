# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.qualcomm.oss_scripts.llama.quantize.ptq import PTQStrategy
from executorch.examples.qualcomm.oss_scripts.llama.quantize.strategy import (
    QuantizationStrategy,
)

__all__ = ["QuantizationStrategy", "PTQStrategy"]
