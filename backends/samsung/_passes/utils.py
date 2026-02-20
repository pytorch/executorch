# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def none_quant_tensor_quant_meta():
    return {
        "quant_dtype": torch.float32,
        "scales": 1,
        "zero_points": 0,
    }
