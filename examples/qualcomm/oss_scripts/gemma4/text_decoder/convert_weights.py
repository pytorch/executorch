# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Re-key weights from upstream convert_hf_to_custom format to QNN decoder format.

Upstream uses:  *.self_attn.*  and  *.mlp.*
QNN uses:       *.attention.*  and  *.feed_forward.*
"""

from typing import Dict

import torch


def remap_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, tensor in state_dict.items():
        new_key = key.replace(".self_attn.", ".attention.").replace(
            ".mlp.", ".feed_forward."
        )
        remapped[new_key] = tensor
    return remapped
