# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
APIs to help lowering edge dialect ops to other dialects.
"""
from typing import Optional

import torch
from torch._ops import OpOverloadPacket


def get_torch_op_overload(
    namespace: str, opname: str, overload: Optional[str]
) -> torch._ops.OpOverload:
    packet: OpOverloadPacket = getattr(getattr(torch.ops, namespace), opname)
    if overload:
        return getattr(packet, overload)
    else:
        return packet.default


def get_callable(name):
    main, suffix = name.split(".")
    return get_torch_op_overload("aten", main, suffix)
