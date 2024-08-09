# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass
from executorch.extension.llm.custom_ops import preprocess_custom_ops  # noqa
from torch._ops import OpOverload

_REPLACE_CUSTOM_OP_WITH_ATEN_OP: Dict[OpOverload, OpOverload] = {
    torch.ops.preprocess.pad.default: torch.ops.aten.constant_pad_nd.default,
}


class ReplaceCustomOpsWithAtenOpsPass(ExportPass):
    """
    Goes through all ops and replaces custom ops with aten ops. In some cases
    aten ops cannot be exported due to dynamism, eg. pad in flamingo preprocess.
    Use a custom op to pass export, and replace it with the aten op post-export,
    which avoids re-writing the op in C++.
    """

    def __init__(self) -> None:
        super().__init__()

    def call_operator(self, op, args, kwargs, meta):
        if op in _REPLACE_CUSTOM_OP_WITH_ATEN_OP:
            return super().call_operator(
                _REPLACE_CUSTOM_OP_WITH_ATEN_OP[op], args, kwargs, meta
            )

        return super().call_operator(op, args, kwargs, meta)
