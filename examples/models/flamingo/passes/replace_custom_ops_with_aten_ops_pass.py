# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.pass_base import ExportPass
from executorch.extension.llm.custom_ops import preprocess_custom_ops  # noqa


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
        if op._name == "preprocess::pad":
            return super().call_operator(
                torch.ops.aten.constant_pad_nd.default, args, kwargs, meta
            )

        return super().call_operator(op, args, kwargs, meta)
