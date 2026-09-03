# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass


class NormalizeTransposePass(ExportPass):
    """
    Even with functionalization on, we still get graph with
    torch.ops.aten.t.default op. Ideally we should fix functionalization.
    TODO: once we have that, we should remove this pass.
    Check test_normalize_transpose_rewrites_transpose_to_copy in test_pass_infra.py
    for more details.
    """

    targeted_ops = {torch.ops.aten.t.default}

    def call_operator(self, op, args, kwargs, meta):
        if op in self.targeted_ops:
            return super().call_operator(
                torch.ops.aten.t_copy.default, (args[0],), kwargs, meta
            )
        return super().call_operator(op, args, kwargs, meta)
