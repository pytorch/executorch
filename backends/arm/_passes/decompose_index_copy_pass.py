# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.get_decomposition_pass import GetDecompositionPass
from executorch.backends.arm._passes.insert_int32_casts_after_int64_placeholders import (
    InsertInt32CastsAfterInt64PlaceholdersPass,
)
from executorch.exir.pass_base import ExportPass


class DecomposeIndexCopyPass(GetDecompositionPass):
    """Decomposes aten.index_copy into aten.index_put, as well as it's
    surrounding operators.

    This pass is intended to be called in transform_for_annotation to prepare
    the graph for quantization. After quantization, this operator will be
    prepared for lowering to TOSA using the RewriteIndexPut pass

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        InsertInt32CastsAfterInt64PlaceholdersPass
    }

    targeted_ops = [
        torch.ops.aten.index_copy.default,
        torch.ops.aten.index_copy_.default,
    ]
