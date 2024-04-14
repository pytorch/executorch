# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import _operator

from executorch.exir.dialects._ops import ops as exir_ops


not_supported_operator = [
    exir_ops.edge.aten.arange.start_step,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.index.Tensor,
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.slice_scatter.default,
    exir_ops.edge.aten.index_put.default,
]

allow_list_operator = [
    _operator.getitem,
]
