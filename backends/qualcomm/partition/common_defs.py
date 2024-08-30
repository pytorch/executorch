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
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.slice_scatter.default,
    exir_ops.edge.aten.copy.default,
]

to_be_implemented_operator = [
    exir_ops.edge.aten.any.dim,
    exir_ops.edge.aten.eq.Scalar,
    exir_ops.edge.aten.full_like.default,
    exir_ops.edge.aten.logical_not.default,
    exir_ops.edge.aten.where.self,
]

allow_list_operator = [
    _operator.getitem,
]
