# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import ComputeConstantOpsAOTPass
from executorch.exir.dialects._ops import ops as exir_ops


class ComputeQuantizedConstantOpsPass(ComputeConstantOpsAOTPass):
    """Give the quantized constant ops their int8 values.

    Folding removes the Q node after these ops but leaves them producing float,
    so the int8 kernels they feed would read float data. A ``full`` stays a
    runtime fill with its fill value quantized, so it adds no constant data; the
    other ops are computed in float and stored as quantized constants. Runs
    after ``ConvertFullLikeToFullPass``, so a ``full_like`` arrives as ``full``.
    """

    def compute_node_aot(self, node: torch.fx.Node) -> bool:
        output_qparams = node.meta.get("output_qparams")
        if not output_qparams:
            return False
        if node.target == exir_ops.edge.aten.full.default:
            qparams = output_qparams[0]
            fill_value = node.args[1]
            assert isinstance(fill_value, (int, float))
            fill = qparams.quantize_value(torch.tensor(float(fill_value)))
            node.args = (node.args[0], int(fill.item()))
            node.kwargs = {**node.kwargs, "dtype": qparams.dtype}
            return True
        return super().compute_node_aot(node)
