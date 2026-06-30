# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Importing registers the channels_last dialect (and its edge overloads).
import executorch.backends.transforms.channels_last_ops  # noqa: F401
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

# NCHW <-> NHWC permutations (true, data-moving permutes).
_NHWC_TO_NCHW = [0, 3, 1, 2]
_NCHW_TO_NHWC = [0, 2, 3, 1]

# channels_last op -> the channels-first aten op it wraps.
_DECOMPOSITIONS = {
    exir_ops.edge.channels_last.convolution.default: exir_ops.edge.aten.convolution.default,
    exir_ops.edge.channels_last.avg_pool2d.default: exir_ops.edge.aten.avg_pool2d.default,
    exir_ops.edge.channels_last.adaptive_avg_pool2d.default: exir_ops.edge.aten._adaptive_avg_pool2d.default,
    exir_ops.edge.channels_last.upsample_bilinear2d.default: exir_ops.edge.aten.upsample_bilinear2d.vec,
    exir_ops.edge.channels_last.upsample_nearest2d.default: exir_ops.edge.aten.upsample_nearest2d.vec,
}


class DecomposeChannelsLastPass(ExportPass):
    """Decompose channels_last dialect ops into permute + aten op + permute.

    This is the channels-first CPU fallback: a channels_last op operating on
    (N, H, W, C) data is rewritten to permute the activation to (N, C, H, W),
    run the standard aten op, and permute the result back. Intended to run on
    the portion of the graph that no backend claimed; backends instead replace
    the channels_last ops with their own channels-last kernels.
    """

    def call_operator(self, op, args, kwargs, meta):
        aten_op = _DECOMPOSITIONS.get(op)
        if aten_op is not None:
            nchw_in = super().call_operator(
                exir_ops.edge.aten.permute_copy.default,
                (args[0], _NHWC_TO_NCHW),
                {},
                meta,
            )
            nchw_out = super().call_operator(
                aten_op, (nchw_in, *args[1:]), kwargs, meta
            )
            return super().call_operator(
                exir_ops.edge.aten.permute_copy.default,
                (nchw_out, _NCHW_TO_NHWC),
                {},
                meta,
            )
        if op == exir_ops.edge.channels_last.permute_copy.default:
            return super().call_operator(
                exir_ops.edge.aten.permute_copy.default, args, kwargs, meta
            )
        return super().call_operator(op, args, kwargs, meta)
