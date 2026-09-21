# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Tuple

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue

from torch.fx.node import Argument


class NormalizeConvolutionArgs(ExportPass):
    """
    Broadcasts single element stride/padding/dilation/output_padding lists of a
    convolution to the number of spatial dimensions.

    ATen allows these arguments to be given as a single value that applies to
    every spatial dimension, and torch.nn does exactly that for
    `padding="valid"`, which exports as `padding=[0]` rather than `[0, 0]`.
    The Vulkan convolution reads them as fixed width vectors
    (`make_ivec2_from_list` -> `make_ivec2`, which requires exactly 2 elements),
    so a 2D convolution written that way aborts at the first inference:

      make_ivec2 ... (ints.size() == 2) is false!

    Normalizing here keeps the graph ATen compliant and leaves the runtime
    unchanged.
    """

    # arg index -> name, for the list arguments of aten.convolution
    _list_arg_indices: Dict[int, str] = {
        3: "stride",
        4: "padding",
        5: "dilation",
        7: "output_padding",
    }

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.convolution.default:
            return super().call_operator(op, args, kwargs, meta)

        # weight is (out_channels, in_channels / groups, *kernel_size)
        weight = args[1]
        # pyre-ignore[16]
        spatial_dims = len(weight.node.meta["val"].shape) - 2
        if spatial_dims < 2:
            return super().call_operator(op, args, kwargs, meta)

        new_args: List[Argument] = list(args)
        modified = False
        for idx in self._list_arg_indices:
            if idx >= len(new_args):
                continue
            value = new_args[idx]
            if isinstance(value, (list, tuple)) and len(value) == 1:
                new_args[idx] = [value[0]] * spatial_dims
                modified = True

        if not modified:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(op, tuple(new_args), kwargs, meta)
