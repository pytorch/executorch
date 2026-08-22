# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes import ToOutVarPass
from torch.fx.passes.infra.pass_base import PassResult


class CadenceToOutVarPass(ToOutVarPass):
    """Adds support for custom cadence inplace ops."""

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for slice_scatter_inplace in graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.cadence.slice_scatter_.default
        ):
            slice_scatter_inplace.target = slice_scatter_inplace.target._op
        return super().call(graph_module)
