# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.pass_base import ExportPass, PassResult


class AnnotateStack(ExportPass):
    """
    During decomposition stage, some unsqueeze op will appear.
    These unsqueeze op does not carry quant attributes and will need to use previous node's quant attributes
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if (
                node.meta.get("torch_fn", ("", ""))[1]
                == "builtin_function_or_method.stack"
            ):
                if (
                    QCOM_QUANT_ATTRS not in node.meta
                    and QCOM_QUANT_ATTRS in node.args[0].meta
                ):
                    node.meta[QCOM_QUANT_ATTRS] = node.args[0].meta[QCOM_QUANT_ATTRS]

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
