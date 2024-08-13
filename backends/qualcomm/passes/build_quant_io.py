# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED_IO

from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.tensor import TensorSpec


class BuildQuantIo(ExportPass):
    """
    To make lowering process correct, the pass assign the correct quantized dtype to spec of call_delegate.
    """

    def __init__(self):
        super(BuildQuantIo, self).__init__()

    def _make_spec(self, x):
        if isinstance(x, torch.Tensor):
            return TensorSpec.from_tensor(x)
        elif isinstance(x, (int, bool, float)):
            return x
        else:
            return None

    def _build(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # forcely update delegate node's meta['spec'] to get correct output
        # tensor size in runtime
        call_delegate = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.name == "executorch_call_delegate"
        ]
        assert len(call_delegate) == 1
        spec = []
        for n in graph_module.graph.nodes:
            if QCOM_QUANTIZED_IO in n.meta:
                n.meta["val"] = n.meta["val"].to(dtype=n.meta[QCOM_QUANTIZED_IO])
            if n.op == "call_function" and "getitem" in n.name:
                fake_tensor = n.meta["val"]
                if QCOM_QUANTIZED_IO in n.meta:
                    fake_tensor = fake_tensor.to(dtype=n.meta[QCOM_QUANTIZED_IO])
                spec.append(self._make_spec(fake_tensor))

        call_delegate[0].meta["spec"] = tuple(spec)

    def call(self, graph_module: torch.fx.GraphModule):
        self._build(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
