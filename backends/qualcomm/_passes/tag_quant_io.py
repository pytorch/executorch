# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable

import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_QUANT_ATTRS,
    QCOM_QUANT_ATTRS_MAP,
    QCOM_QUANTIZED_IO,
)
from executorch.exir.pass_base import ExportPass, PassResult


class TagQuantIO(ExportPass):
    """
    Tag the IO nodes that handle quantized tensors to avoid inserting Q/DQ operations in qnn_preprocess.
    """

    def __init__(self, get_quant_io_dtype_fn: Callable = None):
        super(TagQuantIO, self).__init__()
        self.get_quant_io_dtype_fn = get_quant_io_dtype_fn

    def _tag_quant_io(self, gm: torch.fx.GraphModule):
        for node in gm.graph.nodes:
            if dtype := self.get_quant_io_dtype_fn(node):
                node.meta[QCOM_QUANTIZED_IO] = dtype

    def _record_output_quant_attrs_map(self, gm: torch.fx.GraphModule):
        for node in gm.graph.nodes:
            if node.op == "output":
                node.meta.setdefault(QCOM_QUANT_ATTRS_MAP, {})
                for arg in node.args[0]:
                    if QCOM_QUANT_ATTRS in arg.meta:
                        node.meta[QCOM_QUANT_ATTRS_MAP][arg] = arg.meta[
                            QCOM_QUANT_ATTRS
                        ]

    def call(self, graph_module: torch.fx.GraphModule):
        self._tag_quant_io(graph_module)
        self._record_output_quant_attrs_map(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
