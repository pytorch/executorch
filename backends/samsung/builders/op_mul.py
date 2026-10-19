# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.samsung.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.samsung.builders.utils import get_map_dtype, get_tensor
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.transforms.utils import is_param_node


@register_node_visitor
class MulVisitor(NodeVisitor):
    target = "aten.mul.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        enn_graph: EnnGraph,
        vals_to_ids: Dict[torch.Tensor, int],
    ) -> bool:

        input_ids = []
        for input_node in node.args[:2]:
            tensor = get_tensor(self.exported_program, input_node)
            if is_param_node(self.exported_program, input_node) and tensor.dim() == 0:
                # The SDK's multiply exporter cannot broadcast scalar constants.
                tensor = tensor.expand(node.meta["val"].shape).contiguous()
                const_data = None
                if not isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor):
                    const_data = tensor.cpu().detach().numpy()
                input_ids.append(
                    enn_graph.define_tensor(
                        f"{node.name}_{input_node.name}",
                        list(tensor.shape),
                        get_map_dtype(tensor.dtype),
                        "CONSTANT",
                        const_data,
                        quant_param=input_node.meta.get("quantize_attrs"),
                    )
                )
            else:
                input_ids.append(self.define_tensor(input_node, enn_graph, vals_to_ids))
        params = {}
        self._update_params_qdtype(node, params)

        output_id = self.define_tensor(node, enn_graph, vals_to_ids)

        enn_graph.define_op(node.name, "ELTMUL", input_ids, [output_id], params)

        return True
