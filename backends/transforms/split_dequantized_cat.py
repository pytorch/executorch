# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections import defaultdict
from typing import DefaultDict, List, Tuple

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    get_overload_packet,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class SplitDequantizedCatPass(RemoveOrReplacePassInterface):
    """Split a cat node so that quantize consumers get their own copy.

    Fires when a cat has all floating-point inputs, at least one dequantize
    input, and at least one quantize consumer.  Quant consumers are grouped
    by matching qparams; each group receives a dedicated duplicate of the
    cat node.  Non-quant consumers stay on the original cat, whose
    semantics are unchanged.

    A later pass (e.g. AdvanceQuantizeOpAboveDefChainPass extended for cat)
    can then hoist each quant above its single-consumer cat copy without
    affecting the non-quant paths.
    """

    quantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
    }
    dequantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.cat.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        cat_inputs = node.args[0]
        if not isinstance(cat_inputs, (list, tuple)):
            return False

        has_dequant_input = False
        for inp in cat_inputs:
            assert isinstance(inp, torch.fx.Node)
            val = inp.meta["val"]
            if val is None or not val.is_floating_point():
                return False
            if get_overload_packet(inp.target) in self.dequantize_op_packets:
                has_dequant_input = True

        if not has_dequant_input:
            return False

        quant_groups: DefaultDict[Tuple, List[torch.fx.Node]] = defaultdict(list)
        for user in list(node.users.keys()):
            if get_overload_packet(user.target) in self.quantize_op_packets:
                quant_groups[user.args[1:]].append(user)

        if not quant_groups:
            return False

        graph = node.graph
        dim = get_arg(node, "dim", int)
        for quant_consumers in quant_groups.values():
            with graph.inserting_after(node):
                dup_cat = graph.call_function(
                    exir_ops.edge.aten.cat.default,
                    args=(list(cat_inputs), dim),
                )
                dup_cat.meta = node.meta.copy()

            for q_node in quant_consumers:
                q_node.replace_input_with(node, dup_cat)

        return True
