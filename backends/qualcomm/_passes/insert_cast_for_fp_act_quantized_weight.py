# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import copy_meta

TARGET_OPS = {
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.aten.linear.default,
}


class InsertCastForFpActQuantizedWeight(ExportPass):
    """
    Insert fp32↔fp16 casts around conv/linear nodes that have a quantized
    weight but a floating-point activation.

    Background — QNN vs PyTorch dtype contract:
      In PyTorch, a conv/linear with fp32 activation and int8 weight (e.g.
      produced by fp16a8w quantization) is valid: the weight is stored as int8
      but dequantized to fp32 before the multiply-accumulate.  QNN HTP, however,
      requires that when the weight is quantized (int8/int4) the activation must
      also be fp16, not fp32.  Passing an fp32 activation to such an op causes a
      QNN compilation error.

    Fix:
      Wrap the offending node with an fp32→fp16 cast on the input activation and
      an fp16→fp32 cast on the output, so the node itself operates in fp16 while
      the surrounding graph continues to see fp32 tensors.

      Before:  [fp32 act] → conv/linear(w=int8) → [fp32 out]
      After:   [fp32 act] → cast(fp16) → conv/linear(w=int8) → cast(fp32) → [fp32 out]

    Pattern matched:
      - Node target is in TARGET_OPS (convolution, linear)
      - Node has no QCOM_QUANT_ATTRS (activation is not quantized, i.e. fp32)
      - Weight arg (args[1]) is a parameter with QCOM_QUANT_ATTRS,
        optionally wrapped in a dequantize op
      - Input activation dtype is fp32

    The bias meta["val"] is also updated to fp16 to stay consistent with the
    fp16 compute domain of the node.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def _get_weight_param_node(self, weight: torch.fx.Node):
        """Return the underlying parameter node for a weight, unwrapping a DQ op if present."""
        if is_parameter(weight, self.edge_program):
            return weight
        if weight.target in dq_ops:
            param_node = weight.args[0]
            if isinstance(param_node, torch.fx.Node) and is_parameter(
                param_node, self.edge_program
            ):
                return param_node
        return None

    def _has_quantized_weight(self, node: torch.fx.Node) -> bool:
        if node.target not in TARGET_OPS or len(node.args) < 2:
            return False
        weight = node.args[1]
        if not isinstance(weight, torch.fx.Node):
            return False
        param_node = self._get_weight_param_node(weight)
        return param_node is not None and bool(param_node.meta.get(QCOM_QUANT_ATTRS))

    def _insert_fp32_fp16_casts(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ):
        """Wrap node with cast(fp32→fp16) on input and cast(fp16→fp32) on output."""
        input_act = node.args[0]

        with graph_module.graph.inserting_before(node):
            cast_in = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten._to_copy.default,
                (input_act,),
                {"dtype": torch.float16},
            )
            cast_in.meta = copy_meta(
                node.meta,
                lambda m: {**m, "val": input_act.meta["val"].to(torch.float16)},
            )
        node.replace_input_with(input_act, cast_in)

        # Update bias meta["val"] to fp16 if present.
        if len(node.args) > 2 and node.args[2] is not None:
            bias_node = node.args[2]
            if isinstance(bias_node, torch.fx.Node) and "val" in bias_node.meta:
                if bias_node.meta["val"].dtype == torch.float32:
                    bias_node.meta["val"] = bias_node.meta["val"].to(torch.float16)

        users = list(node.users.keys())
        orig_output_val = node.meta["val"]
        node.meta["val"] = orig_output_val.to(torch.float16)

        with graph_module.graph.inserting_after(node):
            cast_out = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten._to_copy.default,
                (node,),
                {"dtype": torch.float32},
            )
            cast_out.meta = copy_meta(
                node.meta,
                lambda m: {**m, "val": orig_output_val.to(torch.float32)},
            )

        for user in users:
            user.replace_input_with(node, cast_out)

    def call(self, graph_module: torch.fx.GraphModule):
        for node in list(graph_module.graph.nodes):
            if node.meta.get(QCOM_QUANT_ATTRS):
                continue
            if not self._has_quantized_weight(node):
                continue
            input_act = node.args[0]
            if not isinstance(input_act, torch.fx.Node):
                continue
            input_val = input_act.meta.get("val")
            if input_val is not None and input_val.dtype == torch.float32:
                self._insert_fp32_fp16_casts(graph_module, node)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
