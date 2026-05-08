# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_output_qparams,
)
from executorch.backends.cortex_m.passes.passes_utils import (
    get_activation_bounds,
    quantize_val,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger(__name__)


class QuantizedClampActivationPass(ExportPass):
    """Canonicalize remaining clamp-like activations on quantized tensors.

    This pass runs after activation fusion, so any remaining relu/hardtanh/clamp
    still needs to execute in the quantized domain. It rewrites relu and
    hardtanh variants to `aten.clamp.default` and quantizes the clamp bounds so
    the portable kernel consumes and produces int8 tensors.
    """

    TARGETS = {
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.relu_.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.hardtanh_.default,
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.aten.clamp.Tensor,
    }

    def _get_quantized_bounds(
        self, node: Node, qparams_dict: dict[str, Any]
    ) -> tuple[int | None, int | None] | None:
        qmin = qparams_dict["qmin"]
        qmax = qparams_dict["qmax"]
        scale = qparams_dict["scale"]
        zp = qparams_dict["zp"]

        bounds = get_activation_bounds(node)
        if bounds is None:
            logger.warning(
                "Cannot rewrite %s because bounds are not compile-time scalars.",
                node.name,
            )
            return None
        min_val, max_val = bounds

        quantized_min = (
            int(quantize_val(min_val, scale, zp, qmin, qmax))
            if min_val is not None
            else None
        )
        quantized_max = (
            int(quantize_val(max_val, scale, zp, qmin, qmax))
            if max_val is not None
            else None
        )
        return quantized_min, quantized_max

    def _is_quantized_int8_activation(self, node: Node) -> bool:
        input_node = node.args[0] if len(node.args) > 0 else None
        if not isinstance(input_node, Node):
            return False
        try:
            tensor = get_first_fake_tensor(input_node)
        except Exception:
            return False
        if tensor is None or tensor.dtype != torch.int8:
            return False

        try:
            qparams_dict = get_output_qparams(node)[0]._asdict()
        except (ValueError, KeyError):
            logger.warning(
                "Cannot quantize clamp bounds for %s without output qparams.",
                node.name,
            )
            return False

        scale = qparams_dict["scale"]
        zp = qparams_dict["zp"]
        if not isinstance(scale, float) or not isinstance(zp, int):
            logger.warning(
                "Cannot quantize clamp bounds for %s with non per-tensor qparams.",
                node.name,
            )
            return False

        return True

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False

        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in self.TARGETS:
                continue
            if not self._is_quantized_int8_activation(node):
                continue

            qparams_dict = get_output_qparams(node)[0]._asdict()

            quantized_bounds = self._get_quantized_bounds(node, qparams_dict)
            if quantized_bounds is None:
                continue

            quantized_min, quantized_max = quantized_bounds
            node.target = exir_ops.edge.aten.clamp.default
            node.args = (node.args[0], quantized_min, quantized_max)
            modified = True

        if modified:
            graph_module = super().call(graph_module).graph_module
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
