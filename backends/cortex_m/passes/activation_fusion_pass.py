# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.backends.arm._passes.quant_args import QuantArgs

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger(__name__)


class ActivationFusionPass(ExportPass):
    """Fuse activations into preceding Cortex-M quantized operators.

    Supported activation patterns:
        q-> [conv2d, linear] -> [relu, hardtanh, hardsigmoid] -> dq

    Fusing works by clamping the quantized output range (and zero-point when
    required) of the preceding Cortex-M operator, then removing the activation
    node from the graph.
    """

    TARGETS = {
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.hardsigmoid.default,
    }

    FUSE_OPS = {
        exir_ops.edge.aten.linear.default,
        exir_ops.edge.aten.convolution.default,
    }

    def _quantize(self, val, scale, zp, qmin, qmax):
        return min(max(round(val / scale + zp), qmin), qmax)

    def _get_validated_qparams(self, node, input_node):

        if "input_qparams" not in input_node.meta or "output_qparams" not in node.meta:
            logger.warning(
                f"Cannot fuse activation for {input_node.name}->{node.name} as the pattern wasn't quantized properly."
            )
            return None

        qparams_dict = node.meta["output_qparams"][0]._asdict()
        zp = qparams_dict["zp"]
        scale = qparams_dict["scale"]
        qmin = qparams_dict["qmin"]
        qmax = qparams_dict["qmax"]

        if not isinstance(scale, float) or not isinstance(zp, int):
            logger.warning(
                f"Cannot fuse activation {node.name} as quantization parameters are not per tensor."
            )
            return None

        match node.target:
            case exir_ops.edge.aten.relu.default:
                quantized_min_val = self._quantize(0, scale, zp, qmin, qmax)
                quantized_max_val = qmax
            case exir_ops.edge.aten.hardtanh.default:
                quantized_min_val = self._quantize(node.args[1], scale, zp, qmin, qmax)
                quantized_max_val = self._quantize(node.args[2], scale, zp, qmin, qmax)
            case exir_ops.edge.aten.hardsigmoid.default:
                quantized_min_val = self._quantize(0, scale, zp, qmin, qmax)
                quantized_max_val = self._quantize(1, scale, zp, qmin, qmax)
            case _:
                raise RuntimeError("Unexpected target {node.target}.")

        # If the minimal quantized value is larger than the qmin, it means that the quantized range contains
        # invalid values [qmin, ..., quantized_min_val-1], indicating bad quantization parameters.
        if qparams_dict["qmin"] != quantized_min_val:
            logger.warning(
                f"Cannot fuse activation {node.name} as qmin is out of range."
            )
            return None

        # If the maximal quantized value is smaller than the qmax, it means that the quantized range contains
        # invalid values [quantized_max_val + 1, ... , qmax], indicating bad quantization parameters.
        if quantized_max_val != qparams_dict["qmax"]:
            logger.warning(
                f"Cannot fuse activation {node.name} as qmax is out of range."
            )
            return None

        return qparams_dict

    def _update_qparams_hardsigmoid(self, quant_dict):
        """
        Returns quant_dict with scale and zp updated to match hardsigmoid activation.

        The quantized output from the hard sigmoid is defined by
            Q(y) = clamp(round(y/scale + zp), qmin, qmax)
            y = clamp(x/6 + 1/2, 0, 1)
        where x is the output of the fused activation op, conv or linear.

        Q(y) can be rewritten as a function of only x:
            Q(y) = clamp(round(clamp(x/6 + 1/2, 0, 1)/scale + zp), qmin, qmax)
            Q(y) = clamp(round(clamp((x/(6*scale) + 1/(2*scale) + zp, zp, 1/scale + zp)), qmin, qmax)

        From definition of the qparams mapping the output in the range [0,1] to quantized range
        [qmin, qmax], we have:
            zp = Q(0) <= qmin
            1/scale + zp = Q(1) >= qmax
        which makes the inner clamp redundant.

        Therefore, hardsigmoid is equivalent to a quantization with modified parameters
            new_scale := 6*scale
            new_zp = zp + 1/(2*scale) ~= zp + round(1/(2*scale))
        """

        new_scale = quant_dict["scale"] * 6

        new_zp = quant_dict["zp"] + round(1 / (2 * quant_dict["scale"]))
        clamped_new_zp = max(quant_dict["qmin"], min(quant_dict["qmax"], new_zp))

        quant_dict["scale"] = new_scale
        quant_dict["zp"] = clamped_new_zp

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list[Node] = []

        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in self.TARGETS:
                continue

            input_node = node.args[0]
            if (
                input_node.op != "call_function"
                or input_node.target not in self.FUSE_OPS
            ):
                logger.warning(
                    f"Cannot fuse activation {node.name} as input node {input_node.name} is not a supported fused activation op."
                )
                continue
            if len(input_node.users.values()) > 1:
                logger.warning(
                    f"Cannot fuse activation {node.name} as input node {input_node.name} has multiple users."
                )
                continue

            if (qparams_dict := self._get_validated_qparams(node, input_node)) is None:
                continue

            if node.target == exir_ops.edge.aten.hardsigmoid.default:
                self._update_qparams_hardsigmoid(qparams_dict)

            input_node.meta["output_qparams"][0] = QuantArgs(**qparams_dict)

            node.replace_all_uses_with(input_node)
            nodes_to_erase.append(node)
            modified = True

        for node in nodes_to_erase:
            graph_module.graph.erase_node(node)

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
