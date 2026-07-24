# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import cast, Optional, Tuple

from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
_DIV = exir_ops.edge.aten.div.Tensor
_MUL = exir_ops.edge.aten.mul.Tensor


class FoldScaleIntoQuantizePass(ExportPass):
    """Drop a constant elementwise scale (``x / c`` or ``x * c``) that the
    quantizer already absorbed into the surrounding quantization scale.

    Runs after ``FoldAndAnnotateQParamsPass``. An attention-score ``/sqrt(d)``
    survives that pass as an unannotated fp32 op left inside a
    ``dequantize_per_tensor(S_in) -> div/mul(c) -> quantize_per_tensor(S_out)``
    sandwich: the observer that set ``S_out`` saw the post-scale range, so
    ``S_out == S_in / c`` (div) or ``S_in * c`` (mul) and the two zero-points
    match. Under that relation the sandwich is the identity on the int8 values
    (``quantize(dequantize(q, S_in) / c, S_out) == q``), so it is removed and the
    producer's int8 wired straight into the consumer.

    The pass rewrites no quantization parameters -- it only recognises that the
    constant is already in ``S_out`` and deletes the redundant fp32 round-trip.
    Because no scale changes, a downstream SharedQspec consumer (view/reshape/
    pool) that shares ``S_out`` is never disturbed. If the ``S_out == S_in/c``
    (or ``*c``) and matching-zero-point relation does not hold -- per-channel
    qparams, a non-scalar constant, or a scale the quantizer did not absorb --
    the fold is skipped and the fp32 op is left in place.

    A constant add/sub cannot fold this way: an additive shift moves the affine
    quantization parameters by a generally non-integer amount, so no
    integer-preserving requantize identity exists -- a separate, lossy transform,
    and no current model needs it.
    """

    def __init__(self, exported_program: Optional[ExportedProgram] = None) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: GraphModule) -> PassResult:
        ep = self.exported_program
        if ep is None:
            return PassResult(graph_module, False)

        modified = False
        for node in list(graph_module.graph.nodes):
            match = self._absorbed_scale_sandwich(node, ep)
            if match is None:
                continue
            dequantize, quantize, producer = match
            quantize.replace_all_uses_with(producer)
            graph_module.graph.erase_node(quantize)
            graph_module.graph.erase_node(node)
            graph_module.graph.erase_node(dequantize)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    def _absorbed_scale_sandwich(
        self, node: Node, ep: ExportedProgram
    ) -> Optional[Tuple[Node, Node, Node]]:
        """Return ``(dequantize, quantize, producer)`` when ``node`` is a
        constant div/mul the quantizer already folded into the surrounding
        quantize scale, so the sandwich can be deleted bit-exactly; else None."""
        if node.op != "call_function" or node.target not in (_DIV, _MUL):
            return None
        scaled, const = node.args[0], node.args[1]
        if not (isinstance(scaled, Node) and isinstance(const, Node)):
            return None
        # A dequantize -> op -> quantize sandwich, each side used only by the op.
        if scaled.target is not _DEQUANTIZE or len(scaled.users) != 1:
            return None
        # A dequantize's input is always a Node.
        producer = cast(Node, scaled.args[0])
        users = list(node.users)
        if len(users) != 1 or users[0].target is not _QUANTIZE:
            return None
        quantize = users[0]

        c = self._scalar_constant(const, ep)
        if c is None:
            return None

        # (de)quantize args: (input, scale, zero_point, qmin, qmax, dtype).
        s_in, zp_in = scaled.args[1], scaled.args[2]
        s_out, zp_out = quantize.args[1], quantize.args[2]
        if not (isinstance(s_in, float) and isinstance(s_out, float)):
            return None
        expected = s_in / c if node.target is _DIV else s_in * c
        # Bit-exact only if the quantizer folded c into s_out, the zero-points
        # match, and the quantize's clamp range/dtype (args 3..5) match the
        # dequantize's -- so the removed requantize is a no-op on the producer int8.
        if (
            zp_in != zp_out
            or scaled.args[3:6] != quantize.args[3:6]
            or not math.isclose(s_out, expected, rel_tol=1e-6)
        ):
            return None
        return scaled, quantize, producer

    def _scalar_constant(self, const: Node, ep: ExportedProgram) -> Optional[float]:
        if not is_param_node(ep, const):
            return None
        const_t = get_param_tensor(ep, const)
        if const_t is None or const_t.numel() != 1:
            return None
        c = float(const_t.reshape(-1)[0])
        if c == 0.0 or not math.isfinite(c):
            return None
        return c
