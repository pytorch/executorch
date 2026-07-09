# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.quant_utils import is_quant, tag_as_implicit_q_dq
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class InsertPadQDQPass(XNNPACKPass):
    """
    Completes the quantization of a constant_pad_nd that sits inside a quantized
    region but was left with an fp32 output.

    An even-kernel 'same'-padding conv decomposes (after quantization) into
    dequant -> constant_pad_nd -> convolution. Because the pad is introduced by
    to_edge decomposition -- after the quantizer has run -- it is never annotated,
    so no quantize follows it and its output would serialize as fp32. The
    downstream conv would then reject its (now fp32) activation.

    A zero-valued pad preserves quantization, so we insert an implicit
    quantize -> dequantize pair after the pad, reusing the feeding dequant's
    params. The pad then delegates as a normal quantized XNNStaticConstantPad and
    the conv sees a proper dequantized activation. The pad node itself is left in
    place, so all graph shapes stay consistent through later retracing passes.
    """

    def _insert_qdq_after(self, graph, pad, q_params):
        with graph.inserting_after(pad):
            q = graph.create_node(
                "call_function",
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                args=(),
            )
            q.meta = pad.meta.copy()
            tag_as_implicit_q_dq(q)
        with graph.inserting_after(q):
            dq = graph.create_node(
                "call_function",
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                args=(q,) + q_params,
            )
            dq.meta = q.meta.copy()
            tag_as_implicit_q_dq(dq)
        pad.replace_all_uses_with(dq)
        # Set last so replace_all_uses_with above does not rewrite the quantize's
        # own input.
        q.args = (pad,) + q_params

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for pad in list(graph.nodes):
            if (
                pad.op != "call_function"
                or pad.target != exir_ops.edge.aten.constant_pad_nd.default
            ):
                continue

            # Only per-tensor static activations are handled: _insert_qdq_after
            # builds quantize_per_tensor.default, so the feeding dequant must have
            # the matching per-tensor signature (a per-channel/per-token/affine
            # dequant would supply mismatched args).
            dq = pad.args[0]
            if (
                not isinstance(dq, torch.fx.Node)
                or dq.target
                != exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):
                continue

            # Skip if the pad's output is already quantized. Requiring *no* user to
            # be a quantize (rather than merely "not all") avoids double-quantizing
            # pre-existing quant consumers when the pad has mixed users.
            if not pad.users or any(is_quant(user) for user in pad.users):
                continue

            self._insert_qdq_after(graph, pad, tuple(dq.args[1:]))

        graph_module.recompile()
        return PassResult(graph_module, True)
