# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


_MUL_SCALAR = torch.ops.aten.mul.Scalar
_BMM = torch.ops.aten.bmm.default
_EXPAND = torch.ops.aten.expand.default
_VIEW = torch.ops.aten.view.default
_SOFTMAX_TARGETS = {
    torch.ops.aten.softmax.int,
    torch.ops.aten._softmax.default,
}

_INPUT_PASSTHROUGH = {
    _EXPAND,
    _VIEW,
}


class MoveSDPAScaleAfterBmmPass(ExportPass):
    """Canonicalize the split default-SDPA score scale before annotation.

    PyTorch's SDPA decomposition applies ``d**(-1/4)`` independently to the
    query and key before the score matmul::

        bmm(q * c, k * c),  c = d**(-1/4)

    For Cortex-M quantization this can place the pre-scale and post-scale
    observers in different shared-qspec domains (for example after a packed
    QKV projection), causing the ordinary quantizer to choose qparams that are
    not proportional across the attention scale.

    For the exact default-SDPA pattern, rewrite the score computation to::

        bmm(q, k) * (c * c)

    so the remaining scale sits directly on the score tensor. The ordinary
    observer then sees ``scores`` and ``scores * scale`` and can determine the
    proportional qparams without backend-specific qspec manipulation.

    This reassociates floating-point operations and is not bit-equivalent in
    general. It deliberately matches only the narrow float32 default-SDPA
    pattern rather than acting as a general reassociation pass.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for bmm in list(graph.nodes):
            match = self._match_default_sdpa_score_bmm(bmm)
            if match is None:
                continue

            lhs_mul, lhs_data, rhs_mul, rhs_data, scale = match

            lhs_user = next(iter(lhs_mul.users))
            rhs_user = next(iter(rhs_mul.users))

            lhs_user.replace_input_with(lhs_mul, lhs_data)
            rhs_user.replace_input_with(rhs_mul, rhs_data)

            old_users = list(bmm.users)
            with graph.inserting_after(bmm):
                scaled_scores = graph.call_function(
                    _MUL_SCALAR,
                    args=(bmm, scale),
                )
                scaled_scores.meta = dict(bmm.meta)

            for user in old_users:
                user.replace_input_with(bmm, scaled_scores)

            assert len(lhs_mul.users) == 0
            assert len(rhs_mul.users) == 0

            graph.erase_node(lhs_mul)
            graph.erase_node(rhs_mul)

            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)

    def _match_default_sdpa_score_bmm(
        self,
        bmm: Node,
    ) -> tuple[Node, Node, Node, Node, float] | None:
        if bmm.op != "call_function" or bmm.target is not _BMM:
            return None

        if len(bmm.args) != 2 or not self._has_default_sdpa_score_consumer(bmm):
            return None

        lhs = bmm.args[0]
        rhs = bmm.args[1]

        if not isinstance(lhs, Node) or not isinstance(rhs, Node):
            return None

        lhs_match = self._trace_scaled_operand(lhs)
        rhs_match = self._trace_scaled_operand(rhs)

        if lhs_match is None or rhs_match is None:
            return None

        lhs_mul, lhs_data, lhs_scale = lhs_match
        rhs_mul, rhs_data, rhs_scale = rhs_match

        # FX Node.users tracks distinct user nodes rather than individual
        # argument occurrences. A graph such as bmm(x * c, x * c) can
        # therefore make the same scalar MUL appear single-use even though
        # both BMM operands reference it. Do not attempt to erase it twice.
        if lhs_mul is rhs_mul:
            return None

        # The shared SDPA decomposition splits one positive score scale evenly
        # across Q and K. Keep this matcher specific to that form.
        if not math.isclose(lhs_scale, rhs_scale, rel_tol=1e-12, abs_tol=0.0):
            return None

        if not (0.0 < lhs_scale <= 1.0):
            return None

        lhs_fake = get_first_fake_tensor(lhs)
        rhs_fake = get_first_fake_tensor(rhs)

        if lhs_fake.dtype != torch.float32 or rhs_fake.dtype != torch.float32:
            return None

        # aten.bmm requires [..., M, K] @ [..., K, N].
        head_dim = lhs_fake.shape[-1]
        if not isinstance(head_dim, int) or head_dim <= 0:
            return None

        expected_split_scale = float(head_dim) ** -0.25

        if not math.isclose(
            lhs_scale,
            expected_split_scale,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            return None

        combined_scale = lhs_scale * rhs_scale

        if not math.isfinite(combined_scale) or combined_scale <= 0.0:
            return None

        return (
            lhs_mul,
            lhs_data,
            rhs_mul,
            rhs_data,
            combined_scale,
        )

    @staticmethod
    def _has_default_sdpa_score_consumer(bmm: Node) -> bool:
        """Return whether BMM feeds the decomposed SDPA score softmax."""

        if len(bmm.users) != 1:
            return False

        score_view = next(iter(bmm.users))
        if (
            score_view.op != "call_function"
            or score_view.target is not _VIEW
            or len(score_view.users) != 1
        ):
            return False

        softmax = next(iter(score_view.users))
        if (
            softmax.op != "call_function"
            or softmax.target not in _SOFTMAX_TARGETS
            or len(softmax.args) < 2
        ):
            return False

        dim = softmax.args[1]
        return isinstance(dim, int) and not isinstance(dim, bool) and dim == -1

    @staticmethod
    def _trace_scaled_operand(
        bmm_input: Node,
    ) -> tuple[Node, Node, float] | None:
        node = bmm_input

        while (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target in _INPUT_PASSTHROUGH
        ):
            if len(node.users) != 1:
                return None

            if not node.args or not isinstance(node.args[0], Node):
                return None

            node = node.args[0]

        if (
            not isinstance(node, Node)
            or node.op != "call_function"
            or node.target is not _MUL_SCALAR
            or len(node.users) != 1
            or len(node.args) != 2
        ):
            return None

        data, scalar = node.args

        if not isinstance(data, Node):
            return None

        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            return None

        scalar = float(scalar)

        if not math.isfinite(scalar) or scalar <= 0.0:
            return None

        return node, data, scalar
