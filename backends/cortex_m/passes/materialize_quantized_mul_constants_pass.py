# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes import ComputeConstantOpsAOTPass
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node


class MaterializeQuantizedMulConstantsPass(ComputeConstantOpsAOTPass):
    """Precompute only quantized scalar constants that feed Cortex-M MUL.

    The generic constant-materialization pass also folds unrelated operators
    such as ``arange``. Cortex-M only needs the ``aten.full`` constants created
    for scalar MULs here, after qparams have been attached, so keep the scope
    limited to those nodes.
    """

    targeted_ops = [exir_ops.edge.aten.full.default]

    _passthrough_ops = {
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }

    @classmethod
    def _feeds_single_quantized_mul(cls, node: Node) -> bool:
        current = node
        visited: set[Node] = set()

        while True:
            users = list(current.users)
            if len(users) != 1:
                return False

            user = users[0]
            if user in visited:
                return False
            visited.add(user)

            if user.target == exir_ops.edge.aten.mul.Tensor:
                return bool(user.meta.get("input_qparams")) and bool(
                    user.meta.get("output_qparams")
                )

            if user.target not in cls._passthrough_ops:
                return False

            current = user

    def compute_node_aot(self, node: Node) -> bool:
        output_qparams = node.meta.get("output_qparams", {})
        if set(output_qparams) != {0}:
            return False
        if not self._feeds_single_quantized_mul(node):
            return False
        return super().compute_node_aot(node)
