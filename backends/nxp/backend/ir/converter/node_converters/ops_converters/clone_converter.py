# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from torch.fx import Node
from torch.nn import Parameter


def _has_supported_memory_format(node: Node) -> bool:
    """The node can either represent an `aten.clone` or a `dim_order_ops._clone_dim_order` operator."""
    memory_format = node.kwargs.get("memory_format", None)  # Attribute of `aten.clone`.
    dim_order = node.kwargs.get(
        "dim_order", None
    )  # Attribute of `dim_order_ops._clone_dim_order`.

    if (memory_format, dim_order) == (torch.preserve_format, None):
        # The operator does nothing (e.g. originated as a `Dropout`).
        return True

    contiguous_dim_order = list(range(len(node.meta["val"].shape)))
    if (memory_format, dim_order) in [
        (torch.contiguous_format, None),
        (None, contiguous_dim_order),
    ]:
        # Sometimes there is a `permute_copy` (Transpose) in Executorch, which doesn't actually permute the data in
        #  memory. Instead, it just changes the `strides` (memory format) to match the permutation. Then, some
        #  following operator may or may not support the particular strides (e.g. `mul` supports anything but
        #  `view_copy` does not), so the `clone` may be inserted to actually permute the data in memory to the
        #  `contiguous` format. This is purely an Executorch issue, and there is no equivalent system in NeutronIR.
        #  In NeutronIR, every tensor is stored in memory exactly as its shape suggests. Therefore, the `clone` can
        #  simply be omitted.
        return True

    return False


class CloneConverter(NodeConverter):
    """
    This converter is responsible for converting both edge operators:
    - aten.clone.default
    - dim_order_ops._clone_dim_order.default
    """

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        return _has_supported_memory_format(node)

    def convert(self, node: Node):
        """Skip `aten.clone` operator if it has no `memory_format` specified."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        self.builder.turn_operator_to_identity(t_op)
        self.builder.append_operators([t_op])
