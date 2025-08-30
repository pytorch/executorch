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
    if "memory_format" in node.kwargs.keys():
        return node.kwargs["memory_format"] == torch.preserve_format

    return True


class CloneConverter(NodeConverter):

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
