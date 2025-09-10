# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.edge_helper import input_rank
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    NodeConverter,
    Target,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    softmax_options,
)
from torch.fx import Node
from torch.nn import Parameter


class SoftmaxConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        target: Target,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        match target:
            case Target.RT700:
                # The eIQ Neutron NPU runtime software has a known issue with the SoftMax operation.
                #  As long as the issue is present, return False for the i.MX RT700 target also.
                return False

            case _:
                return False

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        # The IR only supports the `dim` as the last dimension. But that depends on the format of the input tensor,
        #  which is only known after the `Partitioner` has divided the model. So if the input shape can be channels
        #  first (i.e. is more than 2D), we cannot determine IR support (we assume it's not supported).
        x_rank = input_rank(node, 0)
        if x_rank > 2:
            return False

        dim = SoftmaxConverter._normalize_dim(node.args[1], x_rank)
        if dim != x_rank - 1:
            return False

        return True

    @staticmethod
    def _normalize_dim(dim, rank):
        # convert negative index to positive
        if dim < 0:
            dim += rank
        return dim

    def convert(self, node: Node):
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        t_op.builtin_options = softmax_options.Softmax(beta=1.0)

        self.builder.append_operators([t_op])
