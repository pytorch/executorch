# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.data_format import NXP_NODE_FORMAT
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    softmax_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class SoftmaxConverter(NodeConverter):

    @staticmethod
    def _get_channels_dim(node: Node) -> int:
        """Get the dimension index for channels, based on data format.
        :return: 1 for the channels_first format (NCHW), rank-1 for the channels_last format (NHWC).
        """
        rank = len(node.meta["val"].shape)
        return 1 if node.meta[NXP_NODE_FORMAT].is_channels_first() else rank - 1

    @staticmethod
    def _get_spatial_dims(node: Node) -> list[int]:
        """Extract spatial dimensions from the node's input shape.
        Returns a list with [N, H, W] (or equivalent for other ranks).
        """
        input_shape = list(node.meta["val"].shape)
        if node.meta[NXP_NODE_FORMAT].is_channels_first():
            # NCHW: skip the channel dimension at index 1
            return [input_shape[0]] + input_shape[2:]
        else:
            # NHWC: skip the last dimension
            return input_shape[:-1]

    @staticmethod
    def _get_total_spatial_size(node: Node) -> int:
        """Calculate total spatial size (product of all spatial dimensions)."""
        return int(np.prod(SoftmaxConverter._get_spatial_dims(node)))

    @staticmethod
    def _get_channels(node: Node) -> int:
        """Get the number of channels from the node's input shape."""
        return node.meta["val"].shape[SoftmaxConverter._get_channels_dim(node)]

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        """Check if the softmax operation can be executed on Neutron hardware.

        Hardware constraints:
        1. Input rank must be >= 2 (Neutron does not support 1D)
        2. Channels must be a multiple of num_macs
        3. Channels < 4096 / num_pipes * 4
        4. Total spatial size (N*H*W) <= 4096
        5. (channels * spatial_size) / num_macs <= 65536
        """
        input_shape = node.meta["val"].shape

        # Constraint 1: Neutron does not support 1D SoftMax
        if len(input_shape) == 1:
            return False

        num_macs = neutron_target_spec.get_num_macs()
        num_pipes = neutron_target_spec.get_num_pipes()
        channels = SoftmaxConverter._get_channels(node)
        total_spatial_size = SoftmaxConverter._get_total_spatial_size(node)

        # Constraint 2: Channels must be a multiple of num_macs
        if channels % num_macs != 0:
            return False

        # Constraint 3: Channel size limit
        if channels >= 4096 / num_pipes * 4:
            return False

        # Constraint 4: Spatial size limit
        if total_spatial_size > 4096:
            return False

        # Constraint 5: Total processing size limit
        if channels * total_spatial_size / num_macs > 65536:
            return False

        return True

    @staticmethod
    def _normalize_dim(dim: int, rank: int) -> int:
        """Make sure the dimension index `dim` is positive.
        :arg dim: The dimension index (can be negative)
        :arg rank: The total number of dimensions

        :return: Positive dimension index
        """
        return dim % rank

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        """Check if the softmax operation is supported in NeutronIR.
        NeutronIR only supports softmax along the channels dimension.
        """
        dim = SoftmaxConverter._normalize_dim(node.args[1], len(node.meta["val"].shape))

        # NeutronIR only supports the `dim` as the channels dimension
        channels_dim = SoftmaxConverter._get_channels_dim(node)
        if dim != channels_dim:
            return False

        half_to_float = node.args[2] if len(node.args) > 2 else False
        if half_to_float:
            # This argument states that the Softmax has a float16 input and output, but the computation is done in
            #  float32. Neutron doesn't support float16 quantization, so this case should never happen.
            raise ValueError(
                f"Softmax node `{node}` has `half_to_float = True`, which is not supported. "
                "There is an issue with the NXP backend. Please report this."
            )

        return True

    def convert(self, node: Node):
        """Convert `aten._softmax.default` node to NeutronIR.
        The schema is:
        aten::_softmax(
            Tensor self,
            int dim,
            bool half_to_float
        ) -> Tensor
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = softmax_options.Softmax(beta=1.0)

        self.builder.append_operators([t_op])
