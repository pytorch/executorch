# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    create_channels_last_to_channels_first_permutation,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared.reduce_utils import (
    convert_axes_from_attribute,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    mean_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.backend.node_format import NXP_NODE_FORMAT
from torch.fx import Node
from torch.nn import Parameter


class MeanDimConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        keepdim = node.args[2] if len(node.args) >= 3 else False
        rank = len(node.args[0].meta["val"].shape)
        dim = [MeanDimConverter._to_pos_dim(d, rank) for d in node.args[1]]

        if rank != 4 or not keepdim:
            # neutron-converter/src/OperatorC/GlobalAvgPoolPlugin.cpp#74-77
            return False

        # The `mean.dim` gets converted to AveragePool by the NeutronConverter, so the channels must be a
        #  multiple of `num_macs`.
        # neutron-converter/src/OperatorC/GlobalAvgPoolPlugin.cpp#59-85
        num_macs = neutron_target_spec.get_num_macs()
        channels_dim = 1 if node.meta[NXP_NODE_FORMAT].is_channels_first() else -1
        if (node.meta["val"].shape[channels_dim] % num_macs) != 0:
            return False

        # Neutron only supports reduction over the spatial dimensions H, W.
        if node.meta[NXP_NODE_FORMAT].is_channels_first():
            # The input is NCHW. H and W are at indices 2 and 3.
            if dim not in [[2, 3], [3, 2]]:
                return False
        else:
            # The input is formatless. It can be considered as NHWC, as this is the way Neutron will look at
            #  the dimensions. So H and W are the middle dimensions.
            if dim not in [[1, 2], [2, 1]]:
                return False

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if hasattr(node.kwargs, "dtype") and node.kwargs["dtype"] not in [
            torch.float32,
            torch.uint32,
            torch.uint8,
        ]:
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    @staticmethod
    def _to_pos_dim(d: int, rank: int):
        return d + rank if d < 0 else d

    @staticmethod
    def _normalize_and_to_channel_last_dim(dim: list[int], rank: int) -> list[int]:
        # convert negative index to positive
        dim = [MeanDimConverter._to_pos_dim(d, rank) for d in dim]

        perm = create_channels_last_to_channels_first_permutation(rank, True)
        dim = [perm[d] for d in dim]

        return dim

    # Mean Dim Node format: (Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None)
    def convert(self, node: Node):
        """Convert 'mean.dim' operator to TFLite 'Mean'."""
        self.assert_convertible(node)

        dim = node.args[1]
        keepdim = node.args[2] if len(node.args) >= 3 else False

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = mean_options.Mean(keepdim)
        x = t_op.tmp_inputs[0]

        if x.tensor_format.is_channels_last():
            dim = self._normalize_and_to_channel_last_dim(dim, x.rank)

        convert_axes_from_attribute(t_op, self.builder, dim)
        self.builder.append_operators([t_op])
