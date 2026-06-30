# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import node_has_well_defined_shape
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
    requires_channels_first_format,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.resize_nearest_neighbor_options import (
    ResizeNearestNeighbor,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter

HeightScale = float
WidthScale = float


# noinspection SpellCheckingInspection
@requires_channels_first_format
class UpsampleNearest2DConverter(NodeConverter):

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        h_scale, w_scale = cls._get_effective_scales(node)
        is_alone_in_partition = cls.is_node_alone_in_partition(node, partition_list)

        if is_alone_in_partition and h_scale == w_scale == 1:
            # The operator is a no-op, so the Neutron Converter will skip it. If it's the only node in the
            #  partition, the graph would end up empty.
            return False

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        # The conversion requires the output shape to be known and static.
        if not node_has_well_defined_shape(node):
            return False

        if len(node.meta["val"].shape) != 4:
            # Unexpected case. The input should always be 4D.
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        # The tensors are always 4D and use the channels first format (NCHW).
        _, in_c, in_h, in_w = node.all_input_nodes[0].meta["val"].shape
        _, _, out_h, out_w = node.meta["val"].shape

        if not NodeConverter.uses_quantization_type_for_io(
            node,
            supported_types=[torch.int8, torch.uint8],
            input_indices=[0],
            output_indices=[0],
        ):
            return False

        supported_scales = [1, 2, 4, 8]
        h_scale, w_scale = UpsampleNearest2DConverter._get_effective_scales(node)
        # The H and W scales don't need to be equal but both must be supported.
        if (h_scale not in supported_scales) or (w_scale not in supported_scales):
            return False

        return True

    @staticmethod
    def _get_effective_scales(node: Node) -> tuple[HeightScale, WidthScale]:
        # Neutron supports variants where `align_corners=False` and `align_corners=True`. ExecuTorch doesn't have this
        #  parameter. Its behavior is equivalent to `align_corners=False`. Hence, the scale calculation corresponds to
        #  the `align_corners=False` case in the Neutron documentation.
        _, _, in_h, in_w = node.all_input_nodes[0].meta["val"].shape
        _, _, out_h, out_w = node.meta["val"].shape
        h_scale = out_h / in_h
        w_scale = out_w / in_w

        return h_scale, w_scale

    def convert(self, node: Node):
        """Convert the `aten.upsample_nearest2d.vec` operator to Neutron IR `ResizeNearestNeighbor`.
        The ExecuTorch schema is:
            aten::upsample_nearest2d.vec(
                Tensor input,
                SymInt[]? output_size,
                float[]? scale_factors
            ) -> Tensor
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        # Neutron supports variants where `align_corners=False` and `align_corners=True`. ExecuTorch doesn't have this
        #  parameter. Its behavior is equivalent to `align_corners=False` and `half_pixel_centers=False`.
        t_op.builtin_options = ResizeNearestNeighbor(False, False)

        # The `aten.upsample_nearest2d` can use either the `size` attribute or the `scale_factor` to define the output
        #  size. The Neutron IR `ResizeNearestNeighbor` only supports the `sizes` (output spatial dimensions).
        # Both `size` and `scale_factor` can be easily supported by extracting the output spatial size from the output
        #  tensor's shape and using it as the `sizes`.
        # The `self.assert_convertible(node)` call guarantees that the shape is 4D, channels last (NHWC), and static.
        _, out_h, out_w, _ = y.shape
        sizes = self.builder.create_tensor_for_data(
            np.array([out_h, out_w], np.int32), "sizes"
        )

        t_op.tmp_inputs = [x, sizes]  # Assign the NeutronIR inputs.

        self.builder.append_operators([t_op])
