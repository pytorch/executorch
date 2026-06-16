# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.data_format import DataFormat, NXP_NODE_FORMAT
from executorch.backends.nxp.backend.edge_helper import node_has_well_defined_shape
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    is_not_qdq_node,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.resize_bilinear_options import (
    ResizeBilinear,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter


# noinspection SpellCheckingInspection
class UpsampleBilinear2DConverter(NodeConverter):

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        input_shape = node.all_input_nodes[0].meta["val"].shape
        output_shape = node.meta["val"].shape
        is_alone_in_partition = cls.is_node_alone_in_partition(
            node, partition_list, filter_fn=is_not_qdq_node
        )

        if is_alone_in_partition and input_shape == output_shape:
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

        if node.meta.get(NXP_NODE_FORMAT, DataFormat.NONE) != DataFormat.CHANNELS_FIRST:
            # This should never happen.
            raise NotImplementedError(
                "NXP backend: `aten.upsample_bilinear2d.vec` didn't have correctly identified data"
                " format. Please report this."
            )

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
        align_corners = node.args[2]
        if align_corners:
            if in_h == 1 or in_w == 1:
                return False  # Avoid division by 0.
            h_scale = (out_h - 1) / (in_h - 1)
            w_scale = (out_w - 1) / (in_w - 1)
        else:
            h_scale = out_h / in_h
            w_scale = out_w / in_w

        # The H and W scales don't need to be equal, but both must be supported.
        if (h_scale not in supported_scales) or (w_scale not in supported_scales):
            return False

        return True

    def convert(self, node: Node):
        """Convert the `aten.upsample_bilinear2d.vec` operator to Neutron IR `ResizeBilinear`.
        The ExecuTorch schema is:
        aten::upsample_bilinear2d.vec(
            Tensor input,
            SymInt[]? output_size,
            bool align_corners,
            float[]? scale_factors
        ) -> Tensor
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        # ExecuTorch has 1 paramter (align_corners). NeutronIR has 2 parameters (align_corners, half_pixel_centers).
        # In ExecuTorch, the pixel compute scale is:
        #  `(input_size - 1) / (output_size - 1)` if align_corners else `input_size / output_size`
        # https://github.com/pytorch/executorch/blob/v1.1.0/kernels/portable/cpu/util/upsample_util.h#L65
        # https://github.com/pytorch/executorch/blob/v1.1.0/kernels/portable/cpu/util/upsample_util.h#L52
        # The source index is the computed as:
        #  `scale * dst_idx` if align_corners else `scale * (dst_idx + 0.5) - 0.5`
        # https://github.com/pytorch/executorch/blob/v1.1.0/kernels/portable/cpu/util/upsample_util.h#L81-L87
        #
        # So combined:
        # if align_corners:
        #     src_idx = dst_idx * (input_size - 1) / (output_size - 1)
        # else:
        #     src_idx = (dst_idx + 0.5) * input_size / output_size - 0.5
        #
        # The first equation is exactly what NeutronIR uses when `align_corners == True and half_pixel_centers == False`
        #  and the second one is what NeutronIR uses when `align_corners == False and half_pixel_centers == True`.
        # https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/lite/kernels/internal/reference/resize_bilinear.h#L82-L88
        # https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/lite/kernels/internal/reference/resize_bilinear.h#L172-L180
        # Also, the new Neutron flow requires that `align_corners` and `half_pixel_centers` are not True simultainiously.
        align_corners = node.args[2]
        half_pixel_centers = not align_corners
        t_op.builtin_options = ResizeBilinear(align_corners, half_pixel_centers)

        # The `aten.upsample_bilinear2d` can use either the `size` attribute or the `scale_factor` to define the output
        #  size. The Neutron IR `ResizeBilinear` only supports the `sizes` (output spatial dimensions).
        # Both `size` and `scale_factor` can be easily supported by extracting the output spatial size from the output
        #  tensor's shape and using it as the `sizes`.
        # The `self.assert_convertible(node)` call guarantees that the shape is 4D, channels last (NHWC), and static.
        _, out_h, out_w, _ = y.shape
        sizes = self.builder.create_tensor_for_data(
            np.array([out_h, out_w], np.int32), "sizes"
        )

        t_op.tmp_inputs = [x, sizes]  # Assign the NeutronIR inputs.

        self.builder.append_operators([t_op])
