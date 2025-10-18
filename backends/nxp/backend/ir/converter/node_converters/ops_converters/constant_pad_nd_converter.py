# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Collection

import numpy as np

from executorch.backends.nxp.backend.edge_helper import input_rank
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    apply_permutation_to,
    create_channels_first_to_channels_last_permutation,
    tf_lite_type_to_numpy,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    quantize_int8,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    pad_options,
    pad_v2_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class ConstantPadNDConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        paddings = node.args[1]
        if len(paddings) > 4 and paddings[4:6] != [0, 0]:
            # Attempt to Pad channels dimension, which is not supported on Neutron.
            return False

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        paddings = node.args[1]

        # https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/PadNd.cpp#L38-L40
        if len(paddings) > (input_rank(node, 0) * 2):
            return False

        # https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/PadNd.cpp#L30-L31
        if len(paddings) % 2 != 0:
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        if len(paddings) > 4 and paddings[4:6] != [0, 0]:
            # Attempt to Pad channels dimension -> currently not supported
            return False

        return True

    # noinspection PyMethodMayBeStatic
    def _convert_paddings_to_tflite(
        self, paddings: Collection[int], input_tensor: tflite_model.Tensor
    ) -> list[int]:
        """Convert the PyTorch paddings to TFLite paddings.
            The PyTorch padding is added to the individual dimensions from the back (slightly confusing), see:
             https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
            TFLite padding has shape [input_rank, 2], where start padding and end padding is specified for every
             corresponding dimension.

        :param paddings: The PyTorch paddings.
        :param input_tensor: Main input tensor of the `aten.constant_pad_nd` operator.
        :return: The equivalent TFLite paddings.
        """

        # 1st, group the individual paddings into groups of 2 (padding at the start and at the end for every dimension).
        paddings = np.array(paddings).reshape(-1, 2)

        # 2nd, reverse the padding groups. (The order is inverse between PyTorch and TFLite).
        paddings = list(reversed(paddings))

        # 3rd, add [0, 0]s from the start to get `rank` padding groups.
        paddings = [[0, 0]] * (input_tensor.rank - len(paddings)) + paddings

        if input_tensor.tensor_format.is_channels_last():
            # Permute the `tfl_paddings` to match.
            to_tflite_perm = create_channels_first_to_channels_last_permutation(
                input_tensor.rank
            )
            paddings = apply_permutation_to(paddings, to_tflite_perm)

        return paddings

    def convert(self, node: Node):
        """Convert the `aten.constant_pad_nd` operator to TFLite `PadV2`."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        paddings = node.args[1]
        constant = node.args[2]

        paddings = self._convert_paddings_to_tflite(paddings, x)
        paddings_tensor = self.builder.create_tensor_for_data(
            np.asarray(paddings, "int32"), "paddings"
        )

        if constant == 0.0:
            # We're padding with zeros, we can use traditional Pad op
            t_op.tmp_inputs = [x, paddings_tensor]
            t_op.tmp_outputs = [y]
            t_op.builtin_options = pad_options.Pad()

            self.builder.append_operators([t_op])
            return

        if x.quantization is None:
            constant_tensor = self.builder.create_tensor_for_data(
                np.array([constant], tf_lite_type_to_numpy(x.type)), "constant"
            )
        else:
            quantization = copy.copy(x.quantization)
            scale, zero_point = (
                quantization.scale.vector,
                quantization.zero_point.vector,
            )
            constant_data = quantize_int8(
                np.array([constant], np.float32), scale, zero_point
            )
            constant_tensor = self.builder.create_tensor_for_data(
                constant_data, "constant"
            )
            constant_tensor.quantization = quantization

        # Assign the operator its TFLite inputs and outputs.
        t_op.tmp_inputs = [x, paddings_tensor, constant_tensor]
        t_op.tmp_outputs = [y]
        t_op.builtin_options = pad_v2_options.PadV2()

        self.builder.append_operators([t_op])
