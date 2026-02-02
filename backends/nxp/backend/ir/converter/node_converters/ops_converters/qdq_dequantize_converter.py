# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import numpy as np

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    create_channels_last_to_channels_first_permutation,
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    set_quantization_parameters_to_tensor,
)
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator.tflite_model import Tensor
from torch.fx import Node
from torch.nn import Parameter


class QDQDequantizeConverterBase(NodeConverter, ABC):

    @abstractmethod
    def get_zero_point(self, node: Node) -> np.ndarray:
        pass

    @abstractmethod
    def get_scale(self, node: Node) -> np.ndarray:
        pass

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        zero_point_type = torch_type_to_numpy_type(node.args[-1])
        if "cluster" not in node.meta or zero_point_type not in [np.int8, np.int32]:
            return False

        return True

    def convert(self, node: Node):
        self.assert_convertible(node)

        from_tensor = self.builder.tensor_for_name(node.name)
        to_tensor = self.builder.tensor_for_name(node.args[0].name)

        scale = self.get_scale(node)
        zero_point = self.get_zero_point(node)
        quantized_dimension = 0
        if isinstance(self, QDQPerChannelDequantizeConverter):
            quantized_dimension = self.get_quantization_dimension(from_tensor, node)

        if self.context.parameters_mapping.get(node.args[0].name, None) is None:
            # Convert dequantize as identity op (Transpose that will be removed) because
            # input tensor is input of the model and don't have static data. If we do redirection
            # here we will change input name of the model.
            t_op = self._create_tflite_op_with_io_tensors(node)

            set_quantization_parameters_to_tensor(
                to_tensor, scale, zero_point, quantized_dimension
            )
            set_quantization_parameters_to_tensor(
                from_tensor, scale, zero_point, quantized_dimension
            )
            from_tensor.type = to_tensor.type

            self.builder.turn_operator_to_identity(t_op)
            self.builder.append_operators([t_op])
        else:
            # Dequantize consumes tensor with static data -> convert as a tensor
            set_quantization_parameters_to_tensor(
                to_tensor, scale, zero_point, quantized_dimension
            )

            # Change type so we pass check tensor similarity check when redirecting
            from_tensor.type = to_tensor.type
            self.builder.redirect_tensor(from_tensor, to_tensor)


class QDQPerTensorDequantizeConverter(QDQDequantizeConverterBase):

    def get_zero_point(self, node: Node) -> np.ndarray:
        zero_point_type = torch_type_to_numpy_type(node.args[5])
        return np.array(node.args[2], dtype=zero_point_type)

    def get_scale(self, node: Node) -> np.ndarray:
        return np.array(node.args[1], dtype=np.float32)


class QDQPerChannelDequantizeConverter(QDQDequantizeConverterBase):

    def get_zero_point(self, node: Node) -> np.ndarray:
        return self.context.parameters_mapping[node.args[2].name].numpy()

    def get_scale(self, node: Node) -> np.ndarray:
        return self.context.parameters_mapping[node.args[1].name].numpy()

    def get_quantization_dimension(self, from_tensor: Tensor, node: Node) -> int:
        quantization_dimension = node.args[3]

        # Quantization dimension is affected by tensor format
        if from_tensor.tensor_format == TensorFormat.CHANNELS_LAST:
            tensor_rank = len(from_tensor.shape.vector)
            perm = create_channels_last_to_channels_first_permutation(
                tensor_rank, return_list=True
            )
            quantization_dimension = perm[quantization_dimension]
        return quantization_dimension
