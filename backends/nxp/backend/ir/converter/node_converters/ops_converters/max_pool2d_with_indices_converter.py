# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import try_get_arg
from executorch.backends.nxp.backend.ir.converter.conversion import (
    aten_translator,
    common,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.max_pool_2d_options import (
    MaxPool2D,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter

KernelSize = tuple[int, int]
Stride = tuple[int, int]
Padding = tuple[int, int]
Dilation = tuple[int, int]
CeilMode = bool


class MaxPool2DWithIndicesConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        kernel_size, stride, padding, dilation, ceil_mode = (
            MaxPool2DWithIndicesConverter._get_node_args(node)
        )

        if dilation != (1, 1):
            # The Neutron IR MaxPool2D does not support dilation.
            return False

        if ceil_mode:
            # This argument affects how the output shape is computed. Neutron IR only supports the default `False`.
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        # The second output cannot be represented in Neutron IR. If it's used, do not delegate.
        getitem_nodes = list(node.users)
        if any(n.args[1] == 1 for n in getitem_nodes if n.target == operator.getitem):
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        kernel_size, stride, padding, dilation, ceil_mode = (
            MaxPool2DWithIndicesConverter._get_node_args(node)
        )

        if custom_delegation_options.use_new_flow_neutron_c:
            # Requirements specified by the new Neutron flow documentation.

            supported_types = [torch.int8, torch.uint8]
            if not NodeConverter.uses_quantization_type_for_io(
                node, supported_types, [0], [0]
            ):
                return False

            maximum_supported_kernel_size = 4096
            # If there is no padding, Neutron allows maximum stride of 4096. Otherwise, it's 32. But the converter
            #  always inserts a `Pad` operator to add the padding, so the `MaxPool` never pads it's input itself, so
            #  4096 is always the limit. And similarly, the `MaxPool` input padding limitation does not apply either.
            maximum_supported_stride = 4096

            if any(k > maximum_supported_kernel_size for k in kernel_size):
                return False
            if any(s > maximum_supported_stride for s in stride):
                return False

        else:
            # Shape of the main output (index 0)
            output_shape = node.meta["val"][0].shape
            if output_shape[0] != 1:
                # /neutron-converter/src/OperatorC/MaxPoolPlugin.cpp?at=NEUTRON_SOFTWARE_2.2.2#106
                return False

            # Neutron only has a restriction on `stride_h`. `stride_w` is not restricted.
            stride_h = stride[0]
            if stride_h not in (1, 2):
                # /neutron-library/src/utils/NeutronLibraryInterrogation.cpp?at=refs%2Ftags%2FNEUTRON_SOFTWARE_2.2.2#901
                # /neutron-library/src/utils/NeutronLibraryInterrogation.cpp?at=refs%2Ftags%2FNEUTRON_SOFTWARE_2.2.2#923
                return False

            channels = output_shape[1]
            if channels % neutron_target_spec.get_num_macs() != 0:
                # /neutron-library/src/utils/NeutronLibraryInterrogation.cpp?at=refs%2Ftags%2FNEUTRON_SOFTWARE_2.2.2#903
                # /neutron-library/src/utils/NeutronLibraryInterrogation.cpp?at=refs%2Ftags%2FNEUTRON_SOFTWARE_2.2.2#925
                return False

            if any(pad > kernel_dim for pad, kernel_dim in zip(padding, kernel_size)):
                # /neutron-library/src/utils/NeutronLibraryInterrogation.cpp?at=refs%2Ftags%2FNEUTRON_SOFTWARE_2.2.2#904-907
                # /neutron-library/src/utils/NeutronLibraryInterrogation.cpp?at=refs%2Ftags%2FNEUTRON_SOFTWARE_2.2.2#926-929

                # Cannot be tested as PyTorch crashes in this case. It requires the padding to be at most half of the
                #  effective kernel size, which is an even stricter requirement than what Neutron imposes.
                # https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/_meta_registrations.py#L4483-L4489
                return False

        return True

    @staticmethod
    def _get_pad_constant_value(input_type: TensorType) -> np.ndarray:
        """Get scalar NumPy array with constant value used as constant value for 'Pad' operator.

        :param input_type: Input tensor type.
        :return: Scalar array with single minimum value of given type.
        """

        match input_type:
            case TensorType.INT8:
                return np.asarray([np.iinfo(np.int8).min], dtype=np.int8)
            case TensorType.UINT8:
                return np.asarray([np.iinfo(np.uint8).min], dtype=np.uint8)
            case TensorType.FLOAT32:
                return np.asarray([np.finfo(np.float32).min], dtype=np.float32)
            case _:
                # Should never happen.
                raise RuntimeError(
                    f"Unexpected input type '{input_type}' for MaxPool operator."
                )

    @staticmethod
    def _get_node_args(
        node: Node,
    ) -> tuple[KernelSize, Stride, Padding, Dilation, CeilMode]:
        """Extract and return `aten.max_pool2d_with_indices` arguments from the node.

        :param node: The node representing the `aten.max_pool2d_with_indices` operation.
        :return: Tuple of (kernel_size, stride, padding, dilation, ceil_mode).
        """
        kernel_size = node.args[1]
        stride = node.args[
            2
        ]  # The default value is equal to the kernel_size, so it is never empty here.
        padding = try_get_arg(node, 3) or (0, 0)
        dilation = try_get_arg(node, 4) or (1, 1)
        ceil_mode = try_get_arg(node, 5) or False

        return kernel_size, stride, padding, dilation, ceil_mode

    def convert(self, node: Node):
        """Convert the `aten.max_pool2d_with_indices.default` operator to Neutron IR `MaxPool2D`.
        The schema is:
        aten::max_pool2d_with_indices(
            Tensor self,
            int[2] kernel_size,
            int[2] stride=[],   # The default value is equal to the kernel_size.
            int[2] padding=0,
            int[2] dilation=1,
            bool ceil_mode=False
        ) -> (Tensor, Tensor)

        It produces 2 output tensors:
            1. The first one contains the maximum values selected by the kernel.
            2. The second one contains the indices of the selected values.

        The second output tensor cannot be represented in Neutron IR. So the operator is only supported when the second
         output is unused.
        """
        self.assert_convertible(node)

        kernel_size, stride, padding, dilation, ceil_mode = self._get_node_args(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        ops = OpsList(middle_op=t_op)

        x = t_op.tmp_inputs[0]

        t_op.builtin_options = MaxPool2D()
        t_op.builtin_options.filter_h, t_op.builtin_options.filter_w = kernel_size
        common.assign_2d_strides(t_op.builtin_options, stride)

        t_op.builtin_options.padding, explicit_padding = (
            aten_translator.convert_padding(list(padding))
        )
        if explicit_padding is not None:
            # Need to prepend a 'Pad' operator, which adds min values for type.
            constant_value = self._get_pad_constant_value(x.type)
            pad_op = self.builder.create_pad_operator_before(
                t_op, 0, explicit_padding, constant_value=constant_value
            )
            ops.add_pre(pad_op)

        # The second output of the operator cannot be represented in NeutronIR. The `_is_supported_in_IR()` method
        #  ensures the second output is never used in the model, so it can be safely removed here.
        t_op.tmp_outputs[1:] = []

        self.builder.append_operators(ops.flatten())
