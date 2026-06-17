# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.edge_helper import (
    input_rank,
    node_is_effectively_static_tensor,
    weights_are_effectively_static,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    fully_connected_options,
)

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


# The edge operator signature is: aten.addmm(bias, input, weight, *, beta=1, alpha=1)
MAIN_INPUT_IDX = 1
WEIGHT_IDX = 2
BIAS_IDX = 0


class AddMMConverter(NodeConverter):
    """Convert the `aten.addmm` operator to TFLite `FullyConnected` with a bias input."""

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if len(node.all_input_nodes) != 3:
            return False

        # The weights must be 2D.
        if input_rank(node, WEIGHT_IDX) != 2:
            return False

        alpha, beta = node.kwargs.get("alpha", 1), node.kwargs.get("beta", 1)
        if alpha != 1 or beta != 1:
            # As these cases seem rare, conversion is not implemented for the time being.
            return False

        # The `aten.addmm` operator allows any bias shape, as long as it is broadcastable with the result of the matrix
        #  multiplication. That means it supports 4 different shapes: [N, P], [1, P], [P], [1] (provided the MM result
        #  has shape [N, P]). Out of these 4, Neutron IR allows only [1, P] and [P], both of which are supported on
        #  Neutron.
        bias_shape = list(node.args[BIAS_IDX].meta["val"].shape)
        _, p = node.meta["val"].shape
        if bias_shape not in [[1, p], [p]]:
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        # Main input and output must be `int8` or `uint8`.
        if not NodeConverter.uses_quantization_type_for_io(
            node, [torch.int8, torch.uint8], [MAIN_INPUT_IDX], [0]
        ):
            return False

        # Weights must be `int8`.
        if not NodeConverter.uses_quantization_type_for_io(
            node, [torch.int8], [WEIGHT_IDX], []
        ):
            return False

        # Bias must be `int32`.
        if not NodeConverter.uses_quantization_type_for_io(
            node, [torch.int32], [BIAS_IDX], []
        ):
            return False

        # Weights must be constant.
        if not weights_are_effectively_static(
            node, parameters_mapping, weight_index=WEIGHT_IDX
        ):
            return False

        # The bias must be constant.
        if not node_is_effectively_static_tensor(
            node.args[BIAS_IDX], parameters_mapping
        ):
            return False

        return True

    def convert(self, node: Node):
        """Convert the `aten.addmm` operator to NeutronIR `FullyConnected`.
        The schema is:
            addmm(
                Tensor self,
                Tensor mat1,
                Tensor mat2,
                *,
                Scalar beta=1,
                Scalar alpha=1
            ) -> Tensor
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = fully_connected_options.FullyConnected(
            keep_num_dims=True
        )

        bias = t_op.tmp_inputs[0]
        x = t_op.tmp_inputs[1]
        w = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        # Assign the operator its Neutron IR inputs and outputs
        t_op.tmp_inputs = [x, w, bias]
        t_op.tmp_outputs = [y]

        ops = OpsList(middle_op=t_op)

        # The `aten.addmm` uses main input with shape [M, N] and the weights have the shape [N, O].
        # Neutron IR `FullyConnected` requires the weights to have shape [O, N] (if the main input has shape [M, N]).
        # Insert a `Transpose` operator to permute the weights to achieve correct conversion. (The `Transpose` will not
        #  be present in the output model if the weights are static.)
        ops.add_pre(self.builder.create_transpose_operator_before(t_op, 1, [1, 0]))

        self.builder.append_operators(ops.flatten())
