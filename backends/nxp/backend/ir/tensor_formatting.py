# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
from enum import Enum

from executorch.backends.nxp.backend.node_format_inference import NodeFormat


class TensorFormat(Enum):
    CHANNELS_FIRST = 0

    CHANNELS_LAST = 10

    # The format of TFLite Conv3D weights tensor: [output_channels, input_channels, D, H, W]
    CONV_3D_WEIGHT_FORMAT = 11

    # Intermediate format between 'Transpose' and 'Reshape' ops when single dimension with value 1
    # is added/removed via reshaping
    RESHAPE_SINGLE_UNITARY_TRANSPOSITION = 12

    # The format of TFLite TransposeConv 2D weights tensor: [M/group, kH, kW, C]
    TRANSPOSE_CONV_2D_WEIGHT_FORMAT = 13

    # No special format (matrices, vectors, shapes etc.). All tensors with the FORMATLESS format MUST have EXACTLY
    #  the same shape and data in the NeutronIR model and in the ExecuTorch model.
    FORMATLESS = 20

    NONE = 30  # Format has not been identified

    def is_channels_first(self) -> bool:
        return self == TensorFormat.CHANNELS_FIRST

    def is_channels_last(self) -> bool:
        return self == TensorFormat.CHANNELS_LAST

    @staticmethod
    def from_node_format(node_format: NodeFormat):
        if node_format.is_channels_first():
            return TensorFormat.CHANNELS_LAST
        elif node_format == NodeFormat.FORMATLESS:
            return TensorFormat.FORMATLESS
        else:
            return TensorFormat.NONE

    def to_node_format(self):
        if self == TensorFormat.CHANNELS_LAST:
            return NodeFormat.CHANNELS_FIRST
        elif self == TensorFormat.FORMATLESS:
            return NodeFormat.FORMATLESS
        else:
            return NodeFormat.NONE
