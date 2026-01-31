# Copyright 2023 Martin Pavella
# Copyright 2023-2026 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from enum import Enum

# Key into the `meta` attribute of nodes, which is mapped to their inferred node format.
NXP_NODE_FORMAT = "nxp_node_format"


class DataFormat(Enum):
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
        return self == DataFormat.CHANNELS_FIRST

    def is_channels_last(self) -> bool:
        return self == DataFormat.CHANNELS_LAST

    @staticmethod
    def convert_executorch_format_to_neutron(
        executorch_format: "DataFormat",
    ) -> "DataFormat":
        if executorch_format == DataFormat.CHANNELS_FIRST:
            return DataFormat.CHANNELS_LAST  # Format is converted.

        else:
            return executorch_format  # Other formats remain unchanged.
