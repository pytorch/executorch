# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

# Key into the `meta` attribute of nodes, which is mapped to their inferred node format.
NXP_NODE_FORMAT = "nxp_node_format"


class NodeFormat(Enum):
    # Node's output in NCHW format
    CHANNELS_FIRST = 0

    # Node's output format has no meaning
    FORMATLESS = 1

    # Format has not been identified
    NONE = 2

    def is_channels_first(self) -> bool:
        return self == NodeFormat.CHANNELS_FIRST
