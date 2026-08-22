# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx import Node

FROM_NODE_KEY = "from_node"
DEBUG_HANDLE_KEY = "debug_handle"

UNSET_DEBUG_HANDLE = 0


def get_greatest_ancestor_node_identifier(node: Node) -> str:
    """Get the identifier of the greatest ancestor node of the given node.

    The identifier is the concatenation of the node name and graph id of the
    greatest ancestor node, where the graph id is the unique id for every graph
    module in the export flow and node name is unique within the same graph module.
    """

    node_source = node.meta[FROM_NODE_KEY]
    node_source = node_source[-1]

    while len(node_source.from_node) > 0:
        node_source = node_source.from_node[-1]

    return f"{node_source.name}.{str(node_source.graph_id)}"
