# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Mapping, Optional

from executorch.sdk.debug_format.base_schema import OperatorNode

from executorch.sdk.debug_format.et_schema import FXOperatorGraph, OperatorGraph
from executorch.sdk.etdump.schema_flatcc import ETDumpFlatCC

from executorch.sdk.etdump.serialize import deserialize_from_etdump_flatcc
from executorch.sdk.etrecord import ETRecord

EDGE_DIALECT_GRAPH_KEY = "edge_dialect_graph_module"


def gen_graphs_from_etrecord(
    etrecord: ETRecord,
) -> Mapping[str, OperatorGraph]:
    op_graph_map = {}
    if etrecord.graph_map is not None:
        op_graph_map = {
            name: FXOperatorGraph.gen_operator_graph(exported_program.graph_module)
            for name, exported_program in etrecord.graph_map.items()
        }
    if etrecord.edge_dialect_program is not None:
        op_graph_map[EDGE_DIALECT_GRAPH_KEY] = FXOperatorGraph.gen_operator_graph(
            etrecord.edge_dialect_program.graph_module
        )

    return op_graph_map


# TODO: use anonymous function to avoid passing the dict around
# and move this inside of the OperatorGraphWithStats class
def create_debug_handle_to_op_node_mapping(
    op_graph: OperatorGraph,
    debug_handle_to_op_node_map: Dict[int, OperatorNode],
) -> None:
    """
    Recursive function to traverse all the operator graph nodes of input op_graph and build a mapping
    from each debug handle to the operator node that contains the debug handle in its metadata.
    """
    # Recursively searches through the metadata of nodes
    for element in op_graph.elements:
        if isinstance(element, OperatorGraph):
            create_debug_handle_to_op_node_mapping(element, debug_handle_to_op_node_map)
        if isinstance(element, OperatorNode) and element.metadata is not None:
            metadata = element.metadata
            debug_handle = metadata.get("debug_handle")
            if debug_handle is not None:
                existing_entry = debug_handle_to_op_node_map.get(debug_handle)
                if existing_entry is not None:
                    raise ValueError(
                        f"Duplicated debug handle {str(debug_handle)} shared between {element.name} and {existing_entry.name}. "
                        "No two op nodes of the same graph should have the same debug handle."
                    )
                debug_handle_to_op_node_map[debug_handle] = element


def gen_etdump_object(etdump_path: Optional[str] = None) -> ETDumpFlatCC:
    # Gen event blocks from etdump
    if etdump_path is None:
        raise ValueError("Etdump_path must be specified.")
    with open(etdump_path, "rb") as buff:
        etdump = deserialize_from_etdump_flatcc(buff.read())
        return etdump
