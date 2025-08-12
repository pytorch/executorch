# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional

import serializer.tosa_serializer as ts  # type: ignore
import torch
from executorch.exir.print_program import inspect_node

logger = logging.getLogger(__name__)


def debug_node(node: torch.fx.Node, graph_module: torch.fx.GraphModule):
    # Debug output of node information
    logger.info(get_node_debug_info(node, graph_module))


def get_node_debug_info(
    node: torch.fx.Node, graph_module: torch.fx.GraphModule | None = None
) -> str:
    output = (
        f"  {inspect_node(graph=graph_module.graph, node=node)}\n"
        if graph_module
        else ""
        "-- NODE DEBUG INFO --\n"
        f"  Op is {node.op}\n"
        f"  Name is {node.name}\n"
        f"  Node target is {node.target}\n"
        f"  Node args is {node.args}\n"
        f"  Node kwargs is {node.kwargs}\n"
        f"  Node users is {node.users}\n"
        "  Node.meta = \n"
    )
    for k, v in node.meta.items():
        if k == "stack_trace":
            matches = v.split("\n")
            output += "      'stack_trace =\n"
            for m in matches:
                output += f"      {m}\n"
        else:
            output += f"    '{k}' = {v}\n"

            if isinstance(v, list):
                for i in v:
                    output += f"      {i}\n"
    return output


# Output TOSA flatbuffer and test harness file
def debug_tosa_dump(tosa_graph: ts.TosaSerializer, path: str, suffix: str = ""):
    filename = f"output{suffix}.tosa"

    logger.info(f"Emitting debug output to: {path=}, {suffix=}")

    os.makedirs(path, exist_ok=True)

    fb = tosa_graph.serialize()
    js = tosa_graph.writeJson(filename)

    filepath_tosa_fb = os.path.join(path, filename)
    with open(filepath_tosa_fb, "wb") as f:
        f.write(fb)
    if not os.path.exists(filepath_tosa_fb):
        raise IOError("Failed to write TOSA flatbuffer")

    filepath_desc_json = os.path.join(path, f"desc{suffix}.json")
    with open(filepath_desc_json, "w") as f:
        f.write(js)
    if not os.path.exists(filepath_desc_json):
        raise IOError("Failed to write TOSA JSON")


def debug_fail(
    node,
    graph_module,
    tosa_graph: Optional[ts.TosaSerializer] = None,
    path: Optional[str] = None,
):
    logger.warning("Internal error due to poorly handled node:")
    if tosa_graph is not None and path is not None:
        debug_tosa_dump(tosa_graph, path)
        logger.warning(f"Debug output captured in '{path}'.")
    debug_node(node, graph_module)
