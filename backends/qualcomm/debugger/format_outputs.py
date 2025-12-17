# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
from typing import Any

import pydot
import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_QUANT_ATTRS,
    QCOM_SCALE,
    QCOM_SCALES,
    QCOM_TENSOR_NAME,
    QCOM_ZERO_POINT,
    QCOM_ZERO_POINTS,
)

from .metrics_evaluator import MetricEvaluatorBase


# Copied from site-packages/torch/fx/passes/graph_drawer.py
def typename(target: Any) -> str:
    from torch.fx.node import _get_qualified_name

    if isinstance(target, torch.nn.Module):
        ret = torch.typename(target)
    elif isinstance(target, str):
        ret = target
    else:
        ret = _get_qualified_name(target)

    # Escape "{" and "}" to prevent dot files like:
    # https://gist.github.com/SungMinCho/1a017aab662c75d805c5954d62c5aabc
    # which triggers `Error: bad label format (...)` from dot
    return ret.replace("{", r"\{").replace("}", r"\}")


def retrieve_node_info(evaluator, node, node_tensor_map):

    node_info = {}
    node_info["name"] = node.name
    node_info["op_code"] = node.op
    node_info["target"] = typename(node.target)
    node_info["num_users"] = len(node.users)

    if "val" in node.meta:
        if isinstance(node.meta["val"], torch.Tensor):
            node_info["pytorch_layout"] = node.meta["val"].shape
        elif isinstance(node.meta["val"], (list, tuple)):
            shape_list = []
            for i in range(len(node.meta["val"])):
                shape_list.append(node.meta["val"][i].shape)
            node_info["pytorch_layout"] = shape_list

    if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
        node_info["scale(s)"] = (
            quant_attrs.get(QCOM_SCALES)
            if QCOM_SCALES in quant_attrs
            else quant_attrs.get(QCOM_SCALE)
        )
        node_info["zero_point(s)"] = (
            quant_attrs.get(QCOM_ZERO_POINTS)
            if QCOM_ZERO_POINTS in quant_attrs
            else quant_attrs.get(QCOM_ZERO_POINT)
        )

    if node.name in node_tensor_map:
        qnn_output, cpu_output, meta = node_tensor_map[node.name]
        node_info[QCOM_TENSOR_NAME] = meta.get(QCOM_TENSOR_NAME)
        node_info[evaluator.metric_name()], node_info["is_valid_score"] = (
            evaluator.evaluate(qnn_output, cpu_output)
        )

        # The values in meta are directly retrieved from the node during the forward hook, which means the values should be the same for meta and node.meta.
        # Storing these data during the forward hook helps us compare QNN tensors with CPU tensors without traversing the graph.
        # We only check "scale" and not "scales" since the forward hook only stores the node's output, which should always be per tensor.
        if QCOM_QUANT_ATTRS in node.meta:
            assert (
                node_info["scale(s)"] == node.meta[QCOM_QUANT_ATTRS][QCOM_SCALE]
            ), "node meta scale should be same as scale retrieve during forward hook"
            assert (
                node_info["zero_point(s)"]
                == node.meta[QCOM_QUANT_ATTRS][QCOM_ZERO_POINT]
            ), "node meta zero_point should be same as zero_point retrieve during forward hook"

    return node_info


def export_svg(
    title: str,
    path: str,
    evaluator: MetricEvaluatorBase,
    edge_module: torch.fx.GraphModule,
    node_tensor_map: dict,
):
    def get_node_style(is_valid_score: bool):
        template = {
            "shape": "record",
            "style": '"filled,rounded"',
            "fontcolor": "#000000",
        }

        if is_valid_score is None:
            template["fillcolor"] = "LemonChiffon1"  # No match between QNN and CPU
        elif is_valid_score:
            template["fillcolor"] = "DarkOliveGreen3"  # Good accuracy
        else:
            template["fillcolor"] = "Coral1"  # Bad accuracy

        return template

    pydot_graph = pydot.Dot(graph_type="graph")
    node_map = {}

    # Create node
    for node in edge_module.graph.nodes:
        # These are just nodes before fold_quant and still there
        if len(node.users) == 0 and node.op == "placeholder":
            continue
        node_info = retrieve_node_info(
            evaluator=evaluator, node=node, node_tensor_map=node_tensor_map
        )

        node_label = "{"
        node_label += f"name=%{node_info.get('name')}" + r"\n"
        node_label += f"|op_code={node_info.get('op_code')}" + r"\n"
        node_label += f"|qnn_tensor_name={node_info.get('qnn_tensor_name')}" + r"\n"
        node_label += f"|target={node_info.get('target')}" + r"\n"
        node_label += f"|num_users={node_info.get('num_users')}" + r"\n"
        node_label += f"|pytorch_layout={node_info.get('pytorch_layout')}" + r"\n"
        node_label += f"|scale(s)={node_info.get('scale(s)')}" + r"\n"
        node_label += f"|zero_point(s)={node_info.get('zero_point(s)')}" + r"\n"
        node_label += (
            f"|{evaluator.metric_name()}={node_info.get(evaluator.metric_name())}"
            + r"\n"
        )
        node_label += f"|is_valid_score={node_info.get('is_valid_score')}" + r"\n"
        node_label += "}"

        template = get_node_style(node_info.get("is_valid_score"))
        pydot_node = pydot.Node(node.name, label=node_label, **template)
        node_map[node.name] = pydot_node
        pydot_graph.add_node(pydot_node)

    # Create edge
    for node in edge_module.graph.nodes:
        if len(node.users) == 0 and node.op == "placeholder":
            continue
        cur_pydot_node = node_map[node.name]
        users = list(node.users.keys())
        for user in users:
            user_pydot_node = node_map[user.name]
            pydot_graph.add_edge(
                pydot.Edge(cur_pydot_node, user_pydot_node, dir="forward")
            )

    pydot_graph.write_svg(f"{path}/{title}.svg")
    print(f"Intermediate debugger graph saved at: {path}/{title}.svg")


def export_csv(
    title: str,
    path: str,
    evaluator: MetricEvaluatorBase,
    edge_module: torch.fx.GraphModule,
    node_tensor_map: dict,
):
    node_info_list = []
    for node in edge_module.graph.nodes:
        # These are just nodes before fold_quant and still there
        if len(node.users) == 0 and node.op == "placeholder":
            continue
        node_info = retrieve_node_info(
            evaluator=evaluator, node=node, node_tensor_map=node_tensor_map
        )
        node_info_list.append(node_info)

    # Writing to a CSV file
    with open(f"{path}/{title}.csv", mode="w", newline="") as csv_file:
        fieldnames = [
            "name",
            "op_code",
            "qnn_tensor_name",
            "target",
            "num_users",
            "pytorch_layout",
            "scale(s)",
            "zero_point(s)",
            f"{evaluator.metric_name()}",
            "is_valid_score",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(node_info_list)

    print(f"Intermediate debugger csv saved at: {path}/{title}.csv")


def export_raw(
    path: str,
    edge_module: torch.fx.GraphModule,
    node_tensor_map: dict,
):
    for node in edge_module.graph.nodes:
        # These are just unused nodes before fold_quant and still there
        if len(node.users) == 0 and node.op == "placeholder":
            continue
        if paired_event := node_tensor_map.get(node.name):
            qnn_output, cpu_output, meta = paired_event
            qnn_tensor_name = meta[QCOM_TENSOR_NAME]
            qnn_output_path = os.path.join(path, qnn_tensor_name + "_qnn.raw")
            cpu_output_path = os.path.join(path, qnn_tensor_name + "_cpu.raw")
            qnn_output.numpy().tofile(qnn_output_path)
            cpu_output.numpy().tofile(cpu_output_path)

    print(f"Intermediate debugger raw files saved at: {path}")
