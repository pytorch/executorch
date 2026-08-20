# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os
import subprocess
from typing import Any

import executorch.exir as exir
import pandas
import pydot
import torch
from executorch.backends.qualcomm.debugger.qcom_numerical_comparator_base import (
    QcomNumericalComparatorBase,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_QUANT_ATTRS,
    QCOM_SCALE,
    QCOM_SCALES,
    QCOM_ZERO_POINT,
    QCOM_ZERO_POINTS,
)
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


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


def get_scale_zero_point(node: torch.fx.node.Node):
    scale_zero_point = {"scale(s)": None, "zero_point(s)": None}
    if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
        scale_zero_point["scale(s)"] = (
            quant_attrs.get(QCOM_SCALES)
            if QCOM_SCALES in quant_attrs
            else quant_attrs.get(QCOM_SCALE)
        )
        scale_zero_point["zero_point(s)"] = (
            quant_attrs.get(QCOM_ZERO_POINTS)
            if QCOM_ZERO_POINTS in quant_attrs
            else quant_attrs.get(QCOM_ZERO_POINT)
        )
    return scale_zero_point


def get_pytorch_layout_info(node: torch.fx.node.Node):
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.shape
    return [v.shape for v in val if isinstance(v, torch.Tensor)]


def export_svg(  # noqa: C901
    title: str,
    path: str,
    edge_ep: exir.ExirExportedProgram,
    numeric_results: pandas.core.frame.DataFrame,
    comparator: QcomNumericalComparatorBase,
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
    for node in edge_ep.graph_module.graph.nodes:
        # These are just nodes before fold_quant and still there
        if len(node.users) == 0 and node.op == "placeholder":
            continue

        pytorch_layout = get_pytorch_layout_info(node)
        scale_zero_point = get_scale_zero_point(node)
        scale = scale_zero_point["scale(s)"]
        zero_point = scale_zero_point["zero_point(s)"]

        node_label = "{"
        node_label += f"name=%{node.name}" + r"\n"
        node_label += f"|op_code={node.op}" + r"\n"
        node_label += f"|target={typename(node.target)}" + r"\n"
        node_label += f"|num_users={len(node.users)}" + r"\n"
        node_label += f"|pytorch_layout={pytorch_layout}" + r"\n"
        node_label += f"|scale(s)={scale}" + r"\n"
        node_label += f"|zero_point(s)={zero_point}" + r"\n"

        is_valid_score = None
        if debug_handle := node.meta.get(DEBUG_HANDLE_KEY, None):
            node_label += f"|debug_handle={debug_handle}" + r"\n"
            debug_handle = (debug_handle,)
            if debug_handle in numeric_results.index:
                score = numeric_results.loc[[debug_handle], "gap"].iat[0][0]
                assert isinstance(
                    score, float
                ), f"Expecting QcomNumericalComparatorBase element_compare to return float, but get {type(score)}."
                node_label += f"|{comparator.metric_name()}={score:.3f}" + r"\n"
                is_valid_score = comparator.is_valid_score(score)
        node_label += f"|is_valid_score={is_valid_score}" + r"\n"
        node_label += "}"

        template = get_node_style(is_valid_score)
        pydot_node = pydot.Node(node.name, label=node_label, **template)
        node_map[node.name] = pydot_node
        pydot_graph.add_node(pydot_node)

    # Create edge
    for node in edge_ep.graph_module.graph.nodes:
        if len(node.users) == 0 and node.op == "placeholder":
            continue
        cur_pydot_node = node_map[node.name]
        users = list(node.users.keys())
        for user in users:
            user_pydot_node = node_map[user.name]
            pydot_graph.add_edge(
                pydot.Edge(cur_pydot_node, user_pydot_node, dir="forward")
            )
    dot_file_path = os.path.join(path, f"{title}.dot")
    pydot_graph.write_raw(dot_file_path)
    logging.info(f"Intermediate debugger dot graph saved at: {dot_file_path}")

    svg_file_path = os.path.join(path, f"{title}.svg")
    try:
        subprocess.run(
            ["dot", "-Tsvg", dot_file_path, "-o", svg_file_path],
            timeout=5,
            check=True,
        )
        logging.info(f"Intermediate debugger SVG graph saved at: {svg_file_path}.")
    except subprocess.TimeoutExpired:
        logging.warning(
            f"SVG generation timed out after 5s, skipping. "
            f"Only saving the dot file: {dot_file_path}."
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.warning(f"SVG generation failed ({e}), skipping.")


def export_csv(
    title: str,
    path: str,
    edge_ep: exir.ExirExportedProgram,
    numeric_results: pandas.core.frame.DataFrame,
    comparator: QcomNumericalComparatorBase,
):
    node_info_list = []
    for node in edge_ep.graph_module.graph.nodes:
        # These are just nodes before fold_quant and still there
        if len(node.users) == 0 and node.op == "placeholder":
            continue

        pytorch_layout = get_pytorch_layout_info(node)
        scale_zero_point = get_scale_zero_point(node)
        scale = scale_zero_point["scale(s)"]
        zero_point = scale_zero_point["zero_point(s)"]
        score = None
        is_valid_score = None
        if debug_handle := node.meta.get(DEBUG_HANDLE_KEY, None):
            if (debug_handle,) in numeric_results.index:
                score = numeric_results.loc[[(debug_handle,)], "gap"].iat[0][0]
                assert isinstance(
                    score, float
                ), f"Expecting QcomNumericalComparatorBase element_compare to return float, but get {type(score)}."
                is_valid_score = comparator.is_valid_score(score)

        node_info_list.append(
            {
                "name": node.name,
                "op_code": node.op,
                "target": typename(node.target),
                "num_users": len(node.users),
                "pytorch_layout": pytorch_layout,
                "scale(s)": scale,
                "zero_point(s)": zero_point,
                "debug_handle": debug_handle,
                comparator.metric_name(): score,
                "is_valid_score": is_valid_score,
            }
        )

    # Writing to a CSV file
    csv_file_path = os.path.join(path, f"{title}.csv")
    with open(csv_file_path, mode="w", newline="") as csv_file:
        fieldnames = [
            "name",
            "op_code",
            "target",
            "num_users",
            "pytorch_layout",
            "scale(s)",
            "zero_point(s)",
            "debug_handle",
            comparator.metric_name(),
            "is_valid_score",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(node_info_list)

    print(f"Intermediate debugger csv saved at: {csv_file_path}")
