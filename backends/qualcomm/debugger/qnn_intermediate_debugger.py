# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator
import os
import warnings
from enum import IntEnum

import torch

from executorch.backends.qualcomm._passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_QUANT_ATTRS,
    QCOM_SCALE,
    QCOM_TENSOR_NAME,
    QCOM_ZERO_POINT,
)
from executorch.devtools import Inspector
from executorch.exir.sym_util import eval_shape

from .format_outputs import export_csv, export_raw, export_svg
from .metrics_evaluator import MetricEvaluatorBase


class OutputFormat(IntEnum):
    SVG_GRAPHS = 0
    CSV_FILES = 1
    DUMP_RAW = 2


class IntermediateModule(torch.nn.Module):
    """
    This class serves as an intermediate point and is inserted right after the call_function node.
    It also saves some metadata such as scale, offset, etc.
    Since we just want to check the intermediate output, we will directly return the value during the forward call.
    """

    def __init__(
        self,
        module_name: str,
        qnn_tensor_name: str,
        node_name: str,
        scale: float,
        zero_point: int,
        revert_order: bool = None,
    ):
        super().__init__()
        self.module_name = module_name
        self.qnn_tensor_name = qnn_tensor_name
        self.node_name = node_name
        self.scale = scale
        self.zero_point = zero_point
        self.revert_order = revert_order

    def forward(self, x):
        return x


class QNNIntermediateDebugger:
    """This is a debugger tool capable of retrieving intermediate results for CPU edge EP.
    We can further compare these with QNN's intermediate output to identify any QNN accuracy issues.
    """

    def __init__(self):
        self.intermediate_outputs = {}

    def set_edge_module(self, edge_module: torch.fx.graph_module.GraphModule):
        self.orig_edge = copy.deepcopy(edge_module)
        self.intermediate_output_module = self._insert_intermediate_module(
            copy.deepcopy(edge_module)
        )

    def generate_results(
        self,
        title: str,
        path: str,
        output_format: OutputFormat,
        inspector: Inspector,
        evaluator: MetricEvaluatorBase = None,
        keep_qnn_layout: bool = False,
    ):
        assert isinstance(
            output_format, OutputFormat
        ), "output_format passed in is not an instance of OutputFormat"
        os.makedirs(path, exist_ok=True)
        if keep_qnn_layout:
            warnings.warn(
                "[QNN Delegate Debugger]: keep_qnn_layout is not recommended for general use case. "
                "QNN and CPU has different dtype(FP V.S. Quantized) and data formats(NCHW V.S. NHWC) in a lot of cases.",
                stacklevel=1,
            )

        # Due to users can switch between keep_qnn_layout between generate_results, rematch this every time.
        # Make this a class variable if repeat matching is taking too long and handle keep_qnn_layout.
        node_tensor_map = self._match_tensors(
            inspector=inspector,
            keep_qnn_layout=keep_qnn_layout,
        )

        if output_format == OutputFormat.SVG_GRAPHS:
            assert evaluator is not None, "Please provide an evaluator."
            export_svg(
                title=title,
                path=path,
                evaluator=evaluator,
                edge_module=self.orig_edge,
                node_tensor_map=node_tensor_map,
            )
        elif output_format == OutputFormat.CSV_FILES:
            assert evaluator is not None, "Please provide an evaluator."
            export_csv(
                title=title,
                path=path,
                evaluator=evaluator,
                edge_module=self.orig_edge,
                node_tensor_map=node_tensor_map,
            )
        elif output_format == OutputFormat.DUMP_RAW:
            warnings.warn(
                f"[QNN Delegate Debugger]: Param 'title' will be ignored, all raw files will be stored under: {path}",
                stacklevel=1,
            )
            if evaluator:
                warnings.warn(
                    "[QNN Delegate Debugger]: Param 'evaluator' will be ignored as DUMP_RAW will only dump tensors to raw files but won't perform comparison.",
                    stacklevel=1,
                )
            export_raw(
                path=path,
                edge_module=self.intermediate_output_module,
                node_tensor_map=node_tensor_map,
            )
        else:
            warnings.warn(
                "[QNN Delegate Debugger]: Unknown output format, do nothing.",
                stacklevel=1,
            )
            return

    def _insert_intermediate_module(  # noqa: C901
        self, edge_module: torch.fx.graph_module.GraphModule
    ):
        """
        This feature is for intermediate tensor dump on the host CPU.
        After we get an edge GraphModule, we insert submodule between each call_function node,
        and we register forward hooks to store the intermediate results.
        We have to use the edge GraphModule because this is the graph closest to what QNN is executing
        while still being a valid graph to ExecuTorch.

        Args:
            edge_module (exir.ExirExportedProgram): A deep copy of edge ir graph module.
               We need to deep copy so we don't mess up the original edge_ep.
        Returns:
            exir.ExirExportedProgram: A deep copy of edge graph_module with intermediate modules inserted.
        """

        def hook_fn(module, input, output):
            meta = {}
            meta[QCOM_TENSOR_NAME] = module.qnn_tensor_name
            meta["node_name"] = module.node_name
            meta[QCOM_SCALE] = module.scale
            meta[QCOM_ZERO_POINT] = module.zero_point
            meta["revert_order"] = module.revert_order
            meta["output"] = output  # CPU output

            assert (
                module.qnn_tensor_name not in self.intermediate_outputs
            ), f"{module.qnn_tensor_name} checked already, check if this is a potential error"
            self.intermediate_outputs[module.qnn_tensor_name] = meta

        graph = edge_module.graph
        module_count = 0
        for node in graph.nodes:
            if node.op == "call_function":
                module_name = f"intermediate_module_{module_count}"
                module_count += 1
                with graph.inserting_after(node):
                    scale = None
                    zero_point = None
                    if QCOM_QUANT_ATTRS in node.meta:
                        scale = node.meta[QCOM_QUANT_ATTRS][QCOM_SCALE]
                        zero_point = node.meta[QCOM_QUANT_ATTRS][QCOM_ZERO_POINT]

                    revert_order = QCOM_AXIS_ORDER in node.meta

                    if node.target == operator.getitem:
                        index = node.args[1]
                        # Ex: topk -> intermediate_module -> get_item
                        src_node = node.args[0].args[0]
                        qnn_tensor_name = src_node.meta[QCOM_TENSOR_NAME][index]
                    elif any(user.target == operator.getitem for user in node.users):
                        # For cases like topK, qnn_tensor_name is stored in get_item instead of source_node itself.
                        assert all(
                            user.target == operator.getitem for user in node.users
                        ), "Expect all users to be get_item node"
                        qnn_tensor_name = node.name
                    elif QCOM_TENSOR_NAME in node.meta:
                        assert (
                            len(node.meta[QCOM_TENSOR_NAME]) == 1
                        ), "Expecting a single qnn_tensor name but get more than 1."
                        qnn_tensor_name = node.meta[QCOM_TENSOR_NAME][0]
                    else:
                        # Unused
                        qnn_tensor_name = node.name

                    obs = IntermediateModule(
                        module_name=module_name,
                        qnn_tensor_name=qnn_tensor_name,
                        node_name=node.name,
                        scale=scale,
                        zero_point=zero_point,
                        revert_order=revert_order,
                    )
                    setattr(
                        edge_module,
                        module_name,
                        obs,
                    )
                    new_obs = graph.create_node("call_module", module_name, (node,), {})
                orig_users = list(node.users.keys())
                for user_node in orig_users:
                    if user_node is new_obs:
                        continue
                    user_node.replace_input_with(node, new_obs)

        # Register hooks for all intermediate layers
        for (
            _,
            layer,
        ) in edge_module.named_modules():
            if isinstance(layer, IntermediateModule):
                layer.register_forward_hook(hook_fn)

        graph.eliminate_dead_code()
        edge_module.recompile()

        return edge_module

    def _process_qnn_output(self, qnn_output: torch.tensor, meta: dict) -> torch.tensor:
        """
        QNN intermediate results are all quantized.
        We need to dequantize them to match CPU float values.
        Additionally, we need to revert the layout format for layout-sensitive nodes.

        Args:
            qnn_output (torch.tensor): QNN intermediate output from inspector event
            meta (dict): The meta for this tensor/node that is stored during insert_intermediate_module().

        Returns:
            torch.tensor: Processed tensor that should have same dtype and shape as CPU tensors.
        """
        qnn_output = qnn_output.to(torch.float32)
        if meta[QCOM_SCALE] is not None:
            scale = meta[QCOM_SCALE]
            zero_point = meta[QCOM_ZERO_POINT]
            qnn_output = (
                qnn_output.sub(zero_point).mul(scale).to(torch.float32).contiguous()
            )
        if meta["revert_order"]:
            axis_order = LayoutTransform.get_axis_order(
                eval_shape(qnn_output.shape), reverse=True
            )
            qnn_output = qnn_output.permute(axis_order)
        return qnn_output

    def _match_tensors(self, inspector: Inspector, keep_qnn_layout: bool = False):
        """
        Map QNN tensors back to CPU tensors.
        Create a map using the node name as the key and (preprocessed/postprocessed QNN tensor, CPU tensor, meta) as the value.
        We need meta because it holds values such as scale, offset, layout sensitivity, etc.

        Args:
            inspector (Inspector): Inspector that parse QNN runtime intermediate outputs
            keep_qnn_layout (bool): If true, store QNN outputs in NHWC format. Not recommended for general users.

        Returns:
            A dict storing {node_name : tuple(qnn_output, cpu_output, meta_info)}
            Meta_info is the info stored during forward hook_fn.
        """

        # node_tensor_map {key: tuple(qnn_output, cpu_output, meta_info)}
        node_tensor_map = {}
        # OPs that only exists in QNN but not CPU Golden
        unmatched_qnn_tensors = []
        # E.g.: DELEGATE_CALL (This is the model input data), 'Method::execute'
        ignored_events = []
        # Collected with forward hook
        intermediate_outputs = self.intermediate_outputs
        for event_block in inspector.event_blocks:
            if event_block.name == "Execute":
                for event in event_block.events:
                    # If user enables profiling and dump intermediate outputs the same time, we need to skip the profiling event
                    if event.perf_data is not None and event.is_delegated_op:
                        continue
                    if meta := intermediate_outputs.get(event.name):
                        node_name = meta["node_name"]
                        cpu_output = meta["output"]
                        qnn_output = (
                            event.debug_data[0]
                            if keep_qnn_layout
                            else self._process_qnn_output(event.debug_data[0], meta)
                        )
                        node_tensor_map[node_name] = (
                            qnn_output,
                            cpu_output,
                            meta,
                        )

                    else:
                        (
                            unmatched_qnn_tensors.append(event.name)
                            if event.is_delegated_op
                            else ignored_events.append(event.name)
                        )

        warnings.warn(
            f"The following events are ignored: {ignored_events}", stacklevel=1
        )
        warnings.warn(
            f"The following QNN OPs are missing CPU reference. OPs added during qnn_preprocess will not have CPU reference. Please ensure the operations below are created during qnn_preprocess. {unmatched_qnn_tensors}",
            stacklevel=1,
        )
        return node_tensor_map
