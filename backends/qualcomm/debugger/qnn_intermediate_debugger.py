# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import os
import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import torch
from executorch.backends.qualcomm._passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_DEBUG_HANDLE,
    QCOM_QUANT_ATTRS,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)
from executorch.devtools import Inspector
from executorch.devtools.inspector._intermediate_output_capturer import (
    IntermediateOutputCapturer,
)
from executorch.exir.sym_util import eval_shape

from .format_outputs import export_csv, export_raw, export_svg
from .metrics_evaluator import MetricEvaluatorBase


class OutputFormat(IntEnum):
    SVG_GRAPHS = 0
    CSV_FILES = 1
    DUMP_RAW = 2


class QNNIntermediateDebugger:
    """This is a debugger tool to leverage IntermediateOutputCapturer to dump CPU intermediate results
    and compare it with QNN's intermediate output to identify any QNN accuracy issues.
    """

    @dataclass(frozen=True)
    class QnnDebugMetaData:
        """
        Summary: Meta data that will be used later for CPU V.S. QNN comparison
        handle_id: handle_id under node.meta.
        scale: Scale of the node for quant model, None for FP model.
        zero_point: Zero point of the node for quant model, None for FP model.
        is_qcom_layout: Taking 4D tensor as example, whether the node passed to QNN is in layout of NCHW (pytorch layout) or NHWC(qcom layout).
                        This is directly related is QCOM_AXIS_ORDER meta added during LayoutTransform pass.
        edge_node_name: The node.name during edge ir.
        """

        handle_id: int
        scale: float
        zero_point: int
        is_qcom_layout: bool
        edge_node_name: str

    def __init__(self, keep_qnn_layout: bool = False):
        """

        Args:
            keep_qnn_layout (bool, optional): For general usage, keep this as False.
                                              When comparing CPU result with QNN result, taking 4D tensors as example,
                                              some QNN output is in NHWC format, while CPU is in NCHW format.
                                              If turned to true, debugger will compare QNN using NHWC format.
                                              QNN quantized outputs will also remain quantized instead of FP.
                                              Please notice this will cause significant mismatch between QNN and CPU.
                                              This feature is enabled for internal usage when dumping RAW files.

        """
        self.keep_qnn_layout = keep_qnn_layout
        self.golden_intermediate_outputs = None
        self.node_tensor_map = None

        if self.keep_qnn_layout:
            warnings.warn(
                "[QNN Delegate Debugger]: keep_qnn_layout is not recommended for general use case. "
                "QNN and CPU has different dtype(FP V.S. Quantized) and data formats(NCHW V.S. NHWC) in a lot of cases.",
                stacklevel=1,
            )

    def capture_golden(self, sample_input: Tuple[torch.Tensor]):
        if self.golden_intermediate_outputs:
            warnings.warn(
                "[QNN Delegate Debugger]: Golden is already captured. Override the previous result. Please ensure this is intentional.",
                stacklevel=2,
            )
        self.golden_intermediate_outputs = (
            self.intermediate_golden_capturer.run_and_capture(sample_input)
        )

    def generate_results(
        self,
        title: str,
        path: str,
        output_format: OutputFormat,
        inspector: Inspector,
        evaluator: MetricEvaluatorBase = None,
    ):
        assert isinstance(
            output_format, OutputFormat
        ), "[QNN Delegate Debugger]: output_format passed in is not an instance of OutputFormat"
        os.makedirs(path, exist_ok=True)

        # When use calls this function multiple times, only match tensor during 1st time.
        if self.node_tensor_map is None:
            self.node_tensor_map = self._match_tensors(
                inspector=inspector,
            )

        if output_format == OutputFormat.SVG_GRAPHS:
            assert (
                evaluator is not None
            ), "[QNN Delegate Debugger]: Please provide an evaluator."
            export_svg(
                title=title,
                path=path,
                evaluator=evaluator,
                edge_module=self.edge_module,
                node_tensor_map=self.node_tensor_map,
            )
        elif output_format == OutputFormat.CSV_FILES:
            assert (
                evaluator is not None
            ), "[QNN Delegate Debugger]: Please provide an evaluator."
            export_csv(
                title=title,
                path=path,
                evaluator=evaluator,
                edge_module=self.edge_module,
                node_tensor_map=self.node_tensor_map,
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
                node_tensor_map=self.node_tensor_map,
            )
        else:
            warnings.warn(
                "[QNN Delegate Debugger]: Unknown output format, do nothing.",
                stacklevel=1,
            )
            return

    def _process_qnn_output(
        self, qnn_output: torch.tensor, meta: QnnDebugMetaData
    ) -> torch.tensor:
        """
        QNN intermediate results could be quantized.
        We need to dequantize them to match CPU float values.
        Additionally, we need to revert the layout format for layout-sensitive nodes.

        Args:
            qnn_output (torch.tensor): QNN intermediate output from inspector event
            meta (dict): The meta for this tensor/node that is stored during insert_intermediate_module().

        Returns:
            torch.tensor: Processed tensor that should have same dtype and shape as CPU tensors.
        """
        qnn_output = qnn_output.to(torch.float32)
        if meta.scale is not None:
            scale = meta.scale
            zero_point = meta.zero_point
            qnn_output = (
                qnn_output.sub(zero_point).mul(scale).to(torch.float32).contiguous()
            )
        if meta.is_qcom_layout:
            axis_order = LayoutTransform.get_axis_order(
                eval_shape(qnn_output.shape), reverse=True
            )
            qnn_output = qnn_output.permute(axis_order)
        return qnn_output

    def _match_tensors(self, inspector: Inspector):
        """
        Map QNN tensors back to CPU tensors.
        Create a map using the edge_node_name as the key and (preprocessed/postprocessed QNN tensor, CPU tensor, QnnDebugMetaData) as the value.
        We need meta because it holds values such as scale, offset, layout sensitivity, etc.

        Args:
            inspector (Inspector): Inspector that parse QNN runtime intermediate outputs

        Returns:
            A dict storing {edge_node_name : tuple(qnn_output, cpu_output, QnnDebugMetaData)}
        """

        # node_tensor_map {edge_node_name: tuple(qnn_output, cpu_output, QnnDebugMetaData)}
        node_tensor_map = {}
        # OPs that only exists in QNN but not CPU Golden
        unmatched_qnn_tensors = []
        # E.g.: DELEGATE_CALL (This is the model input data), 'Method::execute'
        ignored_events = []

        for event_block in inspector.event_blocks:
            if event_block.name == "Execute":
                for event in event_block.events:
                    # If user enables profiling and dump intermediate outputs the same time, we need to skip the profiling event
                    if event.perf_data is not None and event.is_delegated_op:
                        continue

                    if (
                        event.name.isdigit()
                        and (int(event.name),) in self.golden_intermediate_outputs
                    ):

                        debug_handle = (int(event.name),)
                        cpu_output = self.golden_intermediate_outputs[debug_handle]
                        if torch.is_tensor(cpu_output):
                            cpu_output = [cpu_output]

                        node_meta = self.node_meta_map[debug_handle]

                        # We can't do assertions here because of some edge cases.
                        # Ex: max_pool2d has 2 outputs. However, QNN only has 1 output and graph only use output[0].
                        #     CPU gen an extra output that's never used.
                        if len(cpu_output) != len(event.debug_data):
                            warnings.warn(
                                f"[QNN Delegate Debugger]: Number of output does not match."
                                f"CPU has {len(cpu_output)} outputs. QNN has {len(event.debug_data)} outputs, possibly due to OP generating multiple outputs and some are unused."
                                f"Check following node_meta info to see if this is desired: {node_meta}",
                                stacklevel=1,
                            )
                        for i, event_data in enumerate(event.debug_data):
                            qnn_output = (
                                event_data
                                if self.keep_qnn_layout
                                else self._process_qnn_output(event_data, node_meta[i])
                            )
                            edge_node_name = node_meta[i].edge_node_name
                            assert (
                                edge_node_name not in node_tensor_map
                            ), f"[QNN Delegate Debugger]: Duplicate tensor name found when visiting {edge_node_name}"
                            node_tensor_map[edge_node_name] = (
                                qnn_output,
                                cpu_output[i],
                                node_meta[i],
                            )
                    else:
                        (
                            unmatched_qnn_tensors.append(event.name)
                            if event.is_delegated_op
                            else ignored_events.append(event.name)
                        )

        warnings.warn(
            f"[QNN Delegate Debugger]: The following events are ignored: {ignored_events}",
            stacklevel=1,
        )
        warnings.warn(
            f"[QNN Delegate Debugger]: The following QNN OPs are missing CPU reference. OPs added during qnn_preprocess will not have CPU reference. Please ensure the operations below are created during qnn_preprocess. {unmatched_qnn_tensors}",
            stacklevel=1,
        )
        return node_tensor_map

    def _set_edge_module(
        self, edge_module: torch.fx.graph_module.GraphModule, debug_handle_map: dict
    ):
        self.edge_module = edge_module
        self.intermediate_golden_capturer = IntermediateOutputCapturer(
            module=self.edge_module
        )
        self.debug_handle_map = debug_handle_map
        self.node_meta_map = {}
        for node in self.edge_module.graph.nodes:

            # For multi output ops like topk,
            # meta info is stored in getitem, so skip source node itself.
            if any(user.target == operator.getitem for user in node.users):
                # Assume if a node user is getitem, all users are getitem
                assert all(
                    user.target == operator.getitem for user in node.users
                ), "[QNN Delegate Debugger]: Expect all users to be get_item node"
                continue

            if handle_id := node.meta.get(QCOM_DEBUG_HANDLE):
                scale = None
                zero_point = None
                is_qcom_layout = QCOM_AXIS_ORDER in node.meta
                if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
                    scale = quant_attrs[QCOM_SCALE]
                    zero_point = quant_attrs[QCOM_ZERO_POINT]

                debug_meta = QNNIntermediateDebugger.QnnDebugMetaData(
                    handle_id=handle_id,
                    scale=scale,
                    zero_point=zero_point,
                    is_qcom_layout=is_qcom_layout,
                    edge_node_name=node.name,
                )
                if node.target == operator.getitem:
                    output_idx = node.args[1]
                    if (handle_id,) in self.node_meta_map:
                        self.node_meta_map[(handle_id,)][output_idx] = debug_meta
                    else:
                        self.node_meta_map[(handle_id,)] = {output_idx: debug_meta}
                else:
                    assert (
                        handle_id,
                    ) not in self.node_meta_map, f"[QNN Delegate Debugger]: Duplicate handle_id {handle_id} found when visiting {node.name}."
                    self.node_meta_map[(handle_id,)] = {0: debug_meta}
