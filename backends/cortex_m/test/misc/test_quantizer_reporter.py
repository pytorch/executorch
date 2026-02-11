# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_ACTIVATION_PER_CHANNEL_QSPEC,
    INT8_WEIGHT_PER_TENSOR_QSPEC,
)
from executorch.backends.cortex_m.quantizer.quantizer import mark_node_as_annotated
from executorch.backends.cortex_m.quantizer.quantizer_reporter import (
    logger as quantizer_logger,
    QuantizerInfo,
    QuantizerReport,
    QuantizerReporter,
    QuantizerReporterUser,
)
from torch.export import export


class _TwoOpModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.relu(x + y)


def _export_two_op_graph_module():
    return export(_TwoOpModule(), (torch.ones(2, 2), torch.ones(2, 2))).graph_module


class _DummyQuantizer(QuantizerReporterUser):
    def __init__(self):
        super().__init__()

    def get_quantizer_info(self) -> QuantizerInfo:
        return QuantizerInfo(
            name="DummyQuantizer",
            targeted_nodes_description="dummy nodes",
            quantization_config_path="dummy.config",
            support_config_path="dummy.support",
        )


def test_warning_log_level(caplog):
    graph_module = _export_two_op_graph_module()

    reporter = QuantizerReporter([])
    with caplog.at_level(logging.WARNING, logger=quantizer_logger.name):
        reporter.log_quantizer_report(graph_module)

    unexpected_string = """----------------------------------------------------------------------------------------------------
                                    QUANTIZATION REPORT                                         
----------------------------------------------------------------------------------------------------
Non annotated nodes: 5
----------------------------------------------------------------------------------------------------"""

    assert unexpected_string not in caplog.text


def test_info_log_level(caplog):
    """Ensure report generation does not fail when INFO/DEBUG is disabled."""
    graph_module = _export_two_op_graph_module()

    reporter = QuantizerReporter([])
    with caplog.at_level(logging.INFO, logger=quantizer_logger.name):
        reporter.log_quantizer_report(graph_module)

    expected_string = """----------------------------------------------------------------------------------------------------
                                        QUANTIZATION REPORT                                         
----------------------------------------------------------------------------------------------------
Non annotated nodes: 5
----------------------------------------------------------------------------------------------------"""

    assert expected_string in caplog.text


def test_debug_log_level(caplog):
    """Ensure report generation does not fail when INFO/DEBUG is disabled."""
    graph_module = _export_two_op_graph_module()

    add_node = next(
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.add.Tensor
    )
    relu_node = next(
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.relu.default
    )

    quantizer1 = _DummyQuantizer()
    quantizer2 = _DummyQuantizer()
    quantizer3 = _DummyQuantizer()

    reporter = QuantizerReporter([quantizer1, quantizer2, quantizer3])

    mark_node_as_annotated(
        add_node,
        {add_node.args[0]: INT8_WEIGHT_PER_TENSOR_QSPEC, add_node.args[1]: None},
        None,
    )
    mark_node_as_annotated(
        relu_node,
        {},
        INT8_ACTIVATION_PER_CHANNEL_QSPEC,
    )
    quantizer1.report_accept([add_node, relu_node])
    quantizer2.report_reject(
        [add_node], QuantizerReport._PREVIOUS_ANNOTATION_REJECT_REASON
    )
    quantizer2.report_reject([relu_node], "Dummy rejection message")

    with caplog.at_level(logging.DEBUG, logger=quantizer_logger.name):
        reporter.log_quantizer_report(graph_module)

    expected_string = """----------------------------------------------------------------------------------------------------
                                        QUANTIZATION REPORT                                         
----------------------------------------------------------------------------------------------------
DummyQuantizer using dummy nodes
Annotating with dummy.config
Supported operators and patterns defined by dummy.support
   Accepted nodes: 2
   Rejected due to previous annotation: 0
   Rejected nodes: 0

       NODE NAME    INPUT QSPEC MAP                  OUTPUT QSPEC MAP
   --  -----------  -------------------------------  ---------------------------------
   ╒   add          x: INT8_WEIGHT_PER_TENSOR_QSPEC  None
   |                y: None
   ╘   relu                                          INT8_ACTIVATION_PER_CHANNEL_QSPEC
----------------------------------------------------------------------------------------------------
DummyQuantizer using dummy nodes
Annotating with dummy.config
Supported operators and patterns defined by dummy.support
   Accepted nodes: 0
   Rejected due to previous annotation: 1
   Rejected nodes: 1

   NODE NAME    INPUT QSPEC MAP                    OUTPUT QSPEC MAP
   -----------  ---------------------------------  ------------------
   relu         Rejected: Dummy rejection message
----------------------------------------------------------------------------------------------------
DummyQuantizer using dummy nodes
Annotating with dummy.config
Supported operators and patterns defined by dummy.support
   No patterns accepted or rejected.

----------------------------------------------------------------------------------------------------
Non annotated nodes:
    x
    y
    output
----------------------------------------------------------------------------------------------------"""

    assert expected_string in caplog.text
