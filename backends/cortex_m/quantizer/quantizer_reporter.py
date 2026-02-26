# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains classes for reporting quantization decisions made by Quantizers.

Basic useage:
1. Implement the QuantizerReporterUser API for all quantizers intending to use the reporter.
2. Instantiate the QuantizerReporter with a list of quantizers to be reported.
3. After annotation, log the report using QuantizerReporter.log_quantizer_report(model).

Logs a summary report at INFO level, and a detailed node-per-node report at DEBUG level.
"""

from __future__ import annotations

import logging
from typing import Dict, List, NamedTuple, Optional

from executorch.backends.cortex_m.quantizer.quantization_configs import (
    __name__ as quantization_configs_module,
    INT8_ACTIVATION_PER_CHANNEL_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
    INT8_WEIGHT_PER_CHANNEL_QSPEC,
    INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC,
    INT8_WEIGHT_PER_TENSOR_QSPEC,
    SOFTMAX_OUTPUT_FIXED_QSPEC,
)
from tabulate import tabulate
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

logger = logging.getLogger(__name__)

# Look-up dicts used to get human readable names for supported quantization configs and specs
SUPPORTED_QCONFIGS = {
    INT8_PER_CHANNEL_CONFIG: f"{quantization_configs_module}.INT8_PER_CHANNEL_QCONFIG",
    INT8_PER_TENSOR_CONFIG: f"{quantization_configs_module}.INT8_PER_TENSOR_QCONFIG",
}


SUPPORTED_QSPECS = {
    INT8_ACTIVATION_PER_TENSOR_QSPEC: "INT8_ACTIVATION_PER_TENSOR_QSPEC",
    INT8_ACTIVATION_PER_CHANNEL_QSPEC: "INT8_ACTIVATION_PER_CHANNEL_QSPEC",
    INT8_WEIGHT_PER_TENSOR_QSPEC: "INT8_WEIGHT_PER_TENSOR_QSPEC",
    INT8_WEIGHT_PER_CHANNEL_QSPEC: "INT8_WEIGHT_PER_CHANNEL_QSPEC",
    INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC: "INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC",
    SOFTMAX_OUTPUT_FIXED_QSPEC: "SOFTMAX_OUTPUT_FIXED_QSPEC",
    None: "None",
}


def _qspec_repr(qspec):
    return SUPPORTED_QSPECS.get(qspec, "CUSTOM_QSPEC")


class QuantizerInfo(NamedTuple):
    """
    NamedTuple storing information about a Quantizer.
    """

    name: str
    targeted_nodes_description: str
    quantization_config_path: str
    support_config_path: str


class NodeQSpecReport(NamedTuple):
    """
    NamedTuple storing annotation info for a single node.
    """

    name: str
    qspec_input_map_lines: List[str]
    qspec_output: str


class AnnotatedPatternReport(NamedTuple):
    """
    NamedTuple storing annotation info for a pattern of nodes.
    """

    nodes: List[NodeQSpecReport]


class RejectedPatternReport(NamedTuple):
    """
    NamedTuple storing rejection info for a pattern of nodes.
    """

    node_names: List[str]
    rejection_reason: str


class QuantizerReport:
    """
    Reporter class for collecting and generating quantization reports from a single Quantizer.
    Used by the QuantizerReporter to aggregate reports from multiple Quantizers.
    """

    _PREVIOUS_ANNOTATION_REJECT_REASON = "Tried annotating already quantized node."

    def __init__(self, quantizer):
        self.quantizer = quantizer.get_quantizer_info()
        self.accepted_patterns: List[AnnotatedPatternReport] = []
        self.rejected_patterns: List[RejectedPatternReport] = []

    @property
    def accepted_node_count(self) -> int:
        return sum(len(report.nodes) for report in self.accepted_patterns)

    @property
    def rejected_node_count(self) -> int:
        return sum(
            len(report.node_names)
            for report in self.rejected_patterns
            if report.rejection_reason != self._PREVIOUS_ANNOTATION_REJECT_REASON
        )

    @property
    def rejected_previous_annotation_count(self) -> int:
        return sum(
            len(report.node_names)
            for report in self.rejected_patterns
            if report.rejection_reason == self._PREVIOUS_ANNOTATION_REJECT_REASON
        )

    def report_accept(self, pattern: List[Node]) -> None:
        """
        Stores an AnnotatedPatternReport containing info about the accepted pattern.
        """
        node_reports = []
        for node in pattern:
            if Q_ANNOTATION_KEY not in node.meta:
                raise ValueError(
                    "Node {node.name} reported as annotated but has no quantization annotation."
                )
            annotation = node.meta.get(Q_ANNOTATION_KEY)
            qspec_input_map_lines = [
                f"{node.name}: {_qspec_repr(qspec)}"
                for node, qspec in annotation.input_qspec_map.items()
            ]

            node_reports.append(
                NodeQSpecReport(
                    node.name,
                    qspec_input_map_lines,
                    _qspec_repr(annotation.output_qspec),
                )
            )

        self.accepted_patterns.append(AnnotatedPatternReport(node_reports))

    def report_reject(self, pattern, reason):
        """
        Stores an RejectedPatternReport containing info about the rejected pattern.
        """
        self.rejected_patterns.append(
            RejectedPatternReport([node.name for node in pattern], reason)
        )

    def get_quantizer_info_rows(self) -> List[str]:
        rows = []
        rows.append(
            f"{self.quantizer.name} using {self.quantizer.targeted_nodes_description}"
        )
        rows.append(f"Annotating with {self.quantizer.quantization_config_path}")
        rows.append(
            f"Supported operators and patterns defined by {self.quantizer.support_config_path}"
        )

        if (
            self.accepted_node_count
            or self.rejected_node_count
            or self.rejected_previous_annotation_count
        ):
            rows.append(f"   Accepted nodes: {self.accepted_node_count}")
            rows.append(
                f"   Rejected due to previous annotation: {self.rejected_previous_annotation_count}"
            )
            rows.append(f"   Rejected nodes: {self.rejected_node_count}")
        else:
            rows.append("   No patterns accepted or rejected.")

        rows.append("")
        return rows

    def quantizer_report(self, include_table: bool) -> List[str]:
        report_rows = []
        report_rows.extend(self.get_quantizer_info_rows())

        if include_table:
            table = self._format_qspec_table()
            if table:
                report_rows.append(self._indent_block(table, "   "))
        return report_rows

    def _format_qspec_table(self) -> str:
        include_prefix = self._has_multi_node_pattern()
        rows = self._pattern_rows(include_prefix)
        if not rows:
            return ""
        if include_prefix:
            headers = ["", "NODE NAME", "INPUT QSPEC MAP", "OUTPUT QSPEC MAP"]
        else:
            headers = ["NODE NAME", "INPUT QSPEC MAP", "OUTPUT QSPEC MAP"]
        return tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
        )

    def _pattern_rows(self, include_prefix: bool) -> List[List[str]]:
        rows: List[List[str]] = []
        for accepted in self.accepted_patterns:
            if not accepted.nodes:
                continue
            total_lines = sum(
                max(1, len(node.qspec_input_map_lines)) for node in accepted.nodes
            )
            line_index = 0
            for node in accepted.nodes:
                entries = node.qspec_input_map_lines or [""]
                for entry_index, entry in enumerate(entries):
                    prefix = self._pattern_prefix(
                        line_index, total_lines, len(accepted.nodes)
                    )
                    node_name = node.name if entry_index == 0 else ""
                    output = node.qspec_output if entry_index == 0 else ""
                    if include_prefix:
                        rows.append([prefix, node_name, entry, output])
                    else:
                        rows.append([node_name, entry, output])
                    line_index += 1

        for rejected in self.rejected_patterns:
            if not self._should_report_rejection(rejected):
                continue
            node_name = self._rejected_pattern_label(rejected)
            input_qspec = f"Rejected: {rejected.rejection_reason}"
            output_qspec = ""
            if include_prefix:
                rows.append([" ", node_name, input_qspec, output_qspec])
            else:
                rows.append([node_name, input_qspec, output_qspec])
        return rows

    def _has_multi_node_pattern(self) -> bool:
        return any(len(report.nodes) > 1 for report in self.accepted_patterns) or any(
            len(report.node_names) > 1
            for report in self.rejected_patterns
            if self._should_report_rejection(report)
        )

    def _should_report_rejection(self, rejected: RejectedPatternReport) -> bool:
        return rejected.rejection_reason != self._PREVIOUS_ANNOTATION_REJECT_REASON

    def _indent_block(self, text: str, prefix: str) -> str:
        return "\n".join(f"{prefix}{line}" for line in text.splitlines())

    def _pattern_prefix(self, index: int, total: int, pattern_size: int) -> str:
        if pattern_size == 1:
            return " "
        if index == 0:
            return "╒"
        if index == total - 1:
            return "╘"
        return "|"

    def _rejected_pattern_label(self, rejected: RejectedPatternReport) -> str:
        if len(rejected.node_names) == 1:
            return rejected.node_names[0]
        return "(" + ", ".join(rejected.node_names) + ")"


class QuantizerReporter:
    """
    Reporter class for collecting and generating quantization reports from Quantizers
    inheriting from QuantizerReporterUser.
    """

    def __init__(self, quantizers: List[QuantizerReporterUser]):
        self.quantizers: Dict[Quantizer, QuantizerReport] = {}
        self.set_quantizers(quantizers)

    def set_quantizers(self, quantizers: List[QuantizerReporterUser]) -> None:
        """
        Registers quantizers to report their quantization decisions.
        """

        self.quantizers = {}
        for quantizer in quantizers:
            try:
                quantizer.register_reporter(self)
            except AttributeError:
                logger.warning(
                    f"Quantizer {quantizer.__class__.__name__} does not implement QuantizerReporterUser interface and will not report quantization decisions."
                )

            self.quantizers[quantizer] = QuantizerReport(quantizer)

    def report_reject(
        self, quantizer: QuantizerReporterUser, pattern: List[Node], reason: str
    ):
        """
        Reports a node pattern rejected by a quantizer with a given reason.
        """
        quantizer_entry = self.quantizers.get(quantizer, None)
        if quantizer_entry is not None:
            quantizer_entry.report_reject(pattern, reason)
        else:
            raise ValueError(
                f"Quantizer {quantizer.__class__.__name__} not registred in reporter."
            )

    def report_accept(
        self,
        quantizer: QuantizerReporterUser,
        pattern: List[Node],
    ):
        """
        Reports a node pattern accepted by a quantizer.
        """
        quantizer_entry = self.quantizers.get(quantizer, None)
        if quantizer_entry is not None:
            quantizer_entry.report_accept(pattern)
        else:
            raise ValueError(
                f"Quantizer {quantizer.__class__.__name__} not registred in reporter."
            )

    def log_quantizer_report(self, model: Optional[GraphModule] = None):
        """
        Logs the quantization report for all registered quantizers.

        If the logger is set to DEBUG level, a node-per-node report is generated and
        logged at DEBUG level. Otherwise, a summary report is logged at INFO level.
        """
        extended_report = logger.isEnabledFor(logging.DEBUG)

        report = self.get_quantization_report(model, extended_report)
        if extended_report:
            logger.debug(report)
        else:
            logger.info(report)

    def get_quantization_report(
        self, model: Optional[GraphModule], extended_report: bool
    ) -> str:
        """
        Generates the quantization report for all registered quantizers
        """
        report_rows: List[str] = []
        separator = "-" * 100
        report_rows.append(separator)
        report_rows.append(" " * 39 + " QUANTIZATION REPORT " + " " * 40)
        report_rows.append(separator)

        for report in self.quantizers.values():
            report_rows.extend(report.quantizer_report(extended_report))
            report_rows.append(separator)

        report_rows.extend(self.unannotated_nodes_report(model, extended_report))
        report_rows.append(separator)

        report = "\n" + "\n".join(report_rows)
        return report

    def unannotated_nodes_report(
        self, model: Optional[GraphModule], extended_report: bool
    ) -> List[str]:
        """
        Generates the quantization report for all non-annotated nodes in the model.
        """
        non_quantized_nodes = [
            node for node in model.graph.nodes if Q_ANNOTATION_KEY not in node.meta
        ]

        rows = []
        if extended_report:
            rows.append("Non annotated nodes:")
            if non_quantized_nodes:
                for node in non_quantized_nodes:
                    rows.append(f"    {self._pattern_repr([node])}")
            else:
                rows.append("    None")
        else:
            rows.append(f"Non annotated nodes: {len(non_quantized_nodes)}")
        return rows

    def _pattern_repr(self, nodes: List[Node]) -> str:
        names = [n.name for n in nodes]
        if len(names) == 1:
            return f"{names[0]}"
        return "(" + ", ".join(names) + ")"


class QuantizerReporterUser:
    """
    Mixin class for Quantizers, to be used with QuantizerReporter.

    Handles reporter registration and ensures that that the quantizer does not crash
    without a reporter registred
    """

    def __init__(self):
        self.reporter: QuantizerReporter = None

    def register_reporter(self, reporter: QuantizerReporter) -> None:
        """
        Used by QuantizerReporter to register itself with the Quantizer.
        """
        self.reporter = reporter

    def report_reject(self, pattern: List[Node], reason: str) -> None:
        """
        Reports a node pattern rejected by a quantizer, if a reporter is registered.
        """
        if self.reporter is not None:
            self.reporter.report_reject(self, pattern, reason)

    def report_accept(self, pattern: List[Node]) -> None:
        """
        Reports a node pattern accepted by a quantizer, if a reporter is registered.
        """
        if self.reporter is not None:
            self.reporter.report_accept(self, pattern)

    def get_quantizer_info(self) -> "QuantizerInfo":
        """
        Returns a QuantizerInfo NamedTuple with information about the quantizer.
        """
        raise NotImplementedError("Quantizer must implement get_quantizer_info method.")
