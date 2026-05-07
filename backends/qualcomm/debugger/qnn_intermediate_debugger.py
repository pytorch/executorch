# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from enum import IntEnum

import executorch.exir as exir
from executorch.backends.qualcomm.debugger.qcom_numerical_comparator_base import (
    QcomNumericalComparatorBase,
)
from executorch.devtools import Inspector

from .format_outputs import export_csv, export_svg


class OutputFormat(IntEnum):
    SVG_GRAPH = 0
    CSV_FILE = 1


class QNNIntermediateDebugger:
    """This is a debugger tool capable of retrieving intermediate results for CPU edge EP.
    We can further compare these with QNN's intermediate output to identify any accuracy issues.
    """

    def __init__(self, sample_input):
        self.sample_input = sample_input
        self.edge_ep = None
        self.etrecord_file_path = None
        self.inspector = None
        # Support single to edge after transform forward graph for now.
        self.reference_graph_name = "edge_after_transform/forward"

    def set_edge_ep(self, edge_ep: exir.ExirExportedProgram):
        self.edge_ep = edge_ep

    def set_etrecord_file_path(self, etrecord_file_path: str):
        self.etrecord_file_path = etrecord_file_path

    def setup_inspector(self, etdump_path: str, debug_buffer_path: str):
        self.inspector = Inspector(
            etdump_path=etdump_path,
            debug_buffer_path=debug_buffer_path,
            etrecord=self.etrecord_file_path,
            reference_graph_name=self.reference_graph_name,
        )

    def create_comparator(
        self, comparator_cls: type[QcomNumericalComparatorBase], **kwargs
    ) -> QcomNumericalComparatorBase:
        # No need to pass edge_ep — the factory injects it automatically.
        # Just pass the comparator class and any comparator-specific args:
        #   comparator = debugger.create_comparator(QcomMSEComparator, threshold=1e-4)
        assert (
            self.edge_ep is not None
        ), "edge_ep must be set before creating a comparator."
        return comparator_cls(edge_ep=self.edge_ep, **kwargs)

    def generate_results(
        self,
        title: str,
        path: str,
        output_format: OutputFormat,
        comparator: QcomNumericalComparatorBase,
    ):
        assert isinstance(
            output_format, OutputFormat
        ), "output_format passed in is not an instance of OutputFormat"
        os.makedirs(path, exist_ok=True)

        numeric_results = self.inspector.calculate_numeric_gap(
            distance=comparator, reference_graph=self.reference_graph_name
        )
        numeric_results = numeric_results.set_index("runtime_debug_handle")

        if output_format == OutputFormat.SVG_GRAPH:
            export_svg(
                title=title,
                path=path,
                edge_ep=self.edge_ep,
                numeric_results=numeric_results,
                comparator=comparator,
            )
        elif output_format == OutputFormat.CSV_FILE:
            export_csv(
                title=title,
                path=path,
                edge_ep=self.edge_ep,
                numeric_results=numeric_results,
                comparator=comparator,
            )
        else:
            warnings.warn(
                "[QNN Delegate Debugger]: Unknown output format, do nothing.",
                stacklevel=1,
            )
            return
