# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower
from torch.export import export


class TestXnnpackPartitioner(unittest.TestCase):
    """Test cases for XnnpackPartitioner functionality and deprecation warnings."""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    def test_deprecation_warning_for_to_backend_workflow(self):
        """
        Test that the deprecated to_edge + to_backend workflow shows a deprecation warning.
        """
        model = self.SimpleModel()
        x = torch.randn(1, 10)

        exported_model = export(model, (x,))

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        logger = logging.getLogger(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)

        edge = to_edge(exported_model)
        partitioner = XnnpackPartitioner()

        edge.to_backend(partitioner)

        log_contents = log_capture_string.getvalue()
        self.assertIn("DEPRECATION WARNING", log_contents)
        self.assertIn("to_edge() + to_backend()", log_contents)
        self.assertIn("to_edge_transform_and_lower()", log_contents)

    def test_no_warning_for_to_edge_transform_and_lower_workflow(self):
        """
        Test that the recommended to_edge_transform_and_lower workflow does NOT show a deprecation warning.
        """

        model = self.SimpleModel()
        x = torch.randn(1, 10)

        exported_model = export(model, (x,))

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        logger = logging.getLogger(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)

        partitioner = XnnpackPartitioner()

        to_edge_transform_and_lower(exported_model, partitioner=[partitioner])

        log_contents = log_capture_string.getvalue()
        self.assertNotIn("DEPRECATION WARNING", log_contents)
