# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests that export models with XNNPACK delegation and verify
them through the Python runtime.

Covers the export → .pte → Python runtime flow that developers use to
validate exported models before deploying to device.  Complements the
existing C++ runner tests in .ci/scripts/test_model.sh.

See https://github.com/pytorch/executorch/issues/11225
"""

import tempfile
import unittest
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification
from torch.export import export


def _export_and_load(model: torch.nn.Module, example_inputs: tuple):
    """Export *model* with XNNPACK, save to a temp .pte, load via Runtime."""
    model.eval()
    with torch.no_grad():
        aten = export(model, example_inputs, strict=True)
        edge = to_edge_transform_and_lower(
            aten,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[XnnpackPartitioner()],
        )
        et = edge.to_executorch()

    with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
        pte_path = f.name
    et.save(pte_path)

    rt = Runtime.get()
    program = rt.load_program(Path(pte_path), verification=Verification.Minimal)
    return program.load_method("forward"), pte_path


class TestPythonRuntimeXNNPACK(unittest.TestCase):
    """Export → .pte → Python Runtime tests for XNNPACK on Linux."""

    # ------------------------------------------------------------------
    # Simple arithmetic
    # ------------------------------------------------------------------
    def test_add(self):
        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Add()
        inputs = (torch.randn(2, 3), torch.randn(2, 3))
        method, _ = _export_and_load(model, inputs)

        expected = model(*inputs)
        actual = method.execute(inputs)
        torch.testing.assert_close(actual[0], expected, atol=1e-4, rtol=1e-4)

    # ------------------------------------------------------------------
    # Linear layer (fp32)
    # ------------------------------------------------------------------
    def test_linear(self):
        model = torch.nn.Linear(16, 8)
        inputs = (torch.randn(4, 16),)
        method, _ = _export_and_load(model, inputs)

        with torch.no_grad():
            expected = model(*inputs)
        actual = method.execute(inputs)
        torch.testing.assert_close(actual[0], expected, atol=1e-4, rtol=1e-4)

    # ------------------------------------------------------------------
    # Conv2d + ReLU (common vision pattern)
    # ------------------------------------------------------------------
    def test_conv2d_relu(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
        )
        inputs = (torch.randn(1, 3, 8, 8),)
        method, _ = _export_and_load(model, inputs)

        with torch.no_grad():
            expected = model(*inputs)
        actual = method.execute(inputs)
        torch.testing.assert_close(actual[0], expected, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------
    # Small MLP (multiple linear + activation)
    # ------------------------------------------------------------------
    def test_mlp(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )
        inputs = (torch.randn(2, 32),)
        method, _ = _export_and_load(model, inputs)

        with torch.no_grad():
            expected = model(*inputs)
        actual = method.execute(inputs)
        torch.testing.assert_close(actual[0], expected, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------
    # BatchNorm + Conv (common in MobileNet-style models)
    # Skipped: FuseBatchNormPass crashes on Sequential(Conv2d, BN) export.
    # TODO(#11225): re-enable once the XNNPACK pass is fixed.
    # ------------------------------------------------------------------
    @unittest.skip("FuseBatchNormPass bug in XNNPACK backend")
    def test_conv_bn(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        model.eval()
        inputs = (torch.randn(1, 3, 8, 8),)
        method, _ = _export_and_load(model, inputs)

        with torch.no_grad():
            expected = model(*inputs)
        actual = method.execute(inputs)
        torch.testing.assert_close(actual[0], expected, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------
    # Depthwise separable conv (MobileNet building block)
    # ------------------------------------------------------------------
    def test_depthwise_separable_conv(self):
        model = torch.nn.Sequential(
            # Depthwise
            torch.nn.Conv2d(16, 16, 3, padding=1, groups=16),
            torch.nn.ReLU(),
            # Pointwise
            torch.nn.Conv2d(16, 32, 1),
            torch.nn.ReLU(),
        )
        inputs = (torch.randn(1, 16, 8, 8),)
        method, _ = _export_and_load(model, inputs)

        with torch.no_grad():
            expected = model(*inputs)
        actual = method.execute(inputs)
        torch.testing.assert_close(actual[0], expected, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------
    # Avgpool + Flatten + Linear (classifier head)
    # ------------------------------------------------------------------
    def test_classifier_head(self):
        class ClassifierHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(32, 10)

            def forward(self, x):
                x = self.pool(x)
                x = x.flatten(1)
                return self.fc(x)

        model = ClassifierHead()
        inputs = (torch.randn(1, 32, 8, 8),)
        method, _ = _export_and_load(model, inputs)

        with torch.no_grad():
            expected = model(*inputs)
        actual = method.execute(inputs)
        torch.testing.assert_close(actual[0], expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
