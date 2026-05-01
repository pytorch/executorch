# Copyright © 2026 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

"""Tests for coreml_compute_plan.py."""

import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter

import coremltools as ct
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coreml_compute_plan import (  # noqa: E402
    _COMPUTE_UNIT_CHOICES,
    _device_name,
    analyze_one,
)


class _Op:
    def __init__(self, operator_name: str, blocks=None):
        self.operator_name = operator_name
        self.blocks = blocks or []


class _Block:
    def __init__(self, ops):
        self.operations = ops


def _build_small_mlpackage(out_dir: str) -> str:
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu(x @ x.T) + x.sum()

    model = M().eval()
    ep = torch.export.export(model, (torch.randn(8, 8),), strict=True)
    ep = ep.run_decompositions({})
    mlmodel = ct.convert(
        ep,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        skip_model_load=True,
    )
    out = os.path.join(out_dir, "tiny.mlpackage")
    mlmodel.save(out)
    return out


class TestDeviceName(unittest.TestCase):
    def test_none_device(self):
        self.assertEqual(_device_name(None), "unknown")

    def test_known_device_classes(self):
        from coremltools.models.compute_device import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )

        # Don't construct the device classes directly (they wrap proxies that
        # may be unavailable in some envs); just confirm the type-mapping path
        # returns sensible names by mocking the isinstance check with a fake.
        class FakeNE(MLNeuralEngineComputeDevice):
            def __init__(self):
                pass

        self.assertEqual(_device_name(FakeNE()), "ANE")


class TestComputeUnitChoices(unittest.TestCase):
    def test_includes_cpu_and_ne(self):
        self.assertEqual(
            _COMPUTE_UNIT_CHOICES["cpu_and_ne"], ct.ComputeUnit.CPU_AND_NE
        )

    def test_includes_all(self):
        self.assertEqual(_COMPUTE_UNIT_CHOICES["all"], ct.ComputeUnit.ALL)


class TestAnalyzeOne(unittest.TestCase):
    """End-to-end: build a tiny mlpackage and analyze it."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.mlpackage = _build_small_mlpackage(cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_returns_rows_for_dispatched_ops(self):
        rows = analyze_one(self.mlpackage, ct.ComputeUnit.CPU_AND_NE)
        self.assertGreater(len(rows), 0, "expected at least one dispatched op")
        # Every row is (function_name, operator_name, device_name).
        for fname, op_name, device in rows:
            self.assertIsInstance(fname, str)
            self.assertIsInstance(op_name, str)
            self.assertIn(device, {"ANE", "GPU", "CPU", "unknown"})

    def test_main_function_present(self):
        rows = analyze_one(self.mlpackage, ct.ComputeUnit.CPU_ONLY)
        self.assertIn("main", {fname for fname, _, _ in rows})

    def test_op_types_for_relu_matmul_model(self):
        # The toy model is `relu(x @ x.T) + x.sum()` so the lowered MIL
        # should at least contain matmul, relu, add and reduce_sum.
        rows = analyze_one(self.mlpackage, ct.ComputeUnit.CPU_ONLY)
        op_types = Counter(op for _, op, _ in rows)
        # Op names are versioned (e.g. "ios17.matmul"), so match by suffix.
        suffixes = {name.split(".")[-1] for name in op_types}
        for expected in ("matmul", "relu", "add", "reduce_sum"):
            self.assertIn(expected, suffixes, f"missing op {expected}: {suffixes}")


if __name__ == "__main__":
    unittest.main()
