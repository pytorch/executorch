# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for DYNAMIC_UNBOUND tensor support in portable mode.

Verifies that:
1. A model with DYNAMIC_UNBOUND intermediate activations can be loaded and run.
2. Multiple inferences succeed without corruption.
3. Memory does not grow unboundedly (FreeCall works).
4. Normal models continue to work (regression).
"""

import os
import resource
import tempfile
import unittest

import torch
from torch import nn
from torch.export import export

import executorch.exir as exir
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.tensor import TensorSpec


# Keep loaded C++ modules alive to avoid triggering destructor segfaults
# during test cleanup.
_KEEP_ALIVE = []


# ── helpers ──────────────────────────────────────────────────────────────────


class _DynamicUnboundMemoryPlanningPass(MemoryPlanningPass):
    """Marks intermediate activation specs as DYNAMIC_UNBOUND before running
    the standard memory planning algorithm."""

    def run(self, graph_module, graph_signature=None):
        placeholder_names = {
            n.name for n in graph_module.graph.nodes if n.op == "placeholder"
        }
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            spec = node.meta.get("spec")
            if not isinstance(spec, TensorSpec) or spec.const:
                continue
            if all(
                getattr(a, "name", None) in placeholder_names
                for a in node.args
                if isinstance(a, torch.fx.Node)
            ):
                continue
            spec.shape_dynamism = TensorShapeDynamism.DYNAMIC_UNBOUND

        return super().run(graph_module, graph_signature)


class MultiLayerModel(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.linear1 = nn.Linear(size, size, bias=False)
        self.linear2 = nn.Linear(size, size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)
        z = torch.relu(y)
        return self.linear2(z)


def _export_dynamic_unbound_pte(model, example_input, tmp_path):
    """Export model with DYNAMIC_UNBOUND intermediates, return .pte path."""
    ep = export(model, (example_input,))
    edge = to_edge(
        ep,
        compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
    )
    et = edge.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=_DynamicUnboundMemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
        ),
    )
    pte_path = os.path.join(tmp_path, "model.pte")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    return pte_path


def _export_normal_pte(model, example_input, tmp_path):
    """Export model with standard memory planning, return .pte path."""
    ep = export(model, (example_input,))
    edge = to_edge(
        ep,
        compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
    )
    et = edge.to_executorch()
    pte_path = os.path.join(tmp_path, "model.pte")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    return pte_path


def _load(pte_path):
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch,
    )
    mod = _load_for_executorch(pte_path)
    _KEEP_ALIVE.append(mod)
    return mod


# ── tests ────────────────────────────────────────────────────────────────────


class TestDynamicUnboundRuntime(unittest.TestCase):
    """DYNAMIC_UNBOUND model loads and runs in portable mode."""

    def test_model_loads_and_runs(self):
        model = MultiLayerModel(size=8)
        inp = torch.randn(1, 8)
        with tempfile.TemporaryDirectory() as tmp:
            pte = _export_dynamic_unbound_pte(model, inp, tmp)
            mod = _load(pte)
            result = mod.forward([inp])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].shape, (1, 8))

    def test_multiple_inferences_succeed(self):
        model = MultiLayerModel(size=8)
        inp = torch.randn(1, 8)
        with tempfile.TemporaryDirectory() as tmp:
            pte = _export_dynamic_unbound_pte(model, inp, tmp)
            mod = _load(pte)
            for i in range(20):
                result = mod.forward([torch.randn(1, 8)])
                self.assertEqual(result[0].shape, (1, 8))
                self.assertTrue(
                    torch.isfinite(result[0]).all(),
                    f"Inference {i} produced non-finite values",
                )

    def test_memory_does_not_grow_unbounded(self):
        model = MultiLayerModel(size=64)
        inp = torch.randn(1, 64)
        with tempfile.TemporaryDirectory() as tmp:
            pte = _export_dynamic_unbound_pte(model, inp, tmp)
            mod = _load(pte)
            for _ in range(5):
                mod.forward([torch.randn(1, 64)])
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            for _ in range(200):
                mod.forward([torch.randn(1, 64)])
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            import platform
            if platform.system() == "Darwin":
                growth_bytes = rss_after - rss_before
            else:
                growth_bytes = (rss_after - rss_before) * 1024
            self.assertLess(
                growth_bytes,
                10 * 1024 * 1024,
                f"RSS grew by {growth_bytes / 1024 / 1024:.1f} MB over 200 "
                f"inferences; dynamic memory is likely not being freed.",
            )


class ModelWithBuffer(nn.Module):
    """Model with a mutable buffer (stays STATIC) and DYNAMIC_UNBOUND intermediates."""

    def __init__(self, size=8):
        super().__init__()
        self.register_buffer("counter", torch.zeros(1, size))
        self.linear = nn.Linear(size, size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.counter.add_(x)
        return self.linear(torch.relu(self.counter))


class TestMutableBufferWithDynamicUnbound(unittest.TestCase):
    """Mutable buffer state is tracked correctly across inferences while
    intermediates use DYNAMIC_UNBOUND."""

    def test_buffer_state_persists(self):
        size = 4
        model = ModelWithBuffer(size=size)
        inp = torch.ones(1, size)
        with tempfile.TemporaryDirectory() as tmp:
            pte = _export_dynamic_unbound_pte(model, inp, tmp)
            mod = _load(pte)

            # First inference: counter becomes [1,1,1,1]
            r1 = mod.forward([inp])
            self.assertEqual(r1[0].shape, (1, size))

            # Second inference with same input: counter becomes [2,2,2,2]
            # Output should differ from first since buffer accumulated
            r2 = mod.forward([inp])
            self.assertFalse(
                torch.allclose(r1[0], r2[0]),
                "Outputs should differ — mutable buffer should accumulate state",
            )


class TestNormalModelRegression(unittest.TestCase):
    """Normal models with standard memory planning still work."""

    def test_normal_model_runs(self):
        model = MultiLayerModel(size=8)
        inp = torch.randn(1, 8)
        with tempfile.TemporaryDirectory() as tmp:
            pte = _export_normal_pte(model, inp, tmp)
            mod = _load(pte)
            result = mod.forward([inp])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].shape, (1, 8))
            self.assertTrue(torch.isfinite(result[0]).all())


if __name__ == "__main__":
    unittest.main()
