# Copyright © 2024 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import platform
import sys
import unittest
from typing import Dict, Optional, Tuple

import coremltools as ct
import torch

from executorch.backends.apple.coreml.compiler.coreml_preprocess import (
    CoreMLBackend,
    MULTIMETHOD_WEIGHT_SHARING_STRATEGY,
)
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


def is_fbcode():
    return not hasattr(torch.version, "git_version")


def _macos_version_at_least(major: int, minor: int = 0) -> bool:
    """Check if the current macOS version is at least major.minor."""
    if sys.platform != "darwin":
        return False
    try:
        version = platform.mac_ver()[0]
        if not version:
            return False
        parts = version.split(".")
        current_major = int(parts[0])
        current_minor = int(parts[1]) if len(parts) > 1 else 0
        return (current_major, current_minor) >= (major, minor)
    except (ValueError, IndexError):
        return False


# Multifunction CoreML models require macOS 15+ / iOS 18+
_TEST_RUNTIME = (
    sys.platform == "darwin" and not is_fbcode() and _macos_version_at_least(15, 0)
)
if _TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestCoreMLMultifunction(unittest.TestCase):
    """Tests for multifunction (multi-method) CoreML model export."""

    edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)

        def forward(self, x):
            return self.linear(x)

    def _get_compile_specs(
        self,
        strategy: Optional[MULTIMETHOD_WEIGHT_SHARING_STRATEGY] = None,
    ):
        """Get compile specs, optionally with a weight sharing strategy."""
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT32,
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            model_type=CoreMLBackend.MODEL_TYPE.MODEL,
        )
        if strategy is not None:
            compile_specs.append(
                CoreMLBackend.generate_multimethod_weight_sharing_strategy_compile_spec(
                    strategy
                )
            )
        return compile_specs

    def _export_and_lower(
        self,
        model: torch.nn.Module,
        method_inputs: Dict[str, Tuple],
        strategy: Optional[MULTIMETHOD_WEIGHT_SHARING_STRATEGY] = None,
        constant_methods: Optional[Dict[str, int]] = None,
    ):
        """Export methods, partition, and lower to ExecuTorch program.

        Args:
            model: The PyTorch module to export.
            method_inputs: Dict mapping method name to example input tuple.
            strategy: Weight sharing strategy, or None for no strategy spec.
            constant_methods: Optional constant methods metadata dict.

        Returns:
            Tuple of (et_program, edge_manager).
        """
        exported_programs = {
            name: torch.export.export(model, inputs)
            for name, inputs in method_inputs.items()
        }

        partitioner = CoreMLPartitioner(
            compile_specs=self._get_compile_specs(strategy=strategy),
        )

        kwargs = {}
        if constant_methods is not None:
            kwargs["constant_methods"] = constant_methods

        edge_manager = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[partitioner],
            compile_config=self.edge_compile_config,
            **kwargs,
        )

        et_program = edge_manager.to_executorch()
        return et_program, edge_manager

    def test_multifunction_simple_model(self):
        """Test exporting a simple model with multiple methods (forward and prefill)."""
        model = self.SimpleModel()
        model.eval()

        decode_inputs = (torch.randn(1, 1, 16),)
        prefill_inputs = (torch.randn(1, 8, 16),)

        et_program, edge_manager = self._export_and_lower(
            model,
            {"forward": decode_inputs, "prefill": prefill_inputs},
            strategy=MULTIMETHOD_WEIGHT_SHARING_STRATEGY.POSITIONAL,
        )

        self.assertIn("forward", edge_manager.methods)
        self.assertIn("prefill", edge_manager.methods)

        if _TEST_RUNTIME:
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            self.assertIn("forward", program.method_names)
            self.assertIn("prefill", program.method_names)

            forward_method = program.load_method("forward")
            decode_output = forward_method.execute(decode_inputs)
            expected_decode = model(*decode_inputs)
            self.assertTrue(
                torch.allclose(decode_output[0], expected_decode, atol=1e-4, rtol=1e-4)
            )

            prefill_method = program.load_method("prefill")
            prefill_output = prefill_method.execute(prefill_inputs)
            expected_prefill = model(*prefill_inputs)
            self.assertTrue(
                torch.allclose(
                    prefill_output[0], expected_prefill, atol=1e-4, rtol=1e-4
                )
            )

    def test_multifunction_with_kv_cache(self):
        """Test multifunction export with KV cache-like buffers."""

        class ModelWithCache(torch.nn.Module):
            def __init__(self, hidden_dim: int, cache_len: int):
                super().__init__()
                self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
                self.hidden_dim = hidden_dim
                self.cache_len = cache_len

            def forward(self, x, cache):
                out = self.linear(x)
                new_cache = torch.cat([cache[:, 1:, :], out[:, -1:, :]], dim=1)
                return out, new_cache

        hidden_dim = 16
        cache_len = 32
        model = ModelWithCache(hidden_dim, cache_len)
        model.eval()

        decode_inputs = (
            torch.randn(1, 1, hidden_dim),
            torch.randn(1, cache_len, hidden_dim),
        )
        prefill_inputs = (
            torch.randn(1, 8, hidden_dim),
            torch.randn(1, cache_len, hidden_dim),
        )

        et_program, _ = self._export_and_lower(
            model,
            {"forward": decode_inputs, "prefill": prefill_inputs},
            strategy=MULTIMETHOD_WEIGHT_SHARING_STRATEGY.POSITIONAL,
        )

        if _TEST_RUNTIME:
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            forward_method = program.load_method("forward")
            decode_output = forward_method.execute(decode_inputs)
            expected_out, expected_cache = model(*decode_inputs)
            self.assertTrue(
                torch.allclose(decode_output[0], expected_out, atol=1e-4, rtol=1e-4)
            )
            self.assertTrue(
                torch.allclose(decode_output[1], expected_cache, atol=1e-4, rtol=1e-4)
            )

            prefill_method = program.load_method("prefill")
            prefill_output = prefill_method.execute(prefill_inputs)
            expected_out, expected_cache = model(*prefill_inputs)
            self.assertTrue(
                torch.allclose(prefill_output[0], expected_out, atol=1e-4, rtol=1e-4)
            )
            self.assertTrue(
                torch.allclose(prefill_output[1], expected_cache, atol=1e-4, rtol=1e-4)
            )

    def test_multifunction_without_weight_sharing(self):
        """Test multifunction export without weight sharing."""
        model = self.SimpleModel()
        model.eval()

        decode_inputs = (torch.randn(1, 1, 16),)
        prefill_inputs = (torch.randn(1, 8, 16),)

        et_program, edge_manager = self._export_and_lower(
            model,
            {"forward": decode_inputs, "prefill": prefill_inputs},
            strategy=MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED,
        )

        self.assertIn("forward", edge_manager.methods)
        self.assertIn("prefill", edge_manager.methods)

        if _TEST_RUNTIME:
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            forward_method = program.load_method("forward")
            decode_output = forward_method.execute(decode_inputs)
            expected_decode = model(*decode_inputs)
            self.assertTrue(
                torch.allclose(decode_output[0], expected_decode, atol=1e-4, rtol=1e-4)
            )

    def test_multifunction_with_constant_methods(self):
        """Test multifunction export with constant methods (metadata)."""
        model = self.SimpleModel()
        model.eval()

        decode_inputs = (torch.randn(1, 1, 16),)
        prefill_inputs = (torch.randn(1, 8, 16),)

        constant_methods = {
            "vocab_size": 32000,
            "hidden_dim": 16,
            "decode_seqlen": 1,
            "prefill_seqlen": 8,
        }

        et_program, _ = self._export_and_lower(
            model,
            {"forward": decode_inputs, "prefill": prefill_inputs},
            strategy=MULTIMETHOD_WEIGHT_SHARING_STRATEGY.POSITIONAL,
            constant_methods=constant_methods,
        )

        if _TEST_RUNTIME:
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            available_methods = program.method_names
            self.assertIn("forward", available_methods)
            self.assertIn("prefill", available_methods)
            self.assertIn("vocab_size", available_methods)
            self.assertIn("hidden_dim", available_methods)
            self.assertIn("decode_seqlen", available_methods)
            self.assertIn("prefill_seqlen", available_methods)

    def test_multifunction_one_blob_simple_model(self):
        """Test exporting a simple model using ONE_BLOB weight sharing strategy."""
        model = self.SimpleModel()
        model.eval()

        decode_inputs = (torch.randn(1, 1, 16),)
        prefill_inputs = (torch.randn(1, 8, 16),)

        et_program, edge_manager = self._export_and_lower(
            model,
            {"forward": decode_inputs, "prefill": prefill_inputs},
            strategy=MULTIMETHOD_WEIGHT_SHARING_STRATEGY.ONE_BLOB,
        )

        self.assertIn("forward", edge_manager.methods)
        self.assertIn("prefill", edge_manager.methods)

        if _TEST_RUNTIME:
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            self.assertIn("forward", program.method_names)
            self.assertIn("prefill", program.method_names)

            forward_method = program.load_method("forward")
            decode_output = forward_method.execute(decode_inputs)
            expected_decode = model(*decode_inputs)
            self.assertTrue(
                torch.allclose(decode_output[0], expected_decode, atol=1e-4, rtol=1e-4)
            )

            prefill_method = program.load_method("prefill")
            prefill_output = prefill_method.execute(prefill_inputs)
            expected_prefill = model(*prefill_inputs)
            self.assertTrue(
                torch.allclose(
                    prefill_output[0], expected_prefill, atol=1e-4, rtol=1e-4
                )
            )


if __name__ == "__main__":
    test_runner = TestCoreMLMultifunction()
    test_runner.test_multifunction_simple_model()
    test_runner.test_multifunction_with_kv_cache()
    test_runner.test_multifunction_without_weight_sharing()
    test_runner.test_multifunction_with_constant_methods()
    test_runner.test_multifunction_one_blob_simple_model()
    print("All tests passed!")
