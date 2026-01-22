# Copyright © 2024 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import sys
import unittest

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


_TEST_RUNTIME = (sys.platform == "darwin") and not is_fbcode()
if _TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestCoreMLMultifunction(unittest.TestCase):
    """Tests for multifunction (multi-method) CoreML model export."""

    edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)

    def _get_compile_specs(self, weight_sharing: bool = True):
        """Get compile specs for multifunction models."""
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT32,
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            model_type=CoreMLBackend.MODEL_TYPE.MODEL,
        )
        if weight_sharing:
            compile_specs.append(
                CoreMLBackend.generate_multimethod_weight_sharing_strategy_compile_spec(
                    MULTIMETHOD_WEIGHT_SHARING_STRATEGY.POSITIONAL
                )
            )
        return compile_specs

    def test_multifunction_simple_model(self):
        """Test exporting a simple model with multiple methods (forward and prefill)."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()

        # Create example inputs for two different sequence lengths
        decode_inputs = (torch.randn(1, 1, 16),)  # seqlen=1
        prefill_inputs = (torch.randn(1, 8, 16),)  # seqlen=8

        # Export both methods
        decode_ep = torch.export.export(model, decode_inputs)
        prefill_ep = torch.export.export(model, prefill_inputs)

        # Create partitioner with multifunction support
        partitioner = CoreMLPartitioner(
            compile_specs=self._get_compile_specs(weight_sharing=True),
        )

        # Lower to edge with multiple methods
        edge_manager = to_edge_transform_and_lower(
            {"forward": decode_ep, "prefill": prefill_ep},
            partitioner=[partitioner],
            compile_config=self.edge_compile_config,
        )

        # Verify both methods exist
        method_names = edge_manager.methods
        self.assertIn("forward", method_names)
        self.assertIn("prefill", method_names)

        # Convert to ExecuTorch
        et_program = edge_manager.to_executorch()

        if _TEST_RUNTIME:
            # Test runtime execution
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            # Check both methods are available
            available_methods = program.method_names
            self.assertIn("forward", available_methods)
            self.assertIn("prefill", available_methods)

            # Test forward (decode) method
            forward_method = program.load_method("forward")
            decode_output = forward_method.execute(decode_inputs)
            expected_decode = model(*decode_inputs)
            self.assertTrue(
                torch.allclose(decode_output[0], expected_decode, atol=1e-4, rtol=1e-4)
            )

            # Test prefill method
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
                # x: [batch, seqlen, hidden_dim]
                # cache: [batch, cache_len, hidden_dim]
                out = self.linear(x)
                # Simple cache update simulation
                new_cache = torch.cat([cache[:, 1:, :], out[:, -1:, :]], dim=1)
                return out, new_cache

        hidden_dim = 16
        cache_len = 32
        model = ModelWithCache(hidden_dim, cache_len)
        model.eval()

        # Decode: seqlen=1
        decode_inputs = (
            torch.randn(1, 1, hidden_dim),
            torch.randn(1, cache_len, hidden_dim),
        )

        # Prefill: seqlen=8
        prefill_inputs = (
            torch.randn(1, 8, hidden_dim),
            torch.randn(1, cache_len, hidden_dim),
        )

        decode_ep = torch.export.export(model, decode_inputs)
        prefill_ep = torch.export.export(model, prefill_inputs)

        partitioner = CoreMLPartitioner(
            compile_specs=self._get_compile_specs(weight_sharing=True),
        )

        edge_manager = to_edge_transform_and_lower(
            {"forward": decode_ep, "prefill": prefill_ep},
            partitioner=[partitioner],
            compile_config=self.edge_compile_config,
        )

        et_program = edge_manager.to_executorch()

        if _TEST_RUNTIME:
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            # Test decode
            forward_method = program.load_method("forward")
            decode_output = forward_method.execute(decode_inputs)
            expected_out, expected_cache = model(*decode_inputs)
            self.assertTrue(
                torch.allclose(decode_output[0], expected_out, atol=1e-4, rtol=1e-4)
            )
            self.assertTrue(
                torch.allclose(decode_output[1], expected_cache, atol=1e-4, rtol=1e-4)
            )

            # Test prefill
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

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()

        decode_inputs = (torch.randn(1, 1, 16),)
        prefill_inputs = (torch.randn(1, 8, 16),)

        decode_ep = torch.export.export(model, decode_inputs)
        prefill_ep = torch.export.export(model, prefill_inputs)

        # Create partitioner without weight sharing
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT32,
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            model_type=CoreMLBackend.MODEL_TYPE.MODEL,
        )
        compile_specs.append(
            CoreMLBackend.generate_multimethod_weight_sharing_strategy_compile_spec(
                MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED
            )
        )

        partitioner = CoreMLPartitioner(compile_specs=compile_specs)

        edge_manager = to_edge_transform_and_lower(
            {"forward": decode_ep, "prefill": prefill_ep},
            partitioner=[partitioner],
            compile_config=self.edge_compile_config,
        )

        method_names = edge_manager.methods
        self.assertIn("forward", method_names)
        self.assertIn("prefill", method_names)

        et_program = edge_manager.to_executorch()

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

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()

        decode_inputs = (torch.randn(1, 1, 16),)
        prefill_inputs = (torch.randn(1, 8, 16),)

        decode_ep = torch.export.export(model, decode_inputs)
        prefill_ep = torch.export.export(model, prefill_inputs)

        partitioner = CoreMLPartitioner(
            compile_specs=self._get_compile_specs(weight_sharing=True),
        )

        # Add constant methods (metadata)
        constant_methods = {
            "vocab_size": 32000,
            "hidden_dim": 16,
            "decode_seqlen": 1,
            "prefill_seqlen": 8,
        }

        edge_manager = to_edge_transform_and_lower(
            {"forward": decode_ep, "prefill": prefill_ep},
            partitioner=[partitioner],
            constant_methods=constant_methods,
            compile_config=self.edge_compile_config,
        )

        et_program = edge_manager.to_executorch()

        if _TEST_RUNTIME:
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)

            # Check all methods are available (executable + constant)
            available_methods = program.method_names
            self.assertIn("forward", available_methods)
            self.assertIn("prefill", available_methods)
            self.assertIn("vocab_size", available_methods)
            self.assertIn("hidden_dim", available_methods)
            self.assertIn("decode_seqlen", available_methods)
            self.assertIn("prefill_seqlen", available_methods)


if __name__ == "__main__":
    test_runner = TestCoreMLMultifunction()
    test_runner.test_multifunction_simple_model()
    test_runner.test_multifunction_with_kv_cache()
    test_runner.test_multifunction_without_weight_sharing()
    test_runner.test_multifunction_with_constant_methods()
    print("All tests passed!")
