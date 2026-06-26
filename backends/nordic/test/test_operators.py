# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Per-operator tests for the Nordic AXON backend.

Validates that each supported operation type can be:
1. Exported from a PyTorch model
2. Lowered through the TOSA pipeline
3. Converted to AXON layer descriptors
4. Packed into an intermediate binary

These tests run without the Nordic SDK (no compilation to command
buffers — just TOSA lowering and AXON layer conversion).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from executorch.backends.nordic.axon_types import AxonOp


_test_counter = 0


def _lower_to_axon_layers(model, example_input):
    """Export, quantize, and lower a model to AXON layers."""
    global _test_counter
    _test_counter += 1
    model_name = f"optest_{_test_counter}"

    from executorch.backends.arm.tosa.specification import TosaSpecification
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    import tempfile, os

    model.eval()
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    exported = torch.export.export(model, example_input, strict=False)
    captured = exported.module()

    # Strip torch 2.11 _guards_fn nodes
    guard_nodes = [
        n for n in captured.graph.nodes
        if n.op == "call_module" and "_guards" in str(n.target)
    ]
    for n in guard_nodes:
        n.replace_all_uses_with(None)
        captured.graph.erase_node(n)
    for name in list(captured._modules.keys()):
        if "_guards" in name:
            delattr(captured, name)
    captured.graph.lint()
    captured.recompile()

    quantizer = EthosUQuantizer(tosa_spec).set_global(
        get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = prepare_pt2e(captured, quantizer)
    prepared(*example_input)
    quantized = convert_pt2e(prepared)
    re_exported = torch.export.export(quantized, example_input, strict=False)

    compile_spec = AxonCompileSpec(model_name=model_name)
    partitioner = AxonPartitioner(compile_spec)
    to_edge_transform_and_lower(
        re_exported,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    # Read the TOSA debug file (unique per test via model_name)
    tosa_path = os.path.join(tempfile.gettempdir(), f"axon_tosa_debug_{model_name}.tosa")
    if not os.path.exists(tosa_path):
        pytest.skip("TOSA debug file not generated")

    from executorch.backends.nordic.tosa_reader import parse_tosa_flatbuffer
    from executorch.backends.nordic.axon_compiler import tosa_to_axon_layers

    with open(tosa_path, "rb") as f:
        tosa_bytes = f.read()
    graph = parse_tosa_flatbuffer(tosa_bytes)
    return tosa_to_axon_layers(graph)


class TestLinearOp:
    """Test FC (fully connected) layer delegation."""

    def test_simple_linear(self):
        model = nn.Sequential(nn.Linear(16, 8))
        layers = _lower_to_axon_layers(model, (torch.randn(1, 16),))
        assert len(layers) >= 1
        # TOSA lowers Linear to CONV2D
        compute = [l for l in layers if l.operation in (AxonOp.FULLY_CONNECTED, AxonOp.CONV2D)]
        assert len(compute) >= 1

    def test_linear_with_relu(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        layers = _lower_to_axon_layers(model, (torch.randn(1, 16),))
        compute = [l for l in layers if l.operation in (AxonOp.FULLY_CONNECTED, AxonOp.CONV2D)]
        assert len(compute) >= 1

    def test_multi_linear(self):
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        layers = _lower_to_axon_layers(model, (torch.randn(1, 16),))
        # Multiple linears should produce multiple compute layers
        assert len(layers) >= 2


class TestConv2dOp:
    """Test Conv2D layer delegation."""

    def test_simple_conv2d(self):
        model = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, padding=1))
        layers = _lower_to_axon_layers(model, (torch.randn(1, 1, 8, 8),))
        conv_layers = [l for l in layers if l.operation == AxonOp.CONV2D]
        assert len(conv_layers) >= 1

    def test_conv2d_relu(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        layers = _lower_to_axon_layers(model, (torch.randn(1, 1, 8, 8),))
        # ReLU gets fused into the conv, so still just conv layers
        assert len(layers) >= 1

    def test_conv2d_different_filters(self):
        """Test various filter sizes within AXON limits."""
        for k in [1, 3, 5, 7]:
            model = nn.Sequential(nn.Conv2d(1, 4, kernel_size=k, padding=k // 2))
            layers = _lower_to_axon_layers(model, (torch.randn(1, 1, 8, 8),))
            assert len(layers) >= 1, f"Failed for kernel_size={k}"


class TestPoolOp:
    """Test pooling layer delegation."""

    def test_avg_pool2d(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        layers = _lower_to_axon_layers(model, (torch.randn(1, 1, 8, 8),))
        pool_layers = [l for l in layers if l.operation == AxonOp.AVERAGE_POOLING]
        assert len(pool_layers) >= 1

    def test_max_pool2d(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        layers = _lower_to_axon_layers(model, (torch.randn(1, 1, 8, 8),))
        pool_layers = [l for l in layers if l.operation == AxonOp.MAX_POOLING]
        assert len(pool_layers) >= 1


class TestElementwiseOps:
    """Test element-wise operations (add, multiply)."""

    def test_add(self):
        class AddModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
                self.conv2 = nn.Conv2d(1, 4, 3, padding=1)

            def forward(self, x):
                return self.conv1(x) + self.conv2(x)

        layers = _lower_to_axon_layers(AddModel(), (torch.randn(1, 1, 8, 8),))
        add_layers = [l for l in layers if l.operation == AxonOp.ADD2]
        assert len(add_layers) >= 1

    def test_multiply(self):
        class MulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
                self.conv2 = nn.Conv2d(1, 4, 3, padding=1)

            def forward(self, x):
                return self.conv1(x) * self.conv2(x)

        layers = _lower_to_axon_layers(MulModel(), (torch.randn(1, 1, 8, 8),))
        mul_layers = [l for l in layers if l.operation == AxonOp.MULTIPLY]
        assert len(mul_layers) >= 1


class TestBinaryBuilder:
    """Test that AXON layers pack into valid intermediate binaries."""

    def test_linear_binary(self):
        from executorch.backends.nordic.axon_binary import AxonBinaryBuilder

        model = nn.Sequential(nn.Linear(16, 8))
        layers = _lower_to_axon_layers(model, (torch.randn(1, 16),))
        builder = AxonBinaryBuilder()
        binary = builder.build(layers, model_name="test_linear")
        assert len(binary) > 100
        assert b"AXON_INTERMEDIATE_REPRESENTATION_FILE" in binary

    def test_conv_binary(self):
        from executorch.backends.nordic.axon_binary import AxonBinaryBuilder

        model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
        layers = _lower_to_axon_layers(model, (torch.randn(1, 1, 8, 8),))
        builder = AxonBinaryBuilder()
        binary = builder.build(layers, model_name="test_conv")
        assert len(binary) > 100


class TestConstraintChecks:
    """Test that AXON constraint checks work via the partitioner."""

    def test_axon_constraints_importable(self):
        from executorch.backends.nordic.operator_support.axon_constraints import (
            AxonTensorDimensionCheck,
            AxonInputCountCheck,
            AxonConvConstraintCheck,
            AxonFCConstraintCheck,
            get_axon_constraint_checks,
        )
        checks = get_axon_constraint_checks()
        assert len(checks) == 4

    def test_partitioner_with_constraints(self):
        from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
        from executorch.backends.nordic.operator_support.axon_constraints import (
            get_axon_constraint_checks,
        )
        spec = AxonCompileSpec(model_name="test")
        # Constraints are opt-in via additional_checks
        partitioner = AxonPartitioner(spec, additional_checks=get_axon_constraint_checks())
        assert len(partitioner.additional_checks) >= 4

    def test_partitioner_default_no_constraints(self):
        from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
        spec = AxonCompileSpec(model_name="test")
        partitioner = AxonPartitioner(spec)
        assert len(partitioner.additional_checks) == 0
