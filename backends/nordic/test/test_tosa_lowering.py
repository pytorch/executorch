# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the Nordic AXON backend — TOSA lowering stage.

These tests validate that PyTorch models lower correctly through the
TOSA pipeline and produce valid AXON layer descriptors. No Nordic SDK
is required — these run on any machine with ExecuTorch and PyTorch.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestAxonImports:
    """Verify the backend package imports correctly."""

    def test_lazy_imports(self):
        from executorch.backends.nordic import (
            AxonBackend,
            AxonCompileSpec,
            AxonPartitioner,
        )
        assert AxonBackend is not None
        assert AxonCompileSpec is not None
        assert AxonPartitioner is not None

    def test_direct_imports(self):
        from executorch.backends.nordic.axon import AxonBackend
        from executorch.backends.nordic.axon.compile_spec import AxonCompileSpec
        from executorch.backends.nordic.axon.partitioner import AxonPartitioner
        assert AxonBackend.__name__ == "AxonBackend"

    def test_operator_support(self):
        from executorch.backends.nordic.operator_support import (
            AXON_SUPPORTED_OPS,
            AXON_FUSED_ACTIVATIONS,
            AXON_OP_EXTENSIONS,
            check_fully_connected,
            check_conv2d,
            check_pooling,
        )
        assert "fully_connected" in AXON_SUPPORTED_OPS
        assert "conv2d" in AXON_SUPPORTED_OPS
        assert "pointwise_conv2d" in AXON_SUPPORTED_OPS
        assert "channel_padding" in AXON_SUPPORTED_OPS
        assert len(AXON_SUPPORTED_OPS) >= 12
        assert "relu" in AXON_FUSED_ACTIVATIONS
        assert "sigmoid" in AXON_OP_EXTENSIONS
        assert "tanh" in AXON_OP_EXTENSIONS
        assert "softmax" in AXON_OP_EXTENSIONS

    def test_quantizer_import(self):
        from executorch.backends.nordic.axon import AxonQuantizer
        q = AxonQuantizer()
        assert q is not None

    def test_quantizer_lazy_import(self):
        from executorch.backends.nordic import AxonQuantizer
        assert AxonQuantizer is not None


class TestOperatorConstraints:
    """Validate AXON hardware constraint checks."""

    def test_fc_within_limits(self):
        from executorch.backends.nordic.operator_support import check_fully_connected
        ok, msg = check_fully_connected(128, 64)
        assert ok is True

    def test_fc_max_input(self):
        from executorch.backends.nordic.operator_support import check_fully_connected
        ok, msg = check_fully_connected(2048, 64)
        assert ok is True

    def test_fc_exceeds_input(self):
        from executorch.backends.nordic.operator_support import check_fully_connected
        ok, msg = check_fully_connected(4096, 64)
        assert ok is False
        assert "4096" in msg

    def test_fc_exceeds_output(self):
        from executorch.backends.nordic.operator_support import check_fully_connected
        ok, msg = check_fully_connected(128, 4096)
        assert ok is False

    def test_conv2d_within_limits(self):
        from executorch.backends.nordic.operator_support import check_conv2d
        ok, msg = check_conv2d(3, 3, 1, 1, 32)
        assert ok is True

    def test_conv2d_exceeds_filter(self):
        from executorch.backends.nordic.operator_support import check_conv2d
        ok, msg = check_conv2d(32, 32, 1, 1, 32)
        assert ok is False

    def test_pooling_within_limits(self):
        from executorch.backends.nordic.operator_support import check_pooling
        ok, msg = check_pooling(2, 2)
        assert ok is True

    def test_pooling_exceeds_filter(self):
        from executorch.backends.nordic.operator_support import check_pooling
        ok, msg = check_pooling(64, 64)
        assert ok is False

    def test_tensor_dims_within_limits(self):
        from executorch.backends.nordic.operator_support import check_tensor_dimensions
        ok, msg = check_tensor_dimensions(512, 512, 64)
        assert ok is True

    def test_tensor_dims_at_max(self):
        from executorch.backends.nordic.operator_support import check_tensor_dimensions
        ok, msg = check_tensor_dimensions(1024, 1024, 1024)
        assert ok is True

    def test_tensor_dims_exceeds_height(self):
        from executorch.backends.nordic.operator_support import check_tensor_dimensions
        ok, msg = check_tensor_dimensions(2048, 512, 64)
        assert ok is False
        assert "height" in msg

    def test_tensor_dims_exceeds_channels(self):
        from executorch.backends.nordic.operator_support import check_tensor_dimensions
        ok, msg = check_tensor_dimensions(8, 8, 2048)
        assert ok is False
        assert "channels" in msg

    def test_input_count_valid(self):
        from executorch.backends.nordic.operator_support import check_input_count
        ok, msg = check_input_count(1)
        assert ok is True
        ok, msg = check_input_count(2)
        assert ok is True

    def test_input_count_exceeds(self):
        from executorch.backends.nordic.operator_support import check_input_count
        ok, msg = check_input_count(3)
        assert ok is False
        assert "3" in msg

    def test_conv2d_exceeds_stride(self):
        from executorch.backends.nordic.operator_support import check_conv2d
        ok, msg = check_conv2d(3, 3, 32, 32, 16)
        assert ok is False
        assert "stride" in msg

    def test_conv2d_exceeds_channels(self):
        from executorch.backends.nordic.operator_support import check_conv2d
        ok, msg = check_conv2d(3, 3, 1, 1, 2048)
        assert ok is False
        assert "channels" in msg


class TestCompileSpec:
    """Validate AxonCompileSpec serialization."""

    def test_default_spec(self):
        from executorch.backends.nordic.axon import AxonCompileSpec
        spec = AxonCompileSpec()
        compile_specs = spec.to_compile_specs()
        keys = {s.key for s in compile_specs}
        assert "tosa_spec" in keys
        assert "output_format" in keys
        assert "model_name" in keys

    def test_custom_spec(self):
        from executorch.backends.nordic.axon import AxonCompileSpec
        spec = AxonCompileSpec(
            sdk_edge_ai_path="/opt/sdk-edge-ai",
            model_name="test_model",
            axon_generated_dir="/tmp/generated",
        )
        compile_specs = spec.to_compile_specs()
        keys = {s.key for s in compile_specs}
        assert "sdk_edge_ai_path" in keys
        assert "axon_generated_dir" in keys

    def test_spec_without_sdk(self):
        from executorch.backends.nordic.axon import AxonCompileSpec
        spec = AxonCompileSpec(model_name="no_sdk")
        compile_specs = spec.to_compile_specs()
        keys = {s.key for s in compile_specs}
        assert "sdk_edge_ai_path" not in keys


class TestCodegen:
    """Validate codegen utilities."""

    def test_make_marker(self):
        from executorch.backends.nordic.axon.codegen import make_marker
        marker = make_marker("test_model_abc123")
        assert marker[:4] == b"AXNG"
        assert len(marker) % 4 == 0

    def test_derive_subgraph_name(self):
        from executorch.backends.nordic.axon.codegen import derive_subgraph_name
        name1 = derive_subgraph_name("model", b"binary_data_1")
        name2 = derive_subgraph_name("model", b"binary_data_2")
        name3 = derive_subgraph_name("model", b"binary_data_1")
        # Different content → different names
        assert name1 != name2
        # Same content → same name
        assert name1 == name3
        # Starts with prefix
        assert name1.startswith("model_")

    def test_rewrite_header_symbols(self):
        from executorch.backends.nordic.axon.codegen import rewrite_header_symbols
        header = 'const int model_old_name = 1;\n.model_name = "old_name"'
        result = rewrite_header_symbols(header, "old_name", "new_name")
        assert "model_new_name" in result
        assert '.model_name = "new_name"' in result

    def test_rewrite_op_extension_symbols(self):
        from executorch.backends.nordic.axon.codegen import rewrite_op_extension_symbols
        header = "extern void nrf_axon_nn_op_extension_sigmoid(void);"
        result = rewrite_op_extension_symbols(header)
        assert "axon_op_extension_sigmoid" in result
        assert "nrf_axon_nn_op_extension_sigmoid" not in result

    def test_write_and_regenerate(self, tmp_path):
        from executorch.backends.nordic.axon.codegen import (
            write_subgraph_header,
            regenerate_table,
            clean_generated_dir,
        )
        # Write two subgraph headers
        write_subgraph_header(tmp_path, "sub_aaa", "/* header A */\n")
        write_subgraph_header(tmp_path, "sub_bbb", "/* header B */\n")
        # Regenerate table
        table_path = regenerate_table(tmp_path)
        assert table_path.exists()
        content = table_path.read_text()
        assert "AXON_SUBGRAPHS_COUNT 2" in content
        assert '"sub_aaa"' in content
        assert '"sub_bbb"' in content
        # Clean
        removed = clean_generated_dir(tmp_path)
        assert removed == 3  # 2 subgraph headers + 1 table


class TestTosaLowering:
    """Test TOSA lowering for simple models.

    These tests export a simple PyTorch model through ExecuTorch's
    edge lowering and TOSA conversion pipeline. They validate that
    the AXON backend can process the TOSA flatbuffer into AXON layer
    descriptors without requiring the Nordic compiler.
    """

    def _export_to_tosa(self, model, example_input):
        """Export a model through the AXON backend, returning the TOSA
        flatbuffer from the first delegated subgraph.

        Uses the full edge-lower pipeline (same as real deployment) so
        the ARM pass pipeline handles quantized weight decomposition.
        """
        import tempfile
        from executorch.backends.arm.tosa.specification import TosaSpecification
        from executorch.backends.arm.quantizer import (
            EthosUQuantizer,
            get_symmetric_quantization_config,
        )
        from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

        model.eval()
        tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")

        # Quantize
        exported = torch.export.export(model, example_input)
        quantizer = EthosUQuantizer(tosa_spec).set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        prepared = prepare_pt2e(exported.module(), quantizer)
        prepared(*example_input)
        quantized = convert_pt2e(prepared)
        re_exported = torch.export.export(quantized, example_input)

        # Edge lower with AXON partitioner (no SDK needed — returns marker only)
        # Use unique model name to avoid TOSA debug file collisions between tests
        if not hasattr(self, '_tosa_test_counter'):
            type(self)._tosa_test_counter = 0
        type(self)._tosa_test_counter += 1
        model_name = f"tosatest_{self._tosa_test_counter}"

        compile_spec = AxonCompileSpec(model_name=model_name)
        partitioner = AxonPartitioner(compile_spec)
        edge = to_edge_transform_and_lower(
            re_exported,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        # Read the debug TOSA flatbuffer that AxonBackend.preprocess writes
        import os
        tosa_path = os.path.join(tempfile.gettempdir(), f"axon_tosa_debug_{model_name}.tosa")
        if os.path.exists(tosa_path):
            with open(tosa_path, "rb") as f:
                return f.read()

        pytest.skip("TOSA debug file not generated — backend may have skipped")

    def test_simple_linear_to_tosa(self):
        """A simple linear layer lowers to TOSA successfully."""
        model = nn.Sequential(nn.Linear(16, 8))
        example_input = (torch.randn(1, 16),)
        tosa_bytes = self._export_to_tosa(model, example_input)
        assert len(tosa_bytes) > 0

        # Parse the TOSA flatbuffer
        from executorch.backends.nordic.tosa_reader import parse_tosa_flatbuffer
        graph = parse_tosa_flatbuffer(tosa_bytes)
        assert len(graph.operators) > 0
        op_names = [op.op_name for op in graph.get_non_const_operators()]
        # TOSA represents FC as CONV2D with reshapes
        assert "CONV2D" in op_names or "FULLY_CONNECTED" in op_names

    def test_linear_to_axon_layers(self):
        """A simple linear layer converts to AXON layers."""
        model = nn.Sequential(nn.Linear(16, 8))
        example_input = (torch.randn(1, 16),)
        tosa_bytes = self._export_to_tosa(model, example_input)

        from executorch.backends.nordic.tosa_reader import parse_tosa_flatbuffer
        from executorch.backends.nordic.axon_compiler import tosa_to_axon_layers

        graph = parse_tosa_flatbuffer(tosa_bytes)
        layers = tosa_to_axon_layers(graph)
        assert len(layers) >= 1
        # AXON compiler converts TOSA ops to AXON layer descriptors
        # FC may appear as FULLY_CONNECTED (0) or CONV2D (1) depending on TOSA lowering
        from executorch.backends.nordic.axon_compiler import AxonOp
        compute_layers = [l for l in layers if l.operation in (
            AxonOp.FULLY_CONNECTED, AxonOp.CONV2D, AxonOp.POINTWISE_CONV2D,
        )]
        assert len(compute_layers) >= 1

    def test_conv2d_to_axon_layers(self):
        """A Conv2d layer converts to AXON layers."""
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        example_input = (torch.randn(1, 1, 8, 8),)
        tosa_bytes = self._export_to_tosa(model, example_input)

        from executorch.backends.nordic.tosa_reader import parse_tosa_flatbuffer
        from executorch.backends.nordic.axon_compiler import tosa_to_axon_layers

        graph = parse_tosa_flatbuffer(tosa_bytes)
        layers = tosa_to_axon_layers(graph)
        assert len(layers) >= 1

    def test_binary_builder(self):
        """AXON binary builder produces non-empty output."""
        model = nn.Sequential(nn.Linear(16, 8))
        example_input = (torch.randn(1, 16),)
        tosa_bytes = self._export_to_tosa(model, example_input)

        from executorch.backends.nordic.tosa_reader import parse_tosa_flatbuffer
        from executorch.backends.nordic.axon_compiler import tosa_to_axon_layers
        from executorch.backends.nordic.axon_binary import AxonBinaryBuilder

        graph = parse_tosa_flatbuffer(tosa_bytes)
        layers = tosa_to_axon_layers(graph)
        builder = AxonBinaryBuilder()
        binary = builder.build(layers, model_name="test_linear")
        assert len(binary) > 100  # Header alone is ~100 bytes
        # Verify it contains the title string
        assert b"AXON_INTERMEDIATE_REPRESENTATION_FILE" in binary
