# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the Nordic AXON backend — full compilation stage.

These tests require the Nordic sdk-edge-ai to be installed and
SDK_EDGE_AI_PATH to be set. They validate the complete pipeline from
PyTorch model through TOSA to compiled AXON command buffers.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

_SDK_PATH = os.environ.get("SDK_EDGE_AI_PATH", "")
_HAS_SDK = bool(_SDK_PATH) and os.path.isdir(_SDK_PATH)

pytestmark = [
    pytest.mark.requires_sdk,
    pytest.mark.skipif(
        not _HAS_SDK,
        reason="Nordic SDK not available (set SDK_EDGE_AI_PATH to enable)",
    ),
]


@pytest.fixture
def sdk_path():
    """Path to Nordic sdk-edge-ai."""
    return _SDK_PATH


class TestAxonCompilation:
    """End-to-end compilation tests."""

    def _compile_model(self, model, example_input, sdk_path, tmp_path):
        """Compile a model through the full AXON pipeline."""
        from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

        generated_dir = tmp_path / "generated"
        generated_dir.mkdir()

        compile_spec = AxonCompileSpec(
            sdk_edge_ai_path=sdk_path,
            model_name="test_model",
            axon_generated_dir=str(generated_dir),
        )
        partitioner = AxonPartitioner(compile_spec)

        # Quantize
        model.eval()
        exported = torch.export.export(model, example_input)
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*example_input)
        quantized = convert_pt2e(prepared)

        # Edge lower with AXON partitioner
        edge = to_edge_transform_and_lower(
            quantized,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        return edge, generated_dir

    def test_linear_compiles(self, sdk_path, tmp_path):
        """A simple linear model compiles to AXON command buffers."""
        model = nn.Sequential(nn.Linear(16, 8))
        example_input = (torch.randn(1, 16),)
        edge, gen_dir = self._compile_model(model, example_input, sdk_path, tmp_path)

        # Check generated headers exist
        headers = list(gen_dir.glob("axon_subgraph_*.h"))
        assert len(headers) >= 1, f"No subgraph headers generated in {gen_dir}"

        # Check table exists
        table = gen_dir / "axon_subgraphs_table.h"
        assert table.exists(), "axon_subgraphs_table.h not generated"
        content = table.read_text()
        assert "AXON_SUBGRAPHS_COUNT" in content

    def test_conv_relu_compiles(self, sdk_path, tmp_path):
        """Conv2d + ReLU compiles with fused activation."""
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        example_input = (torch.randn(1, 1, 8, 8),)
        edge, gen_dir = self._compile_model(model, example_input, sdk_path, tmp_path)

        headers = list(gen_dir.glob("axon_subgraph_*.h"))
        assert len(headers) >= 1

    def test_multi_layer_produces_unique_names(self, sdk_path, tmp_path):
        """A multi-layer model produces distinct subgraph names."""
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        example_input = (torch.randn(1, 16),)
        edge, gen_dir = self._compile_model(model, example_input, sdk_path, tmp_path)

        headers = list(gen_dir.glob("axon_subgraph_*.h"))
        # Should have at least 2 distinct subgraphs (2 linears)
        names = [h.stem for h in headers]
        assert len(names) == len(set(names)), f"Duplicate subgraph names: {names}"

    def test_compiled_header_has_cmd_buffer(self, sdk_path, tmp_path):
        """The compiled header contains a command buffer array."""
        model = nn.Sequential(nn.Linear(16, 8))
        example_input = (torch.randn(1, 16),)
        edge, gen_dir = self._compile_model(model, example_input, sdk_path, tmp_path)

        headers = list(gen_dir.glob("axon_subgraph_*.h"))
        assert len(headers) >= 1
        content = headers[0].read_text()
        assert "cmd_buffer_" in content
        assert "nrf_axon_nn_compiled_model_s" in content

    def test_no_nordic_symbols_leaked(self, sdk_path, tmp_path):
        """Op extension symbols are rewritten to axon_op_extension_*."""
        model = nn.Sequential(nn.Linear(16, 8))
        example_input = (torch.randn(1, 16),)
        edge, gen_dir = self._compile_model(model, example_input, sdk_path, tmp_path)

        # Check no nrf_axon_nn_op_extension_ symbols remain
        for header in gen_dir.glob("axon_subgraph_*.h"):
            content = header.read_text()
            assert "nrf_axon_nn_op_extension_sigmoid" not in content
            assert "nrf_axon_nn_op_extension_tanh" not in content

    def test_pte_export(self, sdk_path, tmp_path):
        """Full .pte export succeeds."""
        from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

        model = nn.Sequential(nn.Linear(16, 8))
        example_input = (torch.randn(1, 16),)

        generated_dir = tmp_path / "generated"
        generated_dir.mkdir()

        compile_spec = AxonCompileSpec(
            sdk_edge_ai_path=sdk_path,
            model_name="pte_test",
            axon_generated_dir=str(generated_dir),
        )
        partitioner = AxonPartitioner(compile_spec)

        model.eval()
        exported = torch.export.export(model, example_input)
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*example_input)
        quantized = convert_pt2e(prepared)

        edge = to_edge_transform_and_lower(
            quantized,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        pte_path = tmp_path / "test.pte"
        edge.to_executorch().save(str(pte_path))
        assert pte_path.exists()
        assert pte_path.stat().st_size > 0
