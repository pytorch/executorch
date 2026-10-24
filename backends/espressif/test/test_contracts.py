# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contract tests for the partitioner, quantizer, preprocess and serializer:
strict delegation, per-module quantization configs, and the device runtime
limits that must be caught on the host.

    pytest backends/espressif/test/test_contracts.py
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("torchao")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from executorch.backends.espressif.espnn_preprocess import (  # noqa: E402
    EspNNBackend,
    INT8_FORMAT_SPEC,
)
from executorch.backends.espressif.lowering import to_espnn_edge  # noqa: E402
from executorch.backends.espressif.partition import espnn_partitioner  # noqa: E402
from executorch.backends.espressif.quantizer.espnn_quantizer import (  # noqa: E402
    EspNNQuantizer,
    fused_activation_kind,
)
from executorch.backends.espressif.serialization.xts3 import (  # noqa: E402
    build_graph,
    check_runtime_limits,
    MAX_GRAPH_IO,
    MAX_GRAPH_OPERATIONS,
    TENSOR_FLAG_CONSTANT,
    XTS3DType,
    XTS3OpCode,
)
from executorch.backends.espressif.test.test_int8_ops import quantize  # noqa: E402
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (  # noqa: E402
    get_symmetric_quantization_config,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e  # noqa: E402


class FeaturesAndClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Conv2d(8, 4, 1), nn.ReLU())

    def forward(self, x):
        return self.classifier(self.features(x))


def _small_graph():
    _, program = quantize(FeaturesAndClassifier(), [(1, 3, 8, 8)])
    return build_graph(program)


def test_strict_mode_raises_when_a_region_stays_on_cpu(monkeypatch):
    _, program = quantize(FeaturesAndClassifier(), [(1, 3, 8, 8)])
    monkeypatch.setattr(
        espnn_partitioner, "_boundary_is_valid", lambda nodes, resolver: False
    )
    with pytest.raises(RuntimeError, match="would run on the CPU"):
        to_espnn_edge(program, allow_cpu_fallback=False)


def test_fallback_mode_reports_rejected_regions(monkeypatch):
    _, program = quantize(FeaturesAndClassifier(), [(1, 3, 8, 8)])
    monkeypatch.setattr(
        espnn_partitioner, "_boundary_is_valid", lambda nodes, resolver: False
    )
    _, report = to_espnn_edge(program)
    assert report["partitions"] == 0
    assert report["rejected_partitions"] >= 1
    assert report["cpu_ops"]


def _weight_observer(prepared, conv_name_fragment):
    modules = dict(prepared.named_modules())
    for node in prepared.graph.nodes:
        if node.op != "call_function" or "conv" not in str(node.target):
            continue
        stack = node.meta.get("nn_module_stack", {})
        if not any(conv_name_fragment in path for path, _ in stack.values()):
            continue
        weight = node.args[1]
        observer = weight if weight.op == "call_module" else weight.args[0]
        return type(modules[str(observer.target)]).__name__
    raise AssertionError(f"no conv under {conv_name_fragment}")


def test_module_name_config_reaches_fused_conv_relu():
    model = FeaturesAndClassifier().eval()
    example = (torch.randn(1, 3, 8, 8),)
    per_tensor = get_symmetric_quantization_config(is_per_channel=False)
    quantizer = EspNNQuantizer().set_module_name("classifier", per_tensor)
    prepared = prepare_pt2e(
        torch.export.export(model, example, strict=True).module(), quantizer
    )
    assert "PerChannel" in _weight_observer(prepared, "features")
    assert "PerChannel" not in _weight_observer(prepared, "classifier")


def test_hardtanh_bounds_given_as_kwargs_fuse_as_relu6():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    by_kwargs = graph.call_function(
        torch.ops.aten.hardtanh.default, (x,), {"min_val": 0.0, "max_val": 6.0}
    )
    by_args = graph.call_function(torch.ops.aten.hardtanh.default, (x, 0.0, 6.0))
    other = graph.call_function(torch.ops.aten.hardtanh.default, (x,))
    assert fused_activation_kind(by_kwargs) == "relu6"
    assert fused_activation_kind(by_args) == "relu6"
    assert fused_activation_kind(other) is None


def test_preprocess_rejects_unknown_format_spec():
    with pytest.raises(ValueError, match="espnn_format"):
        EspNNBackend.preprocess(None, [CompileSpec(INT8_FORMAT_SPEC.key, b"float")])


def test_runtime_limits_accept_a_lowered_graph():
    check_runtime_limits(_small_graph())


def test_runtime_limits_reject_too_many_outputs():
    graph = _small_graph()
    graph = dataclasses.replace(graph, outputs=list(range(MAX_GRAPH_IO + 1)))
    with pytest.raises(ValueError, match="outputs"):
        check_runtime_limits(graph)


def test_runtime_limits_reject_too_many_operations():
    graph = _small_graph()
    operations = graph.operations * (MAX_GRAPH_OPERATIONS // len(graph.operations) + 1)
    with pytest.raises(ValueError, match="operations"):
        check_runtime_limits(dataclasses.replace(graph, operations=operations))


def test_runtime_limits_reject_v6_opcodes_in_v5_graphs():
    graph = _small_graph()
    copy = dataclasses.replace(graph.operations[0], opcode=XTS3OpCode.COPY)
    with pytest.raises(ValueError, match="version 6"):
        check_runtime_limits(
            dataclasses.replace(
                graph, version=5, operations=[copy], inputs=[], outputs=[]
            )
        )


def test_runtime_limits_reject_misaligned_int32_constants():
    graph = _small_graph()
    tensors = [
        (
            dataclasses.replace(tensor, data_offset=tensor.data_offset + 2)
            if tensor.flags & TENSOR_FLAG_CONSTANT and tensor.dtype == XTS3DType.INT32
            else tensor
        )
        for tensor in graph.tensors
    ]
    with pytest.raises(ValueError, match="4-byte aligned"):
        check_runtime_limits(dataclasses.replace(graph, tensors=tensors))
