# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.vgf.diagnostics import collect_vgf_boundary_manifest
from torch.fx import Graph, GraphModule


def quantize_per_tensor_for_test(x, *args):
    return x


def dequantize_per_tensor_for_test(x, *args):
    return x


def _tensor(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def test_boundary_manifest_attributes_external_qdq_and_bytes():
    graph = Graph()
    x = graph.placeholder("x")
    q = graph.call_function(
        quantize_per_tensor_for_test,
        args=(x, 0.25, 0, -128, 127, torch.int8),
    )
    relu = graph.call_function(torch.ops.aten.relu.default, args=(q,))
    dq = graph.call_function(
        dequantize_per_tensor_for_test,
        args=(relu, 0.25, 0, -128, 127, torch.int8),
    )
    graph.output(dq)

    x.meta["val"] = _tensor((1, 8), torch.float32)
    q.meta["val"] = _tensor((1, 8), torch.int8)
    relu.meta["val"] = _tensor((1, 8), torch.int8)
    dq.meta["val"] = _tensor((1, 8), torch.float32)
    relu.meta["delegation_tag"] = "tag0"

    gm = GraphModule({}, graph)
    manifest = collect_vgf_boundary_manifest(gm, {"tag0": object()})

    assert manifest["summary"]["partition_count"] == 1
    by_kind = manifest["summary"]["conversions_by_kind"]
    assert by_kind["QUANTIZE"]["count"] == 1
    assert by_kind["QUANTIZE"]["known_input_bytes"] == 32
    assert by_kind["QUANTIZE"]["known_output_bytes"] == 8
    assert by_kind["DEQUANTIZE"]["count"] == 1
    assert by_kind["DEQUANTIZE"]["known_input_bytes"] == 8
    assert by_kind["DEQUANTIZE"]["known_output_bytes"] == 32


def test_boundary_manifest_reports_materializing_copy_chain():
    graph = Graph()
    x = graph.placeholder("x")
    clone = graph.call_function(torch.ops.aten.clone.default, args=(x,))
    relu = graph.call_function(torch.ops.aten.relu.default, args=(clone,))
    graph.output(relu)

    x.meta["val"] = _tensor((2, 4), torch.float32)
    clone.meta["val"] = _tensor((2, 4), torch.float32)
    relu.meta["val"] = _tensor((2, 4), torch.float32)
    relu.meta["delegation_tag"] = "tag0"

    gm = GraphModule({}, graph)
    manifest = collect_vgf_boundary_manifest(gm, {"tag0": object()})

    stats = manifest["summary"]["conversions_by_kind"]["MATERIALIZING_COPY"]
    assert stats["count"] == 1
    assert stats["known_input_bytes"] == 32
    assert stats["known_output_bytes"] == 32


def test_boundary_manifest_keeps_unclassified_external_ops_visible():
    graph = Graph()
    x = graph.placeholder("x")
    sigmoid = graph.call_function(torch.ops.aten.sigmoid.default, args=(x,))
    relu = graph.call_function(torch.ops.aten.relu.default, args=(sigmoid,))
    graph.output(relu)

    x.meta["val"] = _tensor((2, 4), torch.float32)
    sigmoid.meta["val"] = _tensor((2, 4), torch.float32)
    relu.meta["val"] = _tensor((2, 4), torch.float32)
    relu.meta["delegation_tag"] = "tag0"

    gm = GraphModule({}, graph)
    manifest = collect_vgf_boundary_manifest(gm, {"tag0": object()})

    summary = manifest["summary"]
    assert summary["unclassified_external_boundary_op_count"] == 1
    assert "sigmoid" in summary["unclassified_external_boundary_ops"][0]["external_op"]
