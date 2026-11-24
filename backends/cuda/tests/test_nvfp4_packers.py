# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
import pytest
import torch
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.tests.nvfp4_gemma_test_utils import GemmaPacked
from executorch.backends.cuda.tests.nvfp4_test_utils import portable_weight
from executorch.examples.models.gemma4_31b.cuda_packers import (
    pack_embedding_for_cuda,
    pack_linear_for_cuda,
)
from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor


@pytest.mark.parametrize("kind", ["linear", "embedding"])
@pytest.mark.parametrize("parameter", [False, True])
def test_prepacked_nvfp4_replaces_meta_weight_without_repacking(kind, parameter):
    n, k = 7, 32
    q = torch.arange(n * (k // 2), dtype=torch.uint8).reshape(n, k // 2)
    s = torch.full((n, k // 16), 56, dtype=torch.uint8)
    packed = CudaNVFP4Tensor(q, s, torch.tensor([0.25]))
    source = torch.nn.Parameter(packed, requires_grad=False) if parameter else packed
    if kind == "linear":
        module = torch.nn.Linear(k, n, bias=False, device="meta", dtype=torch.bfloat16)
        pack = pack_linear_for_cuda
    else:
        module = torch.nn.Embedding(n, k, device="meta", dtype=torch.bfloat16)
        pack = pack_embedding_for_cuda
    pack(module, {"weight": source})
    assert isinstance(module.weight, CudaNVFP4Tensor)
    assert module.weight.device.type == "cpu"
    assert module.weight.shape == (n, k)
    assert not module.weight.requires_grad
    assert module.weight.qdata.data_ptr() == packed.qdata.data_ptr()
    assert module.weight.scale.data_ptr() == packed.scale.data_ptr()
    assert (
        module.weight.per_tensor_scale.data_ptr() == packed.per_tensor_scale.data_ptr()
    )
    torch.testing.assert_close(
        module.weight.dequantize(), packed.dequantize(), rtol=0, atol=0
    )


@pytest.mark.parametrize("pack", [pack_linear_for_cuda, pack_embedding_for_cuda])
def test_portable_nvfp4_is_packed(pack):
    portable = portable_weight(pts=0.5)
    qdata = torch.randint(0, 256, (70, 96), dtype=torch.uint8).view(torch.uint32)
    w = ExportableNVFP4Tensor(
        qdata, portable.scale, portable.per_tensor_scale, 16, portable.orig_dtype
    )
    if pack is pack_linear_for_cuda:
        module = torch.nn.Linear(192, 70, bias=False, device="meta")
    else:
        module = torch.nn.Embedding(70, 192, device="meta")
    pack(module, {"weight": w})
    assert isinstance(module.weight, CudaNVFP4Tensor)
    assert not module.weight.requires_grad
    assert torch.equal(module.weight.dequantize(), w.dequantize())


@pytest.mark.parametrize("native", [False, True])
def test_packed_modules_export_nvfp4_ops(native, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    monkeypatch.delenv("ET_CUDA_NVFP4_ACT", raising=False)
    model = GemmaPacked()
    for packed in (model.embedding.weight, model.linear.weight):
        assert packed.unit_scale
        assert packed.act_per_tensor_scale.shape == (1, 1)
        assert packed.act_per_tensor_scale.item() == pytest.approx(0.02)
    ids = torch.tensor([[1, 5], [69, 0]])
    x = torch.randn(37, 192, dtype=torch.bfloat16)
    ep = torch.export.export(model, (ids, x), strict=True)
    nodes = {
        str(node.target): node for node in ep.graph.nodes if node.op == "call_function"
    }
    assert "triton.nvfp4_embedding.default" in nodes
    if native:
        # The static activation scale reaches the op as a lifted constant.
        prefill = nodes["triton.nvfp4_auto_prefill.default"]
        assert isinstance(prefill.args[4], torch.fx.Node)
    else:
        assert len([t for t in nodes if t.startswith("triton.nvfp4_")]) == 2


@pytest.mark.parametrize("native", [False, True])
def test_packed_linear_decode_receives_unit_scale(native, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    # Native FP4 decode times the GEMV against W4A4 in nvfp4_auto_decode;
    # forced W4A16 keeps the plain GEMV.
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "auto" if native else "w4a16")
    x = torch.randn(1, 192, dtype=torch.bfloat16)
    ep = torch.export.export(GemmaPacked(), (torch.tensor([[1]]), x), strict=True)
    nodes = {
        str(node.target): node for node in ep.graph.nodes if node.op == "call_function"
    }
    node = nodes[f"triton.nvfp4_{'auto_decode' if native else 'gemv'}.default"]
    names = [argument.name for argument in node.target._schema.arguments]
    assert {**dict(zip(names, node.args)), **node.kwargs}["unit_scale"] is True


@pytest.mark.parametrize("pack", [pack_linear_for_cuda, pack_embedding_for_cuda])
def test_gguf_nvfp4_without_its_scale_is_rejected(pack):
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor

    raw = torch.zeros(7, 36, dtype=torch.uint8)
    weight = ExportableGGUFTensor(raw, "nvfp4", torch.bfloat16)
    if pack is pack_linear_for_cuda:
        module = torch.nn.Linear(64, 7, bias=False, device="meta")
    else:
        module = torch.nn.Embedding(7, 64, device="meta")
    with pytest.raises(ValueError, match=r"\.scale"):
        pack(module, {"weight": weight})
