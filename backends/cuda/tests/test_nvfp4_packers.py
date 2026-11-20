# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
import pytest
import torch
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.examples.models.gemma4_31b.cuda_packers import (
    pack_embedding_for_cuda,
    pack_linear_for_cuda,
)


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
    assert module.weight.per_tensor_scale.data_ptr() == packed.per_tensor_scale.data_ptr()
    torch.testing.assert_close(module.weight.dequantize(), packed.dequantize(), rtol=0, atol=0)
