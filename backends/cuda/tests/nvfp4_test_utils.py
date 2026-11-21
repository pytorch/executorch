# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor


def shards(k, device="cpu", dtype=torch.bfloat16):
    generator = torch.Generator(device).manual_seed(2026)
    result = []
    for n, global_scale in zip((31, 19, 47), (0.25, 0.37, 0.083)):
        q = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, device=device, generator=generator
        )
        s = (
            (0.03 + torch.rand(n, k // 16, device=device, generator=generator) * 0.12)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        pts = torch.tensor([global_scale], device=device)
        result.append(CudaNVFP4Tensor(q, s, pts, orig_dtype=dtype))
    return result


def static_scale(value, device="cuda"):
    """A static activation scale as the kernels take it (None: dynamic)."""
    return None if value is None else torch.tensor([[value]], device=device)


def portable_weight(pts=1.0, scale_byte=40):
    return ExportableNVFP4Tensor(
        torch.zeros((70, 24), dtype=torch.uint32),
        torch.full((70, 12), scale_byte, dtype=torch.uint8),
        torch.tensor(pts),
        16,
        torch.bfloat16,
    )


def weight(n, k):
    torch.manual_seed(42)
    q = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    s = (
        (0.05 + torch.rand(n, k // 16, device="cuda") * 0.1)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return CudaNVFP4Tensor(q, s, torch.tensor(0.25, device="cuda"))


def dynamic_rows(mode, tensor, upper):
    """``tensor`` as a fake tensor whose dim 0 is dynamic, bounded by ``upper``
    (None: unbounded)."""
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        StatelessSymbolicContext,
        StrictMinMaxConstraint,
    )
    from torch.utils._sympy.value_ranges import ValueRanges

    constraint = (
        None
        if upper is None
        else StrictMinMaxConstraint(vr=ValueRanges(1, upper), warn_only=False)
    )
    context = StatelessSymbolicContext(
        dynamic_sizes=[DimDynamic.DYNAMIC] + [DimDynamic.STATIC] * (tensor.ndim - 1),
        constraint_sizes=[constraint] + [None] * (tensor.ndim - 1),
    )
    return mode.from_tensor(tensor, symbolic_context=context)


class PackedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        q = torch.randint(0, 256, (96, 96), dtype=torch.uint8)
        s = torch.full((96, 12), 0.0625).to(torch.float8_e4m3fn).view(torch.uint8)
        self.register_buffer("weight", CudaNVFP4Tensor(q, s, torch.tensor(0.25)))
        self.register_buffer("bias", torch.randn(96, dtype=torch.bfloat16))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class GemmaPacked(torch.nn.Module):
    """An embedding and a linear packed by the Gemma 4 CUDA packers from one
    unit-scale ``ExportableNVFP4Tensor`` with a static activation scale."""

    def __init__(self, n=70, k=192, act_scale=0.02):
        from executorch.examples.models.gemma4_31b.cuda_packers import (
            pack_embedding_for_cuda,
            pack_linear_for_cuda,
        )

        super().__init__()
        self.embedding = torch.nn.Embedding(n, k, device="meta")
        self.linear = torch.nn.Linear(k, n, bias=False, device="meta")
        generator = torch.Generator().manual_seed(0)
        qdata = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, generator=generator
        )
        w = ExportableNVFP4Tensor(
            qdata.view(torch.uint32),
            torch.full((n, k // 16), 40, dtype=torch.uint8),
            torch.tensor(1.0),
            16,
            torch.bfloat16,
            act_per_tensor_scale=act_scale,
        )
        pack_embedding_for_cuda(self.embedding, {"weight": w})
        pack_linear_for_cuda(self.linear, {"weight": w})

    def forward(self, ids, x):
        return self.embedding(ids), self.linear(x)
