# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
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


def portable_weight(pts=1.0, scale_byte=40):
    return ExportableNVFP4Tensor(
        torch.zeros((70, 24), dtype=torch.uint32),
        torch.full((70, 12), scale_byte, dtype=torch.uint8),
        torch.tensor(pts),
        16,
        torch.bfloat16,
    )
