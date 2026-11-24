# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.examples.models.gemma4_31b.cuda_packers import (
    pack_embedding_for_cuda,
    pack_linear_for_cuda,
)
from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor


class GemmaPacked(torch.nn.Module):
    """An embedding and a linear packed by the Gemma 4 CUDA packers from one
    unit-scale ``ExportableNVFP4Tensor`` with a static activation scale."""

    def __init__(self, n=70, k=192, act_scale=0.02):
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
