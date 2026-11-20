# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Run every split-K path through a serialized CUDA delegate."""

import tempfile
from pathlib import Path

import pytest
import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_gemv,
    nvfp4_quantize_activation,
    nvfp4_w4a16,
    nvfp4_w4a4,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="CUDA with native E4M3 (SM89+) required",
)


class SplitLinear(torch.nn.Module):
    def __init__(self, mode, row_scales, split):
        super().__init__()
        torch.manual_seed(42)
        self.mode = mode
        self.split = split
        self.register_buffer("q", torch.randint(0, 256, (96, 2112), dtype=torch.uint8))
        self.register_buffer(
            "s", torch.full((96, 264), 0.0625).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        self.register_buffer(
            "pts", torch.linspace(0.1, 0.3, 96) if row_scales else torch.tensor([0.25])
        )

    def forward(self, x):
        if self.mode == "gemv":
            return nvfp4_gemv(x, self.q, self.s, self.pts, split_k=self.split)
        if self.mode == "w4a16":
            return nvfp4_w4a16(x, self.q, self.s, self.pts, split_k=self.split)
        return nvfp4_w4a4(x, self.q, self.s, self.pts, split_k=self.split)


@pytest.mark.parametrize(
    "mode,row_scales,split",
    [
        (mode, rows, split)
        for mode in ("gemv", "w4a16", "w4a4")
        for rows in (False, True)
        for split in (1, 2, 4, 8, 16, 32)
    ],
)
def test_split_k_pte(mode, row_scales, split):
    if mode == "w4a4" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 required")
    module = SplitLinear(mode, row_scales, split).eval()
    x = torch.randn(1 if mode == "gemv" else 17, 4224, dtype=torch.bfloat16)
    dynamic = (
        None if mode == "gemv" else ({0: torch.export.Dim("tokens", min=1, max=64)},)
    )
    exported = torch.export.export(module, (x,), dynamic_shapes=dynamic, strict=True)
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    weight = CudaNVFP4Tensor(
        module.q, module.s, module.pts, orig_dtype=torch.float64
    ).dequantize()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "split.pte"
        with path.open("wb") as output:
            lowered.write_to_file(output)
        lowered.write_tensor_data_to_file(folder)
        data = next(Path(folder).glob("*.ptd"))
        method = (
            Runtime.get()
            .load_program(str(path), data_path=str(data))
            .load_method("forward")
        )
        for rows in (1,) if mode == "gemv" else (1, 5, 17, 37, 64):
            value = torch.randn(rows, 4224, dtype=x.dtype)
            activation = value.double()
            if mode == "w4a4":
                q, s, scale = nvfp4_quantize_activation(value.cuda())
                activation = (
                    CudaNVFP4Tensor(
                        q.cpu(), s.cpu(), torch.ones(1), orig_dtype=torch.float64
                    ).dequantize()
                    * scale.cpu().double()[:, None]
                )
            expected = (activation @ weight.T).bfloat16()
            actual = method.execute([value])[0].cpu()
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)
