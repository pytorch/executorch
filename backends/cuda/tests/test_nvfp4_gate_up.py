# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Serialize the fused NVFP4 epilogue, including every split and row scales."""

import tempfile
from pathlib import Path

import pytest
import torch

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import _has_native_fp4
from executorch.backends.cuda.triton.kernels.nvfp4_gated_gemm import nvfp4_w4a4_gate_up
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import nvfp4_w4a4
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.export.utils import is_fbcode
from executorch.runtime import Runtime

pytestmark = pytest.mark.skipif(
    not _has_native_fp4(),
    reason="native FP4 required",
)


class GateUp(torch.nn.Module):
    def __init__(self, rows, split):
        super().__init__()
        torch.manual_seed(42)
        self.split = split
        self.register_buffer("q", torch.randint(0, 256, (70, 2112), dtype=torch.uint8))
        self.register_buffer(
            "s", torch.full((70, 264), 0.0625).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        self.register_buffer(
            "pts", torch.linspace(0.1, 0.3, 70) if rows else torch.tensor([0.25])
        )

    def forward(self, x):
        return nvfp4_w4a4_gate_up(x, self.q, self.s, self.pts, split_k=self.split)


@pytest.mark.parametrize("row_scales", [False, True])
@pytest.mark.parametrize("split", [1, 2, 4, 8, 16, 32])
def test_gate_up_split_k_pte(row_scales, split):
    module = GateUp(row_scales, split).eval()
    x = torch.randn(17, 4224, dtype=torch.bfloat16)
    exported = torch.export.export(
        module,
        (x,),
        dynamic_shapes=({0: torch.export.Dim("tokens", min=1, max=64)},),
        strict=True,
    )
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "gated.pte"
        with path.open("wb") as output:
            lowered.write_to_file(output)
        lowered.write_tensor_data_to_file(folder)
        data = next(Path(folder).glob("*.ptd"))
        if is_fbcode():
            # The in-process runtime in fbcode has no CUDA backend.
            return
        method = (
            Runtime.get()
            .load_program(str(path), data_path=str(data))
            .load_method("forward")
        )
        for rows in (1, 5, 17, 37, 64):
            value = torch.randn(rows, 4224, dtype=x.dtype)
            linear = nvfp4_w4a4(
                value.cuda(),
                module.q.cuda(),
                module.s.cuda(),
                module.pts.cuda(),
                split_k=split,
            )
            gate, up = linear.chunk(2, dim=1)
            expected = torch.nn.functional.silu(gate) * up
            actual = method.execute([value])[0].cpu()
            # Both GEMMs round to BF16 before SiLU; K accumulation may differ
            # between independently selected tiles.
            torch.testing.assert_close(actual, expected.cpu(), rtol=0.03, atol=0.04)
