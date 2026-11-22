# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Unit-scale metadata and native GEMV equivalence through serialized export."""

import contextlib
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.cuda.autotune.launch_params import clear_launch_param_cache
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import (
    _has_native_fp4,
    _MODE_STRIDE,
)
from executorch.backends.cuda.tests.nvfp4_test_utils import portable_weight as weight


@pytest.fixture(autouse=True)
def cold_pick_cache():
    clear_launch_param_cache()


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("half2", [False, True])
@pytest.mark.parametrize("row_pts", [False, True])
@pytest.mark.parametrize("split", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("native", [False, True])
def test_unit_scale_gemv_bitwise_same_tile(native, dtype, half2, row_pts, split):
    if native and not _has_native_fp4():
        pytest.skip("native FP4 required")
    from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import _gemv

    torch.manual_seed(17)
    m, n, k = 4, 70, 4224
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.0625
    q = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    s = torch.randint(0, 127, (n, k // 16), device="cuda", dtype=torch.uint8)
    pts = torch.ones(n if row_pts else 1, device="cuda")
    old = torch.empty(
        (split, m, n), device="cuda", dtype=torch.float32 if split > 1 else dtype
    )
    new = torch.empty_like(old)
    for output, unit in [(old, False), (new, True)]:
        _gemv.fn[((n + 7) // 8, m, split)](
            x,
            q,
            s,
            pts,
            output,
            m,
            n,
            k,
            split,
            BN=8,
            BK=256,
            NATIVE=native,
            HALF2=half2,
            ROW_PTS=row_pts,
            UNIT_SCALE=unit,
            num_warps=4,
            num_stages=1,
        )
    torch.testing.assert_close(
        new.view(torch.int16 if split == 1 else torch.int32),
        old.view(torch.int16 if split == 1 else torch.int32),
        rtol=0,
        atol=0,
    )


class UnitLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        portable = weight()
        portable.qdata = torch.randint(0, 256, (70, 96), dtype=torch.uint8).view(
            torch.uint32
        )
        self.register_buffer("weight", CudaNVFP4Tensor.from_exportable(portable))

    def forward(self, x):
        return F.linear(x, self.weight)


@CUDA
@pytest.mark.parametrize("mode", ["auto", "w4a16"])
def test_unit_scale_linear_pte(mode, monkeypatch):
    if mode == "auto" and not _has_native_fp4():
        pytest.skip("native FP4 required")
    import executorch.backends.cuda.cuda_backend as cuda_backend
    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm as kernels
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.runtime import Runtime

    observed = []
    picks = []
    original = kernels.launch_nvfp4_kernel
    original_context = cuda_backend.autotune_launch_params

    @contextlib.contextmanager
    def capture_tuning(*args, **kwargs):
        with original_context(*args, **kwargs) as stats:
            yield stats
        picks.extend(stats.picks)

    def capture(kernel, *args, **kwargs):
        if kernel is kernels._gemv:
            observed.append(kwargs["UNIT_SCALE"])
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(kernels, "launch_nvfp4_kernel", capture)
    monkeypatch.setattr(cuda_backend, "autotune_launch_params", capture_tuning)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    model = UnitLinear().eval()
    x = torch.randn(1, 192, dtype=torch.bfloat16)
    exported = torch.export.export(model, (x,), strict=True)
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    assert observed and all(observed)
    with tempfile.TemporaryDirectory() as folder:
        pte = Path(folder) / "unit.pte"
        with pte.open("wb") as output:
            lowered.write_to_file(output)
        lowered.write_tensor_data_to_file(folder)
        data = next(Path(folder).glob("*.ptd"))
        from executorch.export.utils import is_fbcode

        if is_fbcode():
            # The in-process runtime in fbcode has no CUDA backend.
            return
        method = (
            Runtime.get()
            .load_program(str(pte), data_path=str(data))
            .load_method("forward")
        )
        actual = method.execute([x])[0].cpu()
    selected = "w4a16"
    if mode == "auto":
        mode_picks = [pick for pick in picks if pick.param == "mode_split"]
        assert len(mode_picks) == 1
        selected = "w4a4" if mode_picks[0].choice // _MODE_STRIDE else "w4a16"
    if selected == "w4a4":
        q, s, g = kernels.nvfp4_quantize_activation(x.cuda())
        quantized = (
            CudaNVFP4Tensor(
                q.cpu(), s.cpu(), torch.ones(1), orig_dtype=torch.float64
            ).dequantize()
            * g.cpu().double()[:, None]
        )
        expected = (quantized @ model.weight.dequantize(torch.float64).T).to(x.dtype)
    else:
        expected = F.linear(x, model.weight.dequantize())
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
