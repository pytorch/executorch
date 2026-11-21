# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Keep a prefill compiled at M=4096 valid down to a single input token."""

import contextlib
import tempfile
from pathlib import Path

import executorch.backends.cuda.cuda_backend as cuda_backend

import pytest
import torch
from executorch.backends.cuda.autotune.launch_params import clear_launch_param_cache
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import (
    _has_native_fp4,
    _MODE_STRIDE,
)
from executorch.backends.cuda.tests.nvfp4_test_utils import PackedLinear
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_quantize_activation,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


@pytest.fixture(autouse=True)
def cold_pick_cache():
    clear_launch_param_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("mode", ["auto", "w4a4"])
def test_prefill_4096_keeps_compiled_mode_for_dynamic_m(mode, monkeypatch):
    if not _has_native_fp4():
        pytest.skip("native FP4 required")
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    original_context = cuda_backend.autotune_launch_params
    observed = []

    @contextlib.contextmanager
    def capture_tuning(*args, **kwargs):
        with original_context(*args, **kwargs) as stats:
            yield stats
        observed.extend(stats.picks)

    monkeypatch.setattr(cuda_backend, "autotune_launch_params", capture_tuning)
    module = PackedLinear().eval()
    x = torch.randn(4096, 192, dtype=torch.bfloat16)
    ep = torch.export.export(
        module,
        (x,),
        dynamic_shapes=({0: torch.export.Dim("tokens", min=1, max=4096)},),
        strict=True,
    )
    lowered = to_edge_transform_and_lower(
        ep,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    if mode == "auto":
        picks = [pick for pick in observed if pick.param == "mode_split"]
        assert len(picks) == 1
        selected_fp4 = bool(picks[0].choice // _MODE_STRIDE)
    else:
        selected_fp4 = True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixed-prefill.pte"
        with path.open("wb") as stream:
            lowered.write_to_file(stream)
        lowered.write_tensor_data_to_file(tmp)
        data = next(Path(tmp).glob("*.ptd"))
        from executorch.export.utils import is_fbcode

        if is_fbcode():
            # The in-process runtime in fbcode has no CUDA backend.
            return
        from executorch.runtime import Runtime

        method = (
            Runtime.get()
            .load_program(str(path), data_path=str(data))
            .load_method("forward")
        )
        weights = module.weight.dequantize().double()
        for m in (1, 5, 37, 64, 4096):
            value = torch.randn(m, 192, dtype=x.dtype)
            if selected_fp4:
                q, s, pts = nvfp4_quantize_activation(value.cuda())
                quantized = (
                    CudaNVFP4Tensor(
                        q.cpu(), s.cpu(), torch.ones(1), orig_dtype=torch.float64
                    ).dequantize()
                    * pts.cpu().double()[:, None]
                )
                reference = (quantized @ weights.T).bfloat16() + module.bias
            else:
                reference = torch.nn.functional.linear(
                    value, module.weight.dequantize(), module.bias
                )
            actual = method.execute([value])[0].cpu()
            torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.03)
