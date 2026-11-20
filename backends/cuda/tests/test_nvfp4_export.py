# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Exercise packed tensors through export, CUDA AOTI and a serialized PTE."""

import contextlib
import tempfile
from pathlib import Path

import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
import pytest
import torch
import torch.nn.functional as F
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import _MODE_STRIDE
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_quantize_activation,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="CUDA with native E4M3 (SM89+) required",
)


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


@pytest.mark.parametrize(
    "mode,m,dynamic",
    [("w4a16", 1, False), ("w4a4", 16, False), ("auto", 1, False), ("auto", 16, True)],
)
def test_packed_linear_pte(mode, m, dynamic, monkeypatch):
    if mode != "w4a16" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 required")
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    import executorch.backends.cuda.cuda_backend as cuda_backend
    from executorch.backends.cuda.triton.kernels import nvfp4_tuning

    original_context = cuda_backend.autotune_launch_params
    observed_picks = []
    config_graph_timings = []
    original_config_timing = nvfp4_tuning.time_with_cuda_graph_us_counted

    def capture_config_timing(kernel_call, **kwargs):
        assert kwargs == {"statistic": "mean", "replays": 3}
        result = original_config_timing(kernel_call, **kwargs)
        if nvfp4_tuning.graph_timing_active():
            config_graph_timings.append(result)
        return result

    monkeypatch.setattr(
        nvfp4_tuning,
        "time_with_cuda_graph_us_counted",
        capture_config_timing,
    )

    @contextlib.contextmanager
    def capture_tuning(*args, **kwargs):
        with original_context(*args, **kwargs) as stats:
            yield stats
        observed_picks.extend(stats.picks)

    monkeypatch.setattr(cuda_backend, "autotune_launch_params", capture_tuning)
    module = PackedLinear().eval()
    x = torch.randn(m, 192, dtype=torch.bfloat16)
    shapes = ({0: torch.export.Dim("tokens", min=1, max=64)},) if dynamic else None
    ep = torch.export.export(module, (x,), dynamic_shapes=shapes, strict=True)
    lowered = to_edge_transform_and_lower(
        ep,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    assert config_graph_timings, (
        "NVFP4 tile candidates must use CUDA graph timing during AOTI, "
        "as activation mode and split-K candidates do"
    )
    if mode == "auto":
        picks = [pick for pick in observed_picks if pick.param == "mode_split"]
        assert (
            len(picks) == 1
        ), "auto mode must remain an op until AOTI measures activation mode and split-K"
        selected_mode = "w4a4" if picks[0].choice // _MODE_STRIDE else "w4a16"
    else:
        selected_mode = mode

    def reference(value):
        if selected_mode == "w4a16":
            return F.linear(value, module.weight.dequantize(), module.bias)
        q, s, pts = nvfp4_quantize_activation(value.cuda())
        quantized = (
            CudaNVFP4Tensor(
                q.cpu(), s.cpu(), torch.ones(1), orig_dtype=torch.float64
            ).dequantize()
            * pts.cpu().double()[:, None]
        )
        return (
            quantized @ module.weight.dequantize().double().T
        ).bfloat16() + module.bias

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nvfp4.pte"
        with path.open("wb") as f:
            lowered.write_to_file(f)
        lowered.write_tensor_data_to_file(tmp)
        data = next(Path(tmp).glob("*.ptd"))
        runtime = Runtime.get().load_program(str(path), data_path=str(data))
        method = runtime.load_method("forward")
        actual = method.execute([x])[0].clone()
        if dynamic:
            # Exercise grouped M tiling and the split-K workspace at sizes
            # different from the shape used by compilation/autotuning.
            monkeypatch.setenv(
                "ET_CUDA_NVFP4_ACT", "w4a16" if selected_mode == "w4a4" else "w4a4"
            )
            for rows in [1, 5, 37, 64]:
                value = torch.randn(rows, 192, dtype=x.dtype)
                result = method.execute([value])[0].cpu()
                torch.testing.assert_close(
                    result, reference(value), rtol=0.03, atol=0.03
                )
            monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    torch.testing.assert_close(actual.cpu(), reference(x), rtol=0.03, atol=0.03)
