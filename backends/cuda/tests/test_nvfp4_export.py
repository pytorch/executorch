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
from executorch.backends.cuda.autotune.launch_params import clear_launch_param_cache
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import (
    _has_native_fp4,
    _MODE_STRIDE,
)
from executorch.backends.cuda.tests.nvfp4_gemma_test_utils import GemmaPacked
from executorch.backends.cuda.tests.nvfp4_test_utils import PackedLinear
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_quantize_activation,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() < (8, 9),
    reason="NVIDIA CUDA with native E4M3 (SM89+) required",
)


@pytest.fixture(autouse=True)
def cold_pick_cache():
    clear_launch_param_cache()


@pytest.mark.parametrize(
    "mode,m,dynamic,fp32",
    [
        ("w4a16", 1, False, False),
        # Hints no other case tunes, so their tiles are timed here.
        ("w4a16", 37, True, False),
        ("w4a16", 45, True, True),
        ("w4a4", 16, False, False),
        ("auto", 1, False, False),
        ("auto", 16, True, False),
    ],
)
def test_packed_linear_pte(mode, m, dynamic, fp32, monkeypatch):
    if mode != "w4a16" and not _has_native_fp4():
        pytest.skip("native FP4 required")
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    # The FP32 prefill policy is read at export and baked into the program.
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "fp32" if fp32 else "input")
    import executorch.backends.cuda.cuda_backend as cuda_backend
    from executorch.backends.cuda.triton.kernels import quantized_gemm_utils

    original_context = cuda_backend.autotune_launch_params
    observed_picks = []
    config_graph_timings = []
    original_config_timing = quantized_gemm_utils.time_with_cuda_graph_us_counted

    def capture_config_timing(kernel_call, **kwargs):
        assert kwargs == {"statistic": "mean", "replays": 3}
        result = original_config_timing(kernel_call, **kwargs)
        if quantized_gemm_utils.graph_timing_active():
            config_graph_timings.append(result)
        return result

    monkeypatch.setattr(
        quantized_gemm_utils,
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
        if fp32:
            weight = module.weight.dequantize(torch.float32)
            return F.linear(value.float(), weight, module.bias.float())
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
        from executorch.export.utils import is_fbcode

        if is_fbcode():
            # The in-process runtime in fbcode has no CUDA backend.
            return
        from executorch.runtime import Runtime

        runtime = Runtime.get().load_program(str(path), data_path=str(data))
        method = runtime.load_method("forward")
        actual = method.execute([x])[0].clone()
        if dynamic:
            # Exercise grouped M tiling and the split-K workspace at sizes
            # different from the shape used by compilation/autotuning.
            for rows in [1, 5, 37, 64]:
                value = torch.randn(rows, 192, dtype=x.dtype)
                result = method.execute([value])[0].cpu()
                torch.testing.assert_close(
                    result, reference(value), rtol=0.03, atol=0.03
                )
    torch.testing.assert_close(actual.cpu(), reference(x), rtol=0.03, atol=0.03)


@pytest.mark.parametrize("mode", ["w4a16", "w4a4"])
def test_gemma_packed_modules_pte(mode, monkeypatch):
    if mode == "w4a4" and not _has_native_fp4():
        pytest.skip("native FP4 required")
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "input")
    module = GemmaPacked().eval()
    ids = torch.tensor([[1, 5], [69, 0]])
    x = torch.randn(37, 192, dtype=torch.bfloat16)
    lowered = to_edge_transform_and_lower(
        torch.export.export(module, (ids, x), strict=True),
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    weight = module.linear.weight
    if mode == "w4a16":
        expected = F.linear(x, weight.dequantize())
    else:
        q, s, g = nvfp4_quantize_activation(
            x.cuda(), weight.act_per_tensor_scale.cuda()
        )
        a = CudaNVFP4Tensor(q.cpu(), s.cpu(), torch.ones(1), orig_dtype=torch.float32)
        expected = (
            (a.dequantize() * g.cpu()[:, None]) @ weight.dequantize(torch.float32).T
        ).bfloat16()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gemma_packed.pte"
        with path.open("wb") as f:
            lowered.write_to_file(f)
        lowered.write_tensor_data_to_file(tmp)
        data = next(Path(tmp).glob("*.ptd"))
        from executorch.export.utils import is_fbcode

        if is_fbcode():
            return
        from executorch.runtime import Runtime

        method = (
            Runtime.get()
            .load_program(str(path), data_path=str(data))
            .load_method("forward")
        )
        rows, out = (t.cpu() for t in method.execute([ids, x]))
    assert torch.equal(rows, module.embedding.weight.dequantize()[ids])
    torch.testing.assert_close(out, expected, rtol=0.03, atol=0.03)
