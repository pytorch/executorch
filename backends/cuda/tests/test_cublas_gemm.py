# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Matrix multiplications on cuBLAS through the mm, addmm and bmm CUDA shims.

With max_autotune_gemm_backends=ATEN, Inductor calls those shims instead of
compiling Triton matrix multiplications into the model.

Usage:
  python -m pytest backends/cuda/tests/test_cublas_gemm.py -v
"""

import glob
import os
import tempfile
import unittest
from unittest import mock

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes import MemoryPlanningPass
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

_CUBLAS_SHIMS = {
    "aoti_torch_cuda_mm_out",
    "aoti_torch_cuda_addmm_out",
    "aoti_torch_cuda_bmm_out",
}
_ATEN = [CompileSpec("max_autotune_gemm_backends", b"ATEN")]


class GemmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.with_bias = torch.nn.Linear(64, 48)
        self.without_bias = torch.nn.Linear(48, 32, bias=False)
        # Inductor folds this add into addmm, which gets it as a column.
        self.row_offsets = torch.nn.Parameter(torch.randn(16, 1))
        self.head = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x, y):
        h = self.without_bias(self.with_bias(x)) + self.row_offsets
        return self.head(torch.bmm(h.view(4, 4, 32), y))


def _export_and_run(model, inputs, specs):
    """Lowers `model` with the given compile specs and runs it on the
    ExecuTorch runtime. Returns the outputs and the extern shim calls."""
    specs = [CudaBackend.generate_method_name_compile_spec("forward"), *specs]
    called = []
    original = CppWrapperCpu.generate_c_shim_extern_kernel_call

    def record(self, kernel, *args, **kwargs):
        called.append(kernel)
        return original(self, kernel, *args, **kwargs)

    with mock.patch.object(
        CppWrapperCpu, "generate_c_shim_extern_kernel_call", record
    ), torch.no_grad():
        program = to_edge_transform_and_lower(
            torch.export.export(model, inputs, strict=True),
            partitioner=[CudaPartitioner(specs)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        ).to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            )
        )

    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    with tempfile.TemporaryDirectory() as tmpdir:
        pte = os.path.join(tmpdir, "model.pte")
        with open(pte, "wb") as f:
            program.write_to_file(f)
        program.write_tensor_data_to_file(tmpdir)
        ptd = glob.glob(os.path.join(tmpdir, "*.ptd"))[0]
        module = _load_for_executorch(pte, data_path=ptd)
        outputs = module.run_method("forward", [t.cpu() for t in inputs])
    return outputs, called


class TestCublasGemm(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if torch.version.hip is not None:
            self.skipTest("cuBLAS shims are CUDA only")

    def _check(self, dtype, atol, rtol, specs=_ATEN):
        torch.manual_seed(0)
        model = GemmModel().to(device="cuda", dtype=dtype).eval()
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=dtype),
            torch.randn(4, 32, 8, device="cuda", dtype=dtype),
        )
        with torch.no_grad():
            expected = model(*inputs).cpu()

        outputs, called = _export_and_run(model, inputs, specs)

        # Two addmm calls: one for the bias of with_bias, one for row_offsets
        # folded in as a column.
        self.assertEqual(
            sorted(kernel for kernel in called if kernel in _CUBLAS_SHIMS),
            [
                "aoti_torch_cuda_addmm_out",
                "aoti_torch_cuda_addmm_out",
                "aoti_torch_cuda_bmm_out",
                "aoti_torch_cuda_mm_out",
            ],
            called,
        )
        torch.testing.assert_close(outputs[0], expected, atol=atol, rtol=rtol)

    def test_float32(self):
        self._check(torch.float32, atol=1e-4, rtol=1e-4)

    def test_bfloat16(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 is not supported on this GPU")
        self._check(torch.bfloat16, atol=5e-2, rtol=2e-2)

    def test_float16(self):
        self._check(torch.float16, atol=1e-2, rtol=1e-2)

    def test_without_max_autotune(self):
        # Without autotuning there are no Triton templates to choose from.
        self._check(
            torch.float32,
            atol=1e-4,
            rtol=1e-4,
            specs=[CompileSpec("max_autotune", b"OFF")],
        )


class TestGemmBackendsCompileSpec(unittest.TestCase):
    def test_default_is_triton(self):
        options = CudaBackend.get_aoti_compile_options([])
        self.assertEqual(options["max_autotune_gemm_backends"], "TRITON")

    def test_aten_and_triton(self):
        with mock.patch.object(torch.version, "hip", None):
            options = CudaBackend.get_aoti_compile_options(
                [CompileSpec("max_autotune_gemm_backends", b"aten,triton")]
            )
        self.assertEqual(options["max_autotune_gemm_backends"], "ATEN,TRITON")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid max_autotune_gemm_backends"):
            CudaBackend.get_aoti_compile_options(
                [CompileSpec("max_autotune_gemm_backends", b"ATEN,CUTLASS")]
            )

    def test_aten_is_rejected_on_rocm(self):
        with mock.patch.object(torch.version, "hip", "7.0"):
            with self.assertRaisesRegex(ValueError, "cuBLAS"):
                CudaBackend.get_aoti_compile_options(
                    [CompileSpec("max_autotune_gemm_backends", b"ATEN")]
                )

    def test_aten_is_rejected_for_windows(self):
        with mock.patch.object(torch.version, "hip", None):
            with self.assertRaisesRegex(ValueError, "cuBLAS"):
                CudaBackend.get_aoti_compile_options(
                    [
                        CompileSpec("platform", b"windows"),
                        CompileSpec("max_autotune_gemm_backends", b"ATEN"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
