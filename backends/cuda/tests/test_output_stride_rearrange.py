# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test that CUDA backend correctly handles output stride mismatches.

The AOTI delegate always produces contiguous output, but the .pte may
serialize a different dim_order (e.g., from SDPA's efficient attention
which returns a transposed view). The runtime must rearrange the output
data to match the ETensor's expected layout.

Fast path: strides match (raw byte copy).
Slow path: strides differ (element-by-element rearrange).

Usage:
    python -m pytest backends/cuda/tests/test_output_stride_rearrange.py -v
"""

import glob
import os
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


def _run_et_aoti(module, inputs, triton_mode="OFF"):
    """Export and run a model through the ExecuTorch CUDA AOTI backend."""
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    ep = torch.export.export(module, inputs, strict=True)
    compile_specs = [
        CudaBackend.generate_method_name_compile_spec("forward"),
        CompileSpec(key="triton_kernel_mode", value=triton_mode.encode()),
    ]
    partitioner = {"forward": [CudaPartitioner(compile_specs)]}
    et_prog = to_edge_transform_and_lower(
        {"forward": ep},
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        constant_methods={"test": 1},
    )
    et = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        pte = os.path.join(tmpdir, "test.pte")
        with open(pte, "wb") as f:
            et.write_to_file(f)
        ptd = None
        if et._tensor_data:
            et.write_tensor_data_to_file(tmpdir)
            ptd_files = glob.glob(os.path.join(tmpdir, "*.ptd"))
            ptd = ptd_files[0] if ptd_files else None
        mod = _load_for_executorch(pte, data_path=ptd)
        return mod.run_method("forward", [t.cpu() for t in inputs])


class TestOutputStrideRearrange(unittest.TestCase):
    """Test CUDA backend output copy with matching and mismatching strides."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("BF16 not supported on this GPU")

    def _check_sdpa(self, module, inputs, label):
        """Run model through ExecuTorch AOTI with both Triton ON and OFF."""
        with torch.no_grad():
            eager = module(*inputs).float().cpu()

        for mode in ["OFF", "ON"]:
            with self.subTest(triton_mode=mode):
                result = _run_et_aoti(module, inputs, triton_mode=mode)[0].float()
                rel = (result - eager).abs() / eager.abs().clamp(min=1e-6)
                self.assertLess(
                    rel.mean().item(),
                    0.2,
                    f"{label} triton={mode} mean_rel={rel.mean():.4f} too large",
                )

    def test_fast_path_no_mask(self):
        """No mask SDPA — output strides match, uses fast byte copy path."""

        class SDPANoMask(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        D = 64
        q = torch.randn(1, 4, 4, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 200, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 4, 200, D, dtype=torch.bfloat16, device="cuda")

        self._check_sdpa(SDPANoMask().eval(), (q, k, v), "no-mask")

    def test_slow_path_masked_sdpa(self):
        """Masked SDPA — output strides differ, uses rearrange path."""

        class SDPAWithMask(nn.Module):
            def forward(self, q, k, v, mask):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=False
                )

        D = 64
        q = torch.randn(1, 4, 4, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 200, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 4, 200, D, dtype=torch.bfloat16, device="cuda")
        mask = torch.ones(1, 1, 4, 200, dtype=torch.bool, device="cuda")

        self._check_sdpa(SDPAWithMask().eval(), (q, k, v, mask), "masked")

    def test_slow_path_sparse_bool_mask(self):
        """Sparse bool mask — exercises stride rearrange with real masking."""

        class SDPASparseMask(nn.Module):
            def forward(self, q, k, v):
                KV = k.shape[2]
                mask = (
                    (torch.arange(KV, device=q.device) < KV // 2)
                    .view(1, 1, 1, KV)
                    .expand(1, 1, q.shape[2], -1)
                )
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=False
                )

        D = 64
        q = torch.randn(1, 4, 4, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 200, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 4, 200, D, dtype=torch.bfloat16, device="cuda")

        self._check_sdpa(SDPASparseMask().eval(), (q, k, v), "sparse-mask")

    def test_simple_add_no_rearrange(self):
        """Simple add — fast path, no stride mismatch."""

        class AddModule(nn.Module):
            def forward(self, a, b):
                return a + b

        a = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")

        module = AddModule().eval()
        with torch.no_grad():
            eager = module(a, b).float().cpu()

        result = _run_et_aoti(module, (a, b))[0].float()
        self.assertTrue(
            torch.allclose(result, eager, atol=1e-3),
            "simple add should match exactly",
        )


if __name__ == "__main__":
    unittest.main()
