# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export and validate chunk_gated_delta_rule triton kernel on CUDA backend.

Requires: pip install flash-linear-attention

Usage:
  python -m pytest backends/cuda/tests/test_chunk_gated_delta_rule.py -v

  # Standalone export (produces .pte + .ptd):
  python backends/cuda/tests/test_chunk_gated_delta_rule.py --output-dir /tmp/exports
"""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest

import executorch.backends.cuda.triton.kernels.chunk_gated_delta_rule  # noqa: F401

import fla  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export


B, T, H, K, V = 1, 128, 4, 64, 64

EXECUTORCH_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
RUNNER_PATH = os.path.join(EXECUTORCH_ROOT, "cmake-out", "executor_runner")

# Test configurations adapted from FLA's test_gated_delta.py test_chunk()
# Format: (seed, gate_logit_normalizer, mask_p, nonzero_h0, description)
FLA_TEST_CONFIGS = [
    # Basic configs varying gate normalizer
    (42, 1.0, 0.0, False, "basic_norm1"),
    (123, 0.1, 0.0, False, "strong_gate"),
    (7, 10.0, 0.0, False, "weak_gate"),
    # Non-zero initial state
    (42, 1.0, 0.0, True, "nonzero_h0_norm1"),
    (99, 0.1, 0.0, True, "nonzero_h0_strong"),
    (55, 10.0, 0.0, True, "nonzero_h0_weak"),
    # Sparse gating (50% of gates masked to zero)
    (42, 1.0, 0.5, False, "sparse_gate_50pct"),
    (77, 0.1, 0.5, True, "sparse_strong_h0"),
    # Different random patterns
    (0, 1.0, 0.0, False, "seed0"),
    (100, 1.0, 0.0, True, "seed100_h0"),
    (2024, 0.5, 0.0, False, "norm0.5"),
    (999, 5.0, 0.3, True, "norm5_sparse30_h0"),
    # Edge-ish values
    (13, 0.01, 0.0, False, "very_strong_gate"),
    (31, 100.0, 0.0, False, "very_weak_gate"),
    (64, 1.0, 0.9, True, "sparse_90pct_h0"),
]


class ChunkGatedDeltaModel(torch.nn.Module):
    def forward(self, q, k, v, g, beta, initial_state):
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        o, final_state = torch.ops.triton.chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state
        )
        return o, final_state


def _make_inputs_from_fla(
    seed,
    gate_logit_normalizer,
    mask_p=0.0,
    nonzero_h0=False,
    dtype=torch.bfloat16,
    device="cuda",
):
    """Generate inputs following FLA test_chunk() conventions."""
    torch.manual_seed(seed)
    q = torch.rand(B, T, H, K, dtype=dtype, device=device)
    k = torch.rand(B, T, H, K, dtype=dtype, device=device)
    v = torch.rand(B, T, H, V, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid().to(dtype)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device))
    g = (g / gate_logit_normalizer).to(dtype)
    if mask_p > 0:
        g = g * (torch.rand(B, T, H, dtype=dtype, device=device) > mask_p)
    if nonzero_h0:
        h0 = torch.randn(B, H, K, V, dtype=dtype, device=device)
    else:
        h0 = torch.zeros(B, H, K, V, dtype=dtype, device=device)
    return q, k, v, g, beta, h0


def _make_inputs(dtype=torch.bfloat16, device="cuda"):
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, dtype=dtype, device=device))
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    initial_state = torch.randn(B, H, K, V, dtype=dtype, device=device)
    return q, k, v, g, beta, initial_state


def _save_tensor(t, path):
    t_cpu = t.cpu().contiguous()
    with open(path, "wb") as f:
        f.write(bytes(t_cpu.untyped_storage()))


def _load_output(path, shape, dtype):
    data = np.fromfile(path, dtype=np.uint8)
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def export_chunk_gated_delta(output_dir):
    model = ChunkGatedDeltaModel().eval()
    inputs = _make_inputs()

    with torch.no_grad():
        ref_o, ref_s = model(*inputs)
    print(f"Eager output shape: {ref_o.shape}, final_state shape: {ref_s.shape}")

    with torch.no_grad():
        ep = export(model, inputs, strict=True)
    print("Export OK")

    os.makedirs(output_dir, exist_ok=True)

    specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    et_prog = to_edge_transform_and_lower(
        ep,
        partitioner=[CudaPartitioner(specs)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    pte_path = os.path.join(output_dir, "chunk_gated_delta.pte")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)

    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)

    print(f"Saved to {pte_path} ({os.path.getsize(pte_path) / 1024:.0f} KB)")
    return pte_path


def _run_cpp_runner(runner_path, pte_path, ptd_path, input_files, output_base):
    """Run executor_runner and return subprocess result."""
    cmd = [
        runner_path,
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={','.join(input_files)}",
        f"--output_file={output_base}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


class TestChunkGatedDeltaRule(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_eager(self):
        model = ChunkGatedDeltaModel().eval()
        inputs = _make_inputs()
        with torch.no_grad():
            o, s = model(*inputs)
        self.assertEqual(o.shape, torch.Size([B, T, H, V]))
        self.assertEqual(s.shape, torch.Size([B, H, K, V]))
        self.assertEqual(o.dtype, torch.bfloat16)
        self.assertEqual(s.dtype, torch.float32)

    def test_eager_fla_configs(self):
        """Run FLA-style test configurations and verify against naive reference."""
        from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

        model = ChunkGatedDeltaModel().eval()
        for seed, norm, mask_p, nonzero_h0, desc in FLA_TEST_CONFIGS:
            with self.subTest(desc=desc):
                inputs = _make_inputs_from_fla(seed, norm, mask_p, nonzero_h0)
                q, k, v, g, beta, h0 = inputs

                with torch.no_grad():
                    o_ours, s_ours = model(q, k, v, g, beta, h0)

                    o_ref, s_ref = naive_recurrent_gated_delta_rule(
                        q=F.normalize(q, p=2, dim=-1),
                        k=F.normalize(k, p=2, dim=-1),
                        v=v,
                        beta=beta,
                        g=g,
                        initial_state=h0,
                        output_final_state=True,
                    )

                o_diff = (o_ours.float() - o_ref.float()).abs().max().item()
                s_diff = (s_ours.float() - s_ref.float()).abs().max().item()
                self.assertLess(o_diff, 0.01, f"{desc}: output diff {o_diff}")
                self.assertLess(s_diff, 0.01, f"{desc}: state diff {s_diff}")

    def test_eager_matches_fla(self):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_impl

        torch.manual_seed(42)
        inputs = _make_inputs()
        q, k, v, g, beta, h0 = inputs

        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        with torch.no_grad():
            o_ours, _ = torch.ops.triton.chunk_gated_delta_rule(
                q_norm, k_norm, v, g, beta, h0
            )
            o_ref, _ = fla_impl(
                q,
                k,
                v,
                g,
                beta,
                initial_state=h0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        self.assertLess((o_ours.float() - o_ref.float()).abs().max().item(), 0.01)

    def test_export_cuda(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path = export_chunk_gated_delta(tmpdir)
            self.assertTrue(os.path.exists(pte_path))
            self.assertGreater(os.path.getsize(pte_path), 0)

    def test_e2e_cpp_runner(self):
        self.assertTrue(
            os.path.exists(RUNNER_PATH),
            f"executor_runner not found at {RUNNER_PATH}. "
            "Build with: cmake --build cmake-out --target executor_runner",
        )
        """Export, run executor_runner with FLA test inputs, compare with eager."""
        model = ChunkGatedDeltaModel().eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = os.path.join(tmpdir, "export")
            pte_path = export_chunk_gated_delta(export_dir)
            ptd_path = os.path.join(export_dir, "aoti_cuda_blob.ptd")

            for seed, norm, mask_p, nonzero_h0, desc in FLA_TEST_CONFIGS:
                with self.subTest(desc=desc):
                    inputs = _make_inputs_from_fla(seed, norm, mask_p, nonzero_h0)
                    q, k, v, g, beta, h0 = inputs

                    with torch.no_grad():
                        ref_o, ref_s = model(q, k, v, g, beta, h0)

                    run_dir = os.path.join(tmpdir, f"run_{desc}")
                    os.makedirs(run_dir)

                    input_files = []
                    for i, tensor in enumerate(inputs):
                        path = os.path.join(run_dir, f"{i}.bin")
                        _save_tensor(tensor, path)
                        input_files.append(path)

                    output_base = os.path.join(run_dir, "output")
                    result = _run_cpp_runner(
                        RUNNER_PATH, pte_path, ptd_path, input_files, output_base
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"{desc}: executor_runner failed:\n{result.stderr}",
                    )

                    cpp_o = _load_output(
                        f"{output_base}-0.bin",
                        (B, T, H, V),
                        torch.bfloat16,
                    )
                    cpp_s = _load_output(
                        f"{output_base}-1.bin",
                        (B, H, K, V),
                        torch.float32,
                    )

                    o_diff = (cpp_o.float() - ref_o.cpu().float()).abs().max().item()
                    s_diff = (cpp_s.float() - ref_s.cpu().float()).abs().max().item()
                    self.assertLess(o_diff, 0.01, f"{desc}: output diff {o_diff}")
                    self.assertLess(s_diff, 0.1, f"{desc}: state diff {s_diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args, remaining = parser.parse_known_args()

    if args.output_dir:
        export_chunk_gated_delta(args.output_dir)
    else:
        sys.argv = [sys.argv[0]] + remaining
        unittest.main()
