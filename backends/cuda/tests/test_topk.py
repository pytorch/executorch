# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export and validate topk triton kernel on CUDA backend.

Usage:
  python -m pytest backends/cuda/tests/test_topk.py -v

  # Standalone export (produces .pte + .ptd):
  python backends/cuda/tests/test_topk.py --output-dir /tmp/exports
"""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export


RUNNER_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../cmake-out/backends/cuda/tests/topk_runner/topk_runner",
)

# Test configurations: (seed, rows, cols, k, dim, largest, description)
TEST_CONFIGS = [
    (42, 4, 8, 2, -1, True, "basic_4x8_k2"),
    (0, 1, 16, 3, -1, True, "single_row_k3"),
    (7, 8, 4, 1, -1, True, "8x4_k1"),
    (99, 4, 8, 2, -1, False, "smallest_k2"),
    (13, 2, 32, 5, -1, True, "wide_k5"),
    (55, 4, 8, 8, -1, True, "k_equals_n"),
    (77, 1, 4, 2, -1, True, "tiny_1x4_k2"),
    (123, 16, 8, 2, -1, True, "many_rows"),
]


class TopKModel(nn.Module):
    """Linear projection followed by topk."""

    def __init__(self, dim_in=8, k=2, topk_dim=-1, largest=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in, bias=False)
        self.k = k
        self.topk_dim = topk_dim
        self.largest = largest

    def forward(self, x):
        x = self.linear(x)
        values, indices = torch.topk(x, self.k, dim=self.topk_dim, largest=self.largest)
        return values, indices


def _make_inputs(seed, rows, cols, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(seed)
    return (torch.randn(rows, cols, dtype=dtype, device=device),)


def _save_tensor(t, path):
    t_cpu = t.cpu().contiguous()
    with open(path, "wb") as f:
        f.write(bytes(t_cpu.untyped_storage()))


def _load_output(path, shape, dtype):
    data = np.fromfile(path, dtype=np.uint8)
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def export_topk(output_dir, cols=8, k=2, largest=True):
    """Export a TopKModel to .pte + .ptd."""
    torch.manual_seed(42)
    model = (
        TopKModel(dim_in=cols, k=k, largest=largest)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    inputs = _make_inputs(42, 4, cols)

    with torch.no_grad():
        ref_vals, ref_idx = model(*inputs)
    print(f"Eager output: values {ref_vals.shape}, indices {ref_idx.shape}")

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

    pte_path = os.path.join(output_dir, "topk.pte")
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)

    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)

    print(f"Saved to {pte_path} ({os.path.getsize(pte_path) / 1024:.0f} KB)")
    return pte_path, model


def _run_cpp_runner(runner_path, pte_path, ptd_path, input_dir, output_dir):
    cmd = [
        runner_path,
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--input_dir={input_dir}",
        f"--output_dir={output_dir}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestTopK(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_eager(self):
        """Triton topk produces correct shapes and dtypes."""
        model = TopKModel().to(device="cuda", dtype=torch.bfloat16).eval()
        inputs = _make_inputs(42, 4, 8)
        with torch.no_grad():
            vals, idx = model(*inputs)
        self.assertEqual(vals.shape, torch.Size([4, 2]))
        self.assertEqual(idx.shape, torch.Size([4, 2]))
        self.assertEqual(vals.dtype, torch.bfloat16)
        self.assertEqual(idx.dtype, torch.int64)

    def test_eager_correctness(self):
        """Triton topk matches torch.topk across multiple configs."""
        for seed, rows, cols, k, dim, largest, desc in TEST_CONFIGS:
            with self.subTest(desc=desc):
                torch.manual_seed(seed)
                x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

                ref_vals, ref_idx = torch.topk(x, k, dim=dim, largest=largest)

                from executorch.backends.cuda.triton.kernels.topk import (
                    topk as triton_topk,
                )

                tri_vals, tri_idx = triton_topk(x, k, dim=dim, largest=largest)

                v_diff = (tri_vals.float() - ref_vals.float()).abs().max().item()
                self.assertLess(v_diff, 1e-3, f"{desc}: value diff {v_diff}")
                self.assertTrue(
                    torch.equal(tri_idx, ref_idx),
                    f"{desc}: indices mismatch",
                )

    def test_export_cuda(self):
        """Export succeeds and produces non-empty .pte."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path, _ = export_topk(tmpdir)
            self.assertTrue(os.path.exists(pte_path))
            self.assertGreater(os.path.getsize(pte_path), 0)

    @unittest.skipUnless(os.path.exists(RUNNER_PATH), "C++ runner not built")
    def test_e2e_cpp_runner(self):
        """Export, run C++ runner, compare with eager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = os.path.join(tmpdir, "export")
            pte_path, model = export_topk(export_dir)
            ptd_path = os.path.join(export_dir, "aoti_cuda_blob.ptd")

            for seed, rows, cols, k, _dim, largest, desc in TEST_CONFIGS:
                # Skip configs that don't match the exported model shape
                if cols != 8 or k != 2 or not largest or rows != 4:
                    continue

                with self.subTest(desc=desc):
                    inputs = _make_inputs(seed, rows, cols)

                    with torch.no_grad():
                        ref_vals, ref_idx = model(*inputs)

                    input_dir = os.path.join(tmpdir, f"inputs_{desc}")
                    output_dir = os.path.join(tmpdir, f"outputs_{desc}")
                    os.makedirs(input_dir)
                    os.makedirs(output_dir)

                    _save_tensor(inputs[0], os.path.join(input_dir, "x.bin"))

                    result = _run_cpp_runner(
                        RUNNER_PATH, pte_path, ptd_path, input_dir, output_dir
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"{desc}: C++ runner failed:\n{result.stderr}",
                    )

                    cpp_vals = _load_output(
                        os.path.join(output_dir, "output_0.bin"),
                        (rows, k),
                        torch.bfloat16,
                    )
                    cpp_idx = _load_output(
                        os.path.join(output_dir, "output_1.bin"),
                        (rows, k),
                        torch.int64,
                    )

                    v_diff = (
                        (cpp_vals.float() - ref_vals.cpu().float()).abs().max().item()
                    )
                    self.assertLess(v_diff, 0.01, f"{desc}: value diff {v_diff}")
                    self.assertTrue(
                        torch.equal(cpp_idx, ref_idx.cpu()),
                        f"{desc}: indices mismatch\n"
                        f"  cpp: {cpp_idx}\n  ref: {ref_idx.cpu()}",
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args, remaining = parser.parse_known_args()

    if args.output_dir:
        export_topk(args.output_dir)
    else:
        sys.argv = [sys.argv[0]] + remaining
        unittest.main()
