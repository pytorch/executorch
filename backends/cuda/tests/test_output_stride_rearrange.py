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
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXECUTORCH_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
RUNNER_PATH = os.environ.get(
    "EXECUTORCH_EXECUTOR_RUNNER",
    os.path.join(EXECUTORCH_ROOT, "cmake-out", "executor_runner"),
)


def _save_tensor(tensor, path):
    tensor = tensor.cpu().contiguous().clone()
    with open(path, "wb") as f:
        f.write(bytes(tensor.untyped_storage()))


def _load_output(path, shape, strides, dtype):
    data = np.fromfile(path, dtype=np.uint8)
    flat = torch.frombuffer(bytearray(data), dtype=dtype)
    return torch.as_strided(flat, shape, strides).clone()


def _run_rocm_runner(module, inputs, output_strides, pte, ptd, tmpdir):
    if not os.path.exists(RUNNER_PATH):
        raise RuntimeError(
            f"executor_runner not found at {RUNNER_PATH}; set "
            "EXECUTORCH_EXECUTOR_RUNNER to the ROCm runner path"
        )

    input_files = []
    for index, tensor in enumerate(inputs):
        path = os.path.join(tmpdir, f"input-{index}.bin")
        _save_tensor(tensor, path)
        input_files.append(path)

    output_base = os.path.join(tmpdir, "output")
    command = [
        RUNNER_PATH,
        f"--model_path={pte}",
        f"--inputs={','.join(input_files)}",
        f"--output_file={output_base}",
    ]
    if ptd:
        command.append(f"--data_path={ptd}")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [os.path.join(sys.prefix, "lib"), env.get("LD_LIBRARY_PATH")])
    )
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        details = (
            f"executor_runner failed:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        raise RuntimeError(details)

    with torch.no_grad():
        outputs = module(*inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    return [
        _load_output(
            f"{output_base}-{index}.bin",
            output.shape,
            output_strides[index],
            output.dtype,
        )
        for index, output in enumerate(outputs)
    ]


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
        if torch.version.hip is not None:
            from executorch.exir.tensor import stride_from_dim_order

            execution_plan = et.executorch_program.execution_plan[0]
            output_strides = [
                stride_from_dim_order(
                    execution_plan.values[index].val.sizes,
                    execution_plan.values[index].val.dim_order,
                )
                for index in execution_plan.outputs
            ]
            return _run_rocm_runner(module, inputs, output_strides, pte, ptd, tmpdir)

        from executorch.extension.pybindings.portable_lib import _load_for_executorch

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

    def test_detached_output(self):
        """A method that returns y and y.detach() gets two separate outputs."""

        class DetachedOutputModule(nn.Module):
            def forward(self, x):
                y = x * 2
                return y, y.detach()

        x = torch.randn(4, 64, device="cuda")
        module = DetachedOutputModule().eval()
        outputs = _run_et_aoti(module, (x,))
        expected = module(x)
        self.assertEqual(len(outputs), len(expected))
        for output, reference in zip(outputs, expected):
            torch.testing.assert_close(output.cpu(), reference.cpu())


class _SlicedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 32)

    def forward(self, x):
        return self.proj(x)[:, :, :14]


class _Double(nn.Module):
    def __init__(self, memory_format):
        super().__init__()
        self.memory_format = memory_format

    def forward(self, x):
        return (x * 2).contiguous(memory_format=self.memory_format)


class TestStridedOutputWithCudaGraph(unittest.TestCase):
    """A CUDA graph replay copies raw bytes, so a strided output must not reach it."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if torch.version.hip is not None:
            raise unittest.SkipTest("CUDA graphs are not supported on ROCm")

    def _run_with_cuda_graph(self, module, example, inputs, dynamic_shapes=None):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.runtime import Runtime

        with torch.no_grad():
            ep = torch.export.export(module, example, dynamic_shapes=dynamic_shapes)
        spec = CudaBackend.generate_method_name_compile_spec("forward")
        program = to_edge_transform_and_lower(
            ep, partitioner=[CudaPartitioner([spec])]
        ).to_executorch()
        runtime = Runtime.get()
        option = "enable_cuda_graph_for_method"
        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            pte = os.path.join(tmpdir, "model.pte")
            with open(pte, "wb") as f:
                program.write_to_file(f)
            program.write_tensor_data_to_file(tmpdir)
            ptd = glob.glob(os.path.join(tmpdir, "*.ptd"))[0]
            runtime.backend_registry.set_option("CudaBackend", {option: "forward"})
            try:
                method = runtime.load_program(pte, data_path=ptd).load_method("forward")
            finally:
                # The option is process wide.
                runtime.backend_registry.set_option("CudaBackend", {option: ""})
            for x in inputs:
                try:
                    results.append(method.execute([x])[0])
                except RuntimeError as e:
                    results.append(e)
        return results

    def _assert_matches_eager(self, module, x, out):
        self.assertIsInstance(out, torch.Tensor)
        with torch.no_grad():
            expected = module(x.cuda()).cpu()
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_sliced_output_matches_eager_past_warmup(self):
        torch.manual_seed(0)
        module = _SlicedLinear().cuda().eval()
        # Three warmup calls, the capture call, then replays.
        inputs = [torch.randn(1, 50, 64) for _ in range(6)]
        outputs = self._run_with_cuda_graph(module, (inputs[0].cuda(),), inputs)
        for i, (x, out) in enumerate(zip(inputs, outputs)):
            with self.subTest(call=i):
                self._assert_matches_eager(module, x, out)

    def test_strided_output_first_seen_at_capture(self):
        torch.manual_seed(0)
        module = _SlicedLinear().cuda().eval()
        # With one row the slice is contiguous, so only the capture call sees a
        # strided output.
        inputs = [torch.randn(1, rows, 64) for rows in (1, 1, 1, 50, 50, 50)]
        rows = torch.export.Dim("rows", min=1, max=64)
        outputs = self._run_with_cuda_graph(
            module, (inputs[3].cuda(),), inputs, dynamic_shapes=({1: rows},)
        )
        for i, (x, out) in enumerate(zip(inputs, outputs)):
            with self.subTest(call=i):
                if i == 3:
                    self.assertIsInstance(out, RuntimeError)
                else:
                    self._assert_matches_eager(module, x, out)

    def test_output_shape_changes_at_capture(self):
        torch.manual_seed(0)
        # The capture call has a larger batch than the warmup calls before it.
        inputs = [torch.randn(batch, 3, 4, 4) for batch in (1, 1, 1, 2, 2, 2)]
        batch = torch.export.Dim("batch", min=1, max=8)
        for memory_format in (torch.contiguous_format, torch.channels_last):
            module = _Double(memory_format).eval()
            outputs = self._run_with_cuda_graph(
                module, (inputs[3].cuda(),), inputs, dynamic_shapes=({0: batch},)
            )
            for i, (x, out) in enumerate(zip(inputs, outputs)):
                with self.subTest(memory_format=memory_format, call=i):
                    self._assert_matches_eager(module, x, out)


if __name__ == "__main__":
    unittest.main()
