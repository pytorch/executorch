# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatch contract for the AOTI probe c-shim: AOTI's wrapper emits
one aoti_torch_cuda_probe call per FX probe node, and distinct
probe_id arguments survive inductor CSE (the property the
multi-consumer schedule depends on). Counts are exfiltrated via
EXECUTORCH_WEIGHT_OFFLOAD_PROBE_TRACE."""

import os
import subprocess
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

# Importing the pass module registers ``executorch_weight_offload::probe``.
from executorch.backends.cuda.passes import weight_offload_pass  # noqa: F401
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export


EXECUTORCH_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
RUNNER_PATH = os.path.join(EXECUTORCH_ROOT, "cmake-out", "executor_runner")


_PROBE = torch.ops.executorch_weight_offload.probe


# dim chosen large enough that inductor's TRITON-only autotune backend
# (set in ``cuda_backend.py::get_aoti_compile_options``) has at least one
# valid choice for both single- and multi-consumer linears.
_TEST_DIM = 64


class _SingleConsumerProbeModel(nn.Module):
    """One weight, one probe, one consumer."""

    def __init__(self, dim: int = _TEST_DIM):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        wp = _PROBE(self.w, 0)
        return F.linear(x, wp)


class _MultiConsumerProbeModel(nn.Module):
    """One weight, two probes (distinct probe_id), two consumers.

    This is the CSE-survival case. With identical args inductor would CSE
    the two probe calls into one; distinct ``probe_id`` constants force
    AOTI to emit two c-shim calls.
    """

    def __init__(self, dim: int = _TEST_DIM):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        wp0 = _PROBE(self.w, 0)
        out0 = F.linear(x, wp0)
        wp1 = _PROBE(self.w, 1)
        out1 = F.linear(x + 1.0, wp1)
        return out0 + out1


def _save_tensor(t: torch.Tensor, path: str) -> None:
    with open(path, "wb") as f:
        f.write(bytes(t.cpu().contiguous().untyped_storage()))


def _load_output(path: str, shape, dtype) -> torch.Tensor:
    with open(path, "rb") as f:
        return torch.frombuffer(bytearray(f.read()), dtype=dtype).reshape(shape)


def _export_and_lower(model: nn.Module, inputs, out_dir: str) -> tuple[str, str]:
    """Export ``model`` through CudaPartitioner; return ``(pte_path, ptd_path)``."""
    with torch.no_grad():
        ep = export(model, inputs, strict=True)

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

    pte_path = os.path.join(out_dir, "probe_test.pte")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(out_dir)

    ptd_path = os.path.join(out_dir, "aoti_cuda_blob.ptd")
    return pte_path, ptd_path


def _run_with_probe_trace(
    runner_path: str,
    pte_path: str,
    ptd_path: str,
    input_files: list[str],
    out_base: str,
) -> subprocess.CompletedProcess:
    """Run executor_runner with ``EXECUTORCH_WEIGHT_OFFLOAD_PROBE_TRACE=1``.

    Inherits the parent environment (including ``LD_LIBRARY_PATH``) so
    the embedded AOTI ``.so`` resolves its ``GLIBCXX`` requirement
    against whatever ``libstdc++`` the caller has configured. CI's
    ``cuda.yml`` exports the conda-env lib dir before running this
    test; local invocations should do the same (e.g.
    ``LD_LIBRARY_PATH=$(conda info --base)/envs/<env>/lib pytest ...``).
    """
    env = os.environ.copy()
    env["EXECUTORCH_WEIGHT_OFFLOAD_PROBE_TRACE"] = "1"
    cmd = [
        runner_path,
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={','.join(input_files)}",
        f"--output_file={out_base}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _count_probe_lines(stderr: str) -> int:
    return sum(1 for line in stderr.splitlines() if "[ET_WEIGHT_OFFLOAD_PROBE]" in line)


class TestProbeDispatch(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if not os.path.exists(RUNNER_PATH):
            self.skipTest(
                f"executor_runner not found at {RUNNER_PATH}; "
                "build with: cmake --build cmake-out --target executor_runner"
            )

    def _run_and_check_dispatch(
        self,
        model: nn.Module,
        expected_probe_calls: int,
        out_shape: tuple,
    ) -> None:
        torch.manual_seed(0)
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        x = torch.randn(4, _TEST_DIM, dtype=torch.bfloat16, device="cuda")

        with torch.no_grad():
            ref = model(x)

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = os.path.join(tmp, "export")
            os.makedirs(export_dir)
            pte, ptd = _export_and_lower(model, (x,), export_dir)

            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            input_file = os.path.join(run_dir, "0.bin")
            _save_tensor(x, input_file)
            out_base = os.path.join(run_dir, "out")
            result = _run_with_probe_trace(
                RUNNER_PATH, pte, ptd, [input_file], out_base
            )

            self.assertEqual(
                result.returncode,
                0,
                f"executor_runner failed (rc={result.returncode}):\n"
                f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}",
            )

            actual = _count_probe_lines(result.stderr)
            self.assertEqual(
                actual,
                expected_probe_calls,
                f"expected {expected_probe_calls} [ET_WEIGHT_OFFLOAD_PROBE] "
                f"lines, got {actual}.\nstderr:\n{result.stderr}",
            )

            # Identity probe means the lowered output must match eager
            # up to bf16 accumulation noise. Use a relative threshold
            # because the multi-consumer path accumulates two mm results
            # in TRITON's mm_plus_mm fusion, which differs from eager's
            # per-linear bf16 path by a few LSBs of the magnitude.
            out = _load_output(f"{out_base}-0.bin", out_shape, torch.bfloat16).to(
                "cuda"
            )
            diff = (out.float() - ref.float()).abs().max().item()
            rel = diff / max(ref.float().abs().max().item(), 1.0)
            self.assertLess(
                rel,
                2e-2,
                f"identity-probe output diverged from eager "
                f"(max abs diff {diff}, rel {rel})",
            )

    def test_single_consumer_dispatch(self):
        """One probe in graph → one c-shim call at runtime."""
        self._run_and_check_dispatch(
            _SingleConsumerProbeModel(),
            expected_probe_calls=1,
            out_shape=(4, _TEST_DIM),
        )

    def test_multi_consumer_dispatch(self):
        """Two probes on the same weight survive CSE → two c-shim calls."""
        self._run_and_check_dispatch(
            _MultiConsumerProbeModel(),
            expected_probe_calls=2,
            out_shape=(4, _TEST_DIM),
        )


if __name__ == "__main__":
    unittest.main()
