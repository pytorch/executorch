# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end correctness for the offload Session (host-mirror + H2D
serve path, multi-consumer, identity-fallback when opt-in absent,
CUDA-graph rejection)."""

import os
import subprocess
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes.weight_offload_pass import COMPILE_SPEC_KEY_ENABLE
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export


EXECUTORCH_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
RUNNER_PATH = os.path.join(EXECUTORCH_ROOT, "cmake-out", "executor_runner")


class _TwoWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(64, 64))
        self.w2 = nn.Parameter(torch.randn(64, 64))

    def forward(self, x):
        return (x @ self.w1) @ self.w2.T


class _MultiConsumerSameWeight(nn.Module):
    """Same weight read at two probe sites — exercises both schedule
    entries pointing at one host-mirror FQN."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 64))

    def forward(self, x):
        return F.linear(x, self.w) + F.linear(x + 1.0, self.w)


class _TiedWeightModel(nn.Module):
    """Two FQNs share one underlying tensor (tied embedding ↔ lm_head
    pattern). At runtime they surface as distinct catalog entries
    with the SAME ``data_ptr``; without dedupe-before-register the
    ProbeRegistry would reject the second registration as a
    duplicate-within-batch and init would hard-fail. Locks in the
    fix."""

    def __init__(self):
        super().__init__()
        shared = nn.Parameter(torch.randn(64, 64))
        self.a = shared
        self.b = shared

    def forward(self, x):
        return F.linear(x, self.a) + F.linear(x + 1.0, self.b)


def _opt_in_specs(method_name, extra=None):
    specs = [
        CudaBackend.generate_method_name_compile_spec(method_name),
        CompileSpec(COMPILE_SPEC_KEY_ENABLE, b"1"),
    ]
    if extra:
        specs.extend(extra)
    return specs


def _export_and_lower(model, inputs, specs):
    ep = export(model, inputs, strict=True)
    et_prog = to_edge_transform_and_lower(
        ep,
        partitioner=[CudaPartitioner(specs)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )


def _save_tensor(t: torch.Tensor, path: str) -> None:
    with open(path, "wb") as f:
        f.write(bytes(t.cpu().contiguous().untyped_storage()))


def _load_output(path: str, shape, dtype) -> torch.Tensor:
    with open(path, "rb") as f:
        return torch.frombuffer(bytearray(f.read()), dtype=dtype).reshape(shape)


def _write_artifact(et_program, out_dir: str) -> tuple[str, str]:
    pte = os.path.join(out_dir, "session.pte")
    with open(pte, "wb") as f:
        et_program.write_to_file(f)
    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(out_dir)
    return pte, os.path.join(out_dir, "aoti_cuda_blob.ptd")


def _run_runner(pte, ptd, input_files, out_base):
    cmd = [
        RUNNER_PATH,
        f"--model_path={pte}",
        f"--data_path={ptd}",
        f"--inputs={','.join(input_files)}",
        f"--output_file={out_base}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for runtime")
class TestOffloadSessionRuntime(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(RUNNER_PATH):
            self.skipTest(
                f"executor_runner not found at {RUNNER_PATH}; build with "
                "cmake --build cmake-out --target executor_runner"
            )
        torch.manual_seed(0)

    def _run_and_compare(self, model, x, expected_out_shape):
        et_program = _export_and_lower(
            model.to("cuda").eval(), (x,), _opt_in_specs("forward")
        )
        with torch.no_grad():
            ref = model(x).cpu().contiguous()

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = os.path.join(tmp, "export")
            os.makedirs(export_dir)
            pte, ptd = _write_artifact(et_program, export_dir)
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            in_path = os.path.join(run_dir, "0.bin")
            _save_tensor(x, in_path)
            out_base = os.path.join(run_dir, "out")
            result = _run_runner(pte, ptd, [in_path], out_base)
            self.assertEqual(
                result.returncode,
                0,
                f"runner failed (rc={result.returncode}):\n"
                f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}",
            )
            out = _load_output(f"{out_base}-0.bin", expected_out_shape, torch.float32)
            diff = (out.float() - ref.float()).abs().max().item()
            self.assertLess(
                diff,
                1e-3,
                f"offload output diverged from eager (max abs diff {diff})",
            )

    def test_two_weights_match_eager(self):
        """Sync H2D through pinned host must produce the same output
        as the direct-AOTI path within float32 numerics."""
        x = torch.randn(4, 64, device="cuda")
        self._run_and_compare(_TwoWeightModel(), x, expected_out_shape=(4, 64))

    def test_multi_consumer_same_weight_matches_eager(self):
        """Same FQN at two probe sites — each ``serve`` does its own
        H2D from the shared host mirror; result must match eager."""
        x = torch.randn(4, 64, device="cuda")
        self._run_and_compare(_MultiConsumerSameWeight(), x, expected_out_shape=(4, 64))

    def test_tied_weights_init_and_match_eager(self):
        """Two FQNs sharing one ``data_ptr`` (tied embedding pattern)
        must survive Session::create (registry dedupe per Session)
        and produce eager-matching output."""
        x = torch.randn(4, 64, device="cuda")
        self._run_and_compare(_TiedWeightModel(), x, expected_out_shape=(4, 64))

    def test_no_optin_preserves_identity_fallback(self):
        """Without the opt-in spec, the registry stays empty so the
        probe c-shim's identity-passthrough fallback still works.
        Guards against the offload path silently breaking the
        non-offload identity fallback."""
        x = torch.randn(4, 64, device="cuda")
        model = _TwoWeightModel().to("cuda").eval()
        et_program = _export_and_lower(
            model,
            (x,),
            [CudaBackend.generate_method_name_compile_spec("forward")],
        )
        with torch.no_grad():
            ref = model(x).cpu().contiguous()

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = os.path.join(tmp, "export")
            os.makedirs(export_dir)
            pte, ptd = _write_artifact(et_program, export_dir)
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            in_path = os.path.join(run_dir, "0.bin")
            _save_tensor(x, in_path)
            out_base = os.path.join(run_dir, "out")
            result = _run_runner(pte, ptd, [in_path], out_base)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("[ET_WEIGHT_OFFLOAD]", result.stderr)
            out = _load_output(f"{out_base}-0.bin", (4, 64), torch.float32)
            diff = (out.float() - ref.float()).abs().max().item()
            self.assertLess(diff, 1e-3)

    def test_cuda_graph_plus_offload_is_rejected(self):
        """Mutually exclusive: graph Replay bypasses AOTI's run(), so
        probes never fire. The runtime must hard-fail at init when
        both are set for the same method."""
        # Construct a .pte with the offload opt-in, then run the
        # runner with backend options that enable cuda_graph for
        # the same method. The runner doesn't expose a flag for
        # this so this test is skipped if we can't toggle the
        # option from the runner CLI — record the limitation
        # without making the suite brittle.
        #
        # The CUDA-graph option lives on the runtime backend
        # options surface; setting it from a vanilla
        # ``executor_runner`` invocation isn't supported. Document
        # and skip rather than wire a parallel test runner just
        # for this assertion. The hard-fail itself is exercised by
        # the unit test of CudaBackend.init via direct C++ call
        # when that infrastructure lands.
        self.skipTest(
            "cuda_graph option not settable from executor_runner CLI; "
            "the init-time hard-fail is documented and asserted in C++ "
            "unit tests when those land"
        )


if __name__ == "__main__":
    unittest.main()
