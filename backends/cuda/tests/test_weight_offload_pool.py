# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end correctness + accounting for the bounded pool.

Custom budgets are driven through executor_runner's
--cuda_runtime_spec CLI flag (see pinning + public-knob tests below
for the canonical patterns).
"""

import os
import re
import subprocess
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes.weight_offload_pass import (
    COMPILE_SPEC_KEY_ENABLE,
    COMPILE_SPEC_KEY_PIN_FQNS,
)
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
    """Same weight at two probe sites — exercises the pool-hit path
    on the second probe. Also catches accidentally-owning borrowed
    SlimTensor wrapping: a stray deleter on the first probe's
    returned tensor would free the pool memory, and the second
    probe's read would see garbage."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 64))

    def forward(self, x):
        return F.linear(x, self.w) + F.linear(x + 1.0, self.w)


class _LargePinnedModel(nn.Module):
    """Two ~1 MB weights (256x1024 float32 each). Used by the
    public-budget tests where ``weight_offload_budget_mb=1`` (the
    minimum the int spec accepts) must be strictly below
    ``floor + pinned``. The exact floor value depends on the
    pass's formula (which may include prefetch headroom etc.) —
    tests should not hardcode it; they only need `required > 1 MB`,
    which holds reliably since pinned alone is ~1 MB."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(256, 1024))
        self.w2 = nn.Parameter(torch.randn(256, 1024))

    def forward(self, x):
        # x: (batch, 1024); both matmuls produce (batch, 256).
        return (x @ self.w1.T) + (x @ self.w2.T)


def _opt_in_specs(method_name, pin_fqns=None):
    specs = [
        CudaBackend.generate_method_name_compile_spec(method_name),
        CompileSpec(COMPILE_SPEC_KEY_ENABLE, b"1"),
    ]
    if pin_fqns:
        specs.append(
            CompileSpec(
                COMPILE_SPEC_KEY_PIN_FQNS,
                b"\x00".join(f.encode("utf-8") for f in pin_fqns),
            )
        )
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
    pte = os.path.join(out_dir, "pool.pte")
    with open(pte, "wb") as f:
        et_program.write_to_file(f)
    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(out_dir)
    return pte, os.path.join(out_dir, "aoti_cuda_blob.ptd")


def _run_runner(pte, ptd, input_files, out_base, extra_env=None, extra_cli_args=None):
    cmd = [
        RUNNER_PATH,
        f"--model_path={pte}",
        f"--data_path={ptd}",
        f"--inputs={','.join(input_files)}",
        f"--output_file={out_base}",
    ]
    if extra_cli_args:
        cmd.extend(extra_cli_args)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


_STATS_RE = re.compile(
    r"\[ET_WEIGHT_OFFLOAD_STATS\] method=\S+ hits=(?P<hits>\d+) "
    r"misses=(?P<misses>\d+) evictions=(?P<evictions>\d+) "
    r"bytes_h2d=(?P<bytes_h2d>\d+) peak_live_bytes=(?P<peak>\d+) "
    r"budget=(?P<budget>\d+) floor=(?P<floor>\d+) "
    r"prefetch_attempted=(?P<prefetch_attempted>\d+) "
    r"prefetch_succeeded=(?P<prefetch_succeeded>\d+) "
    r"pinned_bytes=(?P<pinned_bytes>\d+) "
    r"streaming_budget=(?P<streaming_budget>\d+)"
)


def _parse_stats(stderr: str) -> dict[str, int] | None:
    m = _STATS_RE.search(stderr)
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for runtime")
class TestPoolRuntime(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(RUNNER_PATH):
            self.skipTest(
                f"executor_runner not found at {RUNNER_PATH}; build with "
                "cmake --build cmake-out --target executor_runner"
            )
        torch.manual_seed(0)

    def _export(self, model, x, pin_fqns=None):
        return _export_and_lower(
            model.to("cuda").eval(),
            (x,),
            _opt_in_specs("forward", pin_fqns=pin_fqns),
        )

    def _run_in_tmp(
        self,
        et_program,
        x,
        tmp,
        extra_env_overrides=None,
        extra_cli_args=None,
    ):
        export_dir = os.path.join(tmp, "export")
        os.makedirs(export_dir)
        pte, ptd = _write_artifact(et_program, export_dir)
        run_dir = os.path.join(tmp, "run")
        os.makedirs(run_dir)
        in_path = os.path.join(run_dir, "0.bin")
        _save_tensor(x, in_path)
        out_base = os.path.join(run_dir, "out")
        env = {"EXECUTORCH_WEIGHT_OFFLOAD_TRACE": "1"}
        if extra_env_overrides:
            env.update(extra_env_overrides)
        result = _run_runner(
            pte,
            ptd,
            [in_path],
            out_base,
            extra_env=env,
            extra_cli_args=extra_cli_args,
        )
        return result, out_base

    def test_correctness_with_default_budget(self):
        """Pool + event-ordered H2D produces eager-matching output
        with the default (floor) budget."""
        x = torch.randn(4, 64, device="cuda")
        model = _TwoWeightModel().to("cuda").eval()
        with torch.no_grad():
            ref = model(x).cpu().contiguous()

        et_program = self._export(model, x)
        with tempfile.TemporaryDirectory() as tmp:
            result, out_base = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(
                result.returncode,
                0,
                f"runner failed (rc={result.returncode}):\n"
                f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}",
            )
            out = _load_output(f"{out_base}-0.bin", (4, 64), torch.float32)
            diff = (out.float() - ref.float()).abs().max().item()
            self.assertLess(diff, 1e-3, f"max abs diff {diff}")

    def test_pool_hit_within_execute(self):
        """Two probes on the same FQN: first miss, second hit. Also
        the borrowed-tensor canary — a stray owning deleter would
        free the pool memory between probes and the second read
        would see garbage, producing numerical mismatch."""
        x = torch.randn(4, 64, device="cuda")
        model = _MultiConsumerSameWeight().to("cuda").eval()
        with torch.no_grad():
            ref = model(x).cpu().contiguous()

        et_program = self._export(model, x)
        with tempfile.TemporaryDirectory() as tmp:
            result, out_base = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)

            # Numerical canary first — if borrowed-tensor wrapping is
            # broken, the second probe corrupts and this fails before
            # we even check stats.
            out = _load_output(f"{out_base}-0.bin", (4, 64), torch.float32)
            diff = (out.float() - ref.float()).abs().max().item()
            self.assertLess(diff, 1e-3, f"max abs diff {diff}")

            stats = _parse_stats(result.stderr)
            self.assertIsNotNone(
                stats, f"missing [ET_WEIGHT_OFFLOAD_STATS]:\n{result.stderr}"
            )
            self.assertEqual(stats["misses"], 1, f"expected 1 miss, got {stats}")
            self.assertGreaterEqual(stats["hits"], 1, f"expected hit, got {stats}")
            self.assertEqual(stats["evictions"], 0)

    def test_peak_live_bytes_within_budget(self):
        """Accounting invariant: ``peak_live_bytes <= budget`` at the
        end of the run. Direct canary for software-cap bugs."""
        x = torch.randn(4, 64, device="cuda")
        et_program = self._export(_TwoWeightModel(), x)
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = _parse_stats(result.stderr)
            self.assertIsNotNone(stats, result.stderr)
            self.assertGreater(stats["budget"], 0)
            self.assertLessEqual(
                stats["peak"],
                stats["budget"],
                f"peak_live_bytes={stats['peak']} exceeded budget={stats['budget']}",
            )

    def test_stats_log_present_with_trace(self):
        """Sanity: the [ET_WEIGHT_OFFLOAD_STATS] line is emitted at
        Session destruction when the trace env var is set."""
        x = torch.randn(4, 64, device="cuda")
        et_program = self._export(_TwoWeightModel(), x)
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[ET_WEIGHT_OFFLOAD_STATS]", result.stderr)

    def test_blob_load_skipped_for_offload(self):
        """Commit 7 contract: offload-enabled methods MUST skip
        ``update_constants_from_blob`` entirely. The
        ``[ET_WEIGHT_OFFLOAD_SKIP]`` stderr trace is the canary —
        unconditional (not ET_LOG_INFO gated), so this works on any
        build. Without this trace we would silently regress to the
        eager-load path."""
        x = torch.randn(4, 64, device="cuda")
        et_program = self._export(_TwoWeightModel(), x)
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "[ET_WEIGHT_OFFLOAD_SKIP] skipped update_constants_from_blob "
                "method=forward",
                result.stderr,
            )
            # Defense-in-depth: the success summary's `installed=N`
            # field must be non-zero, proving we DID install dummies
            # (otherwise AOTI's run() would have read the un-loaded
            # blob and either segfaulted or produced garbage).
            summary = next(
                line
                for line in result.stderr.splitlines()
                if line.startswith("[ET_WEIGHT_OFFLOAD] ")
            )
            self.assertRegex(summary, r"installed=\d+")
            self.assertNotIn("installed=0", summary)

    def test_prefetch_converts_second_probe_to_pool_hit(self):
        """Depth-1 opportunistic prefetch: after the first cold probe,
        every subsequent distinct-FQN probe should hit the pool
        because the prior serve() prefetched it.

        Asserts the stats canary: pool_misses == 1 (just the first
        cold weight; everything else hits because prior serve()
        prefetched it) and prefetch_succeeded >= 1.

        Note: "pool hit" only means the live_ entry was already
        present — the hit path still does cudaStreamWaitEvent on the
        ready_event, so the consuming kernel can stall briefly if the
        prefetch H2D hasn't finished. A true "no-stall" assertion
        needs wall-clock measurement (separate scope from this stats
        canary).

        Wraparound semantics also exercised: _TwoWeightModel's
        schedule has 2 distinct FQNs (w1, w2). serve(probe_id=0)
        misses on the first FQN, prefetches the second. serve(probe_id=1)
        hits the second FQN, prefetches the first (wraparound).
        End-state: 1 miss, 1+ hit, at least 1 successful prefetch."""
        x = torch.randn(4, 64, device="cuda")
        model = _TwoWeightModel().to("cuda").eval()
        with torch.no_grad():
            ref = model(x).cpu().contiguous()

        et_program = self._export(model, x)
        with tempfile.TemporaryDirectory() as tmp:
            result, out_base = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)

            # Correctness canary first — prefetch must not corrupt.
            out = _load_output(f"{out_base}-0.bin", (4, 64), torch.float32)
            diff = (out.float() - ref.float()).abs().max().item()
            self.assertLess(diff, 1e-3, f"max abs diff {diff}")

            stats = _parse_stats(result.stderr)
            self.assertIsNotNone(
                stats, f"missing [ET_WEIGHT_OFFLOAD_STATS]:\n{result.stderr}"
            )
            self.assertEqual(
                stats["misses"],
                1,
                f"expected exactly 1 cold miss; got {stats}",
            )
            self.assertGreaterEqual(
                stats["prefetch_succeeded"],
                1,
                f"expected at least 1 successful prefetch; got {stats}",
            )
            # attempted >= succeeded by construction; the assertion
            # is loose because some attempts can be "already live"
            # short-circuits that bump neither counter.
            self.assertGreaterEqual(
                stats["prefetch_attempted"],
                stats["prefetch_succeeded"],
            )

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def test_pinning_default_budget_covers_pinned(self):
        """With a non-empty pin_fqns and no explicit budget spec, the
        runtime's default budget must be ``floor + pinned`` (not just
        floor — the floor formula excludes pinned weights). Verifies
        the v3 default-budget-with-pins fix: init succeeds, stats show
        ``streaming_budget >= floor`` and ``total = streaming +
        pinned``."""
        x = torch.randn(4, 64, device="cuda")
        model = _TwoWeightModel().to("cuda").eval()
        with torch.no_grad():
            ref = model(x).cpu().contiguous()

        et_program = self._export(model, x, pin_fqns=["w1"])
        with tempfile.TemporaryDirectory() as tmp:
            result, out_base = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)

            out = _load_output(f"{out_base}-0.bin", (4, 64), torch.float32)
            diff = (out.float() - ref.float()).abs().max().item()
            self.assertLess(diff, 1e-3, f"max abs diff {diff}")

            stats = _parse_stats(result.stderr)
            self.assertIsNotNone(stats, result.stderr)
            # w1 is 64*64*4 = 16384 bytes (Float32).
            self.assertEqual(stats["pinned_bytes"], 16384)
            # total_budget = streaming + pinned (the default formula).
            self.assertEqual(
                stats["streaming_budget"] + stats["pinned_bytes"],
                stats["budget"],
            )
            # Streaming budget must cover the floor.
            self.assertGreaterEqual(stats["streaming_budget"], stats["floor"])

    def test_pinning_pinned_fqn_resident_no_streaming_h2d(self):
        """The pinned weight (w1) is allocated once at init via the
        out-of-pool path; the streaming pool only sees w2. So
        ``bytes_h2d_copied`` reflects ONLY w2's nbytes, not w1 + w2."""
        x = torch.randn(4, 64, device="cuda")
        model = _TwoWeightModel().to("cuda").eval()
        with torch.no_grad():
            ref = model(x).cpu().contiguous()

        et_program = self._export(model, x, pin_fqns=["w1"])
        with tempfile.TemporaryDirectory() as tmp:
            result, out_base = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)

            out = _load_output(f"{out_base}-0.bin", (4, 64), torch.float32)
            diff = (out.float() - ref.float()).abs().max().item()
            self.assertLess(diff, 1e-3, f"max abs diff {diff}")

            stats = _parse_stats(result.stderr)
            self.assertIsNotNone(stats, result.stderr)
            # Each weight is 16384 bytes. With w1 pinned, the streaming
            # pool's H2D total covers only w2 (plus any wraparound
            # prefetch — but prefetch on w1 short-circuits because
            # it's pinned, so wraparound from w2's serve targets w1
            # = pinned = no streaming H2D). Net: bytes_h2d_copied
            # should be exactly w2.nbytes = 16384, NOT 32768.
            self.assertEqual(stats["pinned_bytes"], 16384)
            self.assertEqual(
                stats["bytes_h2d"],
                16384,
                f"expected streaming H2D = w2 only (16384); got {stats}",
            )

    def test_pinning_pinned_then_streaming_still_prefetches(self):
        """Pinned fast path must still call opportunistic_prefetch at
        the end of serve() so a pinned→streaming transition doesn't
        lose overlap. Stats canary: ``prefetch_attempted >= 1`` even
        though one of the two probes hit the pinned fast path."""
        x = torch.randn(4, 64, device="cuda")
        model = _TwoWeightModel().to("cuda").eval()
        et_program = self._export(model, x, pin_fqns=["w1"])
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run_in_tmp(et_program, x, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = _parse_stats(result.stderr)
            self.assertIsNotNone(stats, result.stderr)
            # The pinned serve for w1 triggers prefetch for w2; that
            # prefetch should be ATTEMPTED (and succeed, given default
            # budget covers both). If the pinned fast path returned
            # before opportunistic_prefetch, this would be 0.
            self.assertGreaterEqual(
                stats["prefetch_attempted"],
                1,
                f"pinned fast path did not call opportunistic_prefetch; "
                f"got {stats}",
            )

    # ------------------------------------------------------------------
    # Public knobs
    # ------------------------------------------------------------------

    def test_runtime_accepts_public_budget_mb_via_runner_flag(self):
        """The ``--cuda_runtime_spec=weight_offload_budget_mb=N``
        runner flag drives the public runtime spec. Init succeeds and
        the success-summary's ``budget_bytes`` reflects ``N * 1 MB``."""
        x = torch.randn(4, 1024, device="cuda")
        model = _LargePinnedModel().to("cuda").eval()
        et_program = self._export(model, x)
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run_in_tmp(
                et_program,
                x,
                tmp,
                extra_cli_args=[
                    "--cuda_runtime_spec=weight_offload_budget_mb=4",
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = _parse_stats(result.stderr)
            self.assertIsNotNone(stats, result.stderr)
            # 4 MB exactly.
            self.assertEqual(stats["budget"], 4 << 20)

    def test_pinning_below_floor_with_pinned_hard_fails(self):
        """Inject a budget below ``floor + pinned`` via the public
        runner flag. Init hard-fails with the new descriptive UX
        message naming pinned bytes, streaming floor, required total,
        and the spec the user set."""
        x = torch.randn(4, 1024, device="cuda")
        model = _LargePinnedModel().to("cuda").eval()
        et_program = self._export(model, x, pin_fqns=["w1"])
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run_in_tmp(
                et_program,
                x,
                tmp,
                extra_cli_args=[
                    # 1 MB — strictly below required (pinned w1
                    # alone is 1 MB; streaming floor adds the other
                    # ~1 MB; required > 1 MB).
                    "--cuda_runtime_spec=weight_offload_budget_mb=1",
                ],
            )
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertIn("[ET_WEIGHT_OFFLOAD][ERROR]", result.stderr)
            self.assertIn("pinned constants", result.stderr)
            self.assertIn("streaming pool floor", result.stderr)
            self.assertIn("required total", result.stderr)

    def test_floor_message_names_public_spec_when_user_set(self):
        """When the user set ``weight_offload_budget_mb``, the
        below-floor message must name that public spec (not the
        internal byte spec) so the suggested fix is copy-pasteable."""
        x = torch.randn(4, 1024, device="cuda")
        model = _LargePinnedModel().to("cuda").eval()
        et_program = self._export(model, x, pin_fqns=["w1"])
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run_in_tmp(
                et_program,
                x,
                tmp,
                extra_cli_args=[
                    "--cuda_runtime_spec=weight_offload_budget_mb=1",
                ],
            )
            self.assertNotEqual(result.returncode, 0, result.stderr)
            # Names the public spec the user passed.
            self.assertIn("weight_offload_budget_mb", result.stderr)
            # Echoes the user-supplied value.
            self.assertIn("weight_offload_budget_mb=1", result.stderr)
            # Suggested-fix line points at the public spec.
            self.assertIn("Set weight_offload_budget_mb >=", result.stderr)


if __name__ == "__main__":
    unittest.main()
