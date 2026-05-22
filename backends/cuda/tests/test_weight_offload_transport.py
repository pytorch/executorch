# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transport contract for the v2 weight-offload payload: serializer
round-trip + runtime parse/install/serve (success-summary log and
[ET_WEIGHT_OFFLOAD_SKIP] no-blob-load trace are the canaries)."""

import io
import os
import struct
import subprocess
import tempfile
import unittest

import torch
import torch.nn as nn
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes.weight_offload_pass import (
    _apply_weight_offload,
    _serialize_payload,
    COMPILE_SPEC_KEY_ENABLE,
    COMPILE_SPEC_KEY_PIN_FQNS,
    named_data_key_for_method,
    PAYLOAD_KEY_CONSTANTS_METADATA,
    PAYLOAD_KEY_FLOOR,
    PAYLOAD_KEY_METHOD_NAME,
    PAYLOAD_KEY_PIN_FQNS,
    PAYLOAD_KEY_SCHEDULE,
    PAYLOAD_KEY_VERSION,
    PAYLOAD_MAGIC,
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


# --------------------------------------------------------------------------
# Pure serialization round-trip (no export, no runtime)
# --------------------------------------------------------------------------


def _parse_payload_python(blob: bytes) -> dict:
    """Mirror of the C++ parser (v2), used only by these tests to
    verify the Python serializer's output without involving the
    runtime."""
    buf = io.BytesIO(blob)

    def read(n):
        b = buf.read(n)
        if len(b) != n:
            raise ValueError("truncated payload")
        return b

    def read_i32():
        return struct.unpack("<i", read(4))[0]

    def read_u32():
        return struct.unpack("<I", read(4))[0]

    def read_i64():
        return struct.unpack("<q", read(8))[0]

    def read_u64():
        return struct.unpack("<Q", read(8))[0]

    def read_str():
        n = read_u32()
        return read(n).decode("utf-8")

    magic = read_u32()
    if magic != PAYLOAD_MAGIC:
        raise ValueError(f"bad magic: 0x{magic:08x}")
    version = read_u32()
    method_name = read_str()
    floor_bytes = read_u64()
    schedule = [read_str() for _ in range(read_u32())]
    pin_fqns = [read_str() for _ in range(read_u32())]
    constants_metadata = []
    for _ in range(read_u32()):
        fqn = read_str()
        dtype = read_i32()
        ndim = read_u32()
        sizes = [read_i64() for _ in range(ndim)]
        strides = [read_i64() for _ in range(ndim)]
        storage_offset = read_i64()
        nbytes = read_u64()
        device_type = read_i32()
        device_index = read_i32()
        constants_metadata.append(
            {
                "fqn": fqn,
                "dtype": dtype,
                "sizes": sizes,
                "strides": strides,
                "storage_offset": storage_offset,
                "nbytes": nbytes,
                "device_type": device_type,
                "device_index": device_index,
            }
        )
    if buf.read():
        raise ValueError("trailing bytes after payload")
    return {
        PAYLOAD_KEY_VERSION: version,
        PAYLOAD_KEY_METHOD_NAME: method_name,
        PAYLOAD_KEY_FLOOR: floor_bytes,
        PAYLOAD_KEY_SCHEDULE: schedule,
        PAYLOAD_KEY_PIN_FQNS: pin_fqns,
        PAYLOAD_KEY_CONSTANTS_METADATA: constants_metadata,
    }


class TestPayloadSerialization(unittest.TestCase):
    def test_round_trip_preserves_all_fields(self):
        ep = export(_TwoWeightModel(), (torch.randn(4, 64),), strict=True)
        payload = _apply_weight_offload(ep, method_name="prefill", pin_fqns=["w1"])
        blob = _serialize_payload(payload)
        parsed = _parse_payload_python(blob)
        self.assertEqual(parsed[PAYLOAD_KEY_VERSION], payload[PAYLOAD_KEY_VERSION])
        self.assertEqual(parsed[PAYLOAD_KEY_METHOD_NAME], "prefill")
        self.assertEqual(parsed[PAYLOAD_KEY_FLOOR], payload[PAYLOAD_KEY_FLOOR])
        self.assertEqual(parsed[PAYLOAD_KEY_SCHEDULE], payload[PAYLOAD_KEY_SCHEDULE])
        self.assertEqual(parsed[PAYLOAD_KEY_PIN_FQNS], ["w1"])
        # v2 metadata round-trip — one entry per unique(schedule) FQN.
        parsed_meta = parsed[PAYLOAD_KEY_CONSTANTS_METADATA]
        emitted_meta = payload[PAYLOAD_KEY_CONSTANTS_METADATA]
        self.assertEqual(len(parsed_meta), len(emitted_meta))
        for p, e in zip(parsed_meta, emitted_meta):
            self.assertEqual(p["fqn"], e["fqn"])
            self.assertEqual(p["dtype"], e["dtype"])
            self.assertEqual(p["sizes"], e["sizes"])
            self.assertEqual(p["strides"], e["strides"])
            self.assertEqual(p["storage_offset"], e["storage_offset"])
            self.assertEqual(p["nbytes"], e["nbytes"])
            self.assertEqual(p["device_type"], 1)  # CUDA
            self.assertEqual(p["device_index"], 0)

    def test_magic_is_first_four_bytes(self):
        ep = export(_TwoWeightModel(), (torch.randn(4, 64),), strict=True)
        payload = _apply_weight_offload(ep, method_name="forward")
        blob = _serialize_payload(payload)
        self.assertEqual(struct.unpack("<I", blob[:4])[0], PAYLOAD_MAGIC)


# --------------------------------------------------------------------------
# Export pipeline: does the payload land in the NamedDataStore?
# --------------------------------------------------------------------------


def _opt_in_specs(method_name: str, pin_fqns: list[str] | None = None):
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


def _et_program_from_specs(model: nn.Module, inputs, specs):
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


def _named_data_entries(et_program) -> dict[str, bytes]:
    """Flatten the executorch program's NamedDataStoreOutput entries by
    key. Inspects the private ``_named_data`` attribute on
    ``ExecutorchProgramManager`` — the public surface only exposes
    ``write_to_file`` / ``write_tensor_data_to_file`` which already
    serialize the .pte / .ptd. ``pte_data`` is ``dict[key,
    DataEntry]``; ``external_data`` is
    ``dict[tag, dict[key, DataEntry]]``; both index into ``buffers``,
    a ``list[bytes]``."""
    out: dict[str, bytes] = {}
    nd = getattr(et_program, "_named_data", None)
    if nd is None:
        return out
    for key, entry in nd.pte_data.items():
        out[key] = nd.buffers[entry.buffer_index]
    for tag_dict in nd.external_data.values():
        for key, entry in tag_dict.items():
            out[key] = nd.buffers[entry.buffer_index]
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for export")
class TestPayloadTransportExport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_payload_present_when_opt_in_set(self):
        model = _TwoWeightModel().to("cuda").eval()
        x = torch.randn(4, 64, device="cuda")
        specs = _opt_in_specs("forward")
        et_program = _et_program_from_specs(model, (x,), specs)

        entries = _named_data_entries(et_program)
        key = named_data_key_for_method("forward")
        self.assertIn(key, entries, f"missing offload payload at key {key!r}")
        parsed = _parse_payload_python(entries[key])
        self.assertEqual(parsed[PAYLOAD_KEY_METHOD_NAME], "forward")
        self.assertEqual(parsed[PAYLOAD_KEY_VERSION], 2)
        # Two distinct weights, each used once at distinct consumers
        # — schedule length is 2 minimum (could be more after view
        # duplication for the .T).
        self.assertGreaterEqual(len(parsed[PAYLOAD_KEY_SCHEDULE]), 2)
        self.assertEqual(parsed[PAYLOAD_KEY_PIN_FQNS], [])
        # Per-FQN metadata block has one entry per unique scheduled FQN
        self.assertEqual(
            sorted(m["fqn"] for m in parsed[PAYLOAD_KEY_CONSTANTS_METADATA]),
            sorted(set(parsed[PAYLOAD_KEY_SCHEDULE])),
        )

    def test_pin_fqns_round_trip_through_specs(self):
        model = _TwoWeightModel().to("cuda").eval()
        x = torch.randn(4, 64, device="cuda")
        specs = _opt_in_specs("forward", pin_fqns=["w1"])
        et_program = _et_program_from_specs(model, (x,), specs)

        entries = _named_data_entries(et_program)
        parsed = _parse_payload_python(entries[named_data_key_for_method("forward")])
        self.assertEqual(parsed[PAYLOAD_KEY_PIN_FQNS], ["w1"])

    def test_no_payload_when_opt_in_absent(self):
        model = _TwoWeightModel().to("cuda").eval()
        x = torch.randn(4, 64, device="cuda")
        specs = [CudaBackend.generate_method_name_compile_spec("forward")]
        et_program = _et_program_from_specs(model, (x,), specs)

        entries = _named_data_entries(et_program)
        self.assertNotIn(named_data_key_for_method("forward"), entries)


# --------------------------------------------------------------------------
# End-to-end: runtime parse-and-log via executor_runner
# --------------------------------------------------------------------------


def _save_tensor(t: torch.Tensor, path: str) -> None:
    with open(path, "wb") as f:
        f.write(bytes(t.cpu().contiguous().untyped_storage()))


def _write_artifact(et_program, out_dir: str) -> tuple[str, str]:
    pte = os.path.join(out_dir, "offload.pte")
    with open(pte, "wb") as f:
        et_program.write_to_file(f)
    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(out_dir)
    return pte, os.path.join(out_dir, "aoti_cuda_blob.ptd")


def _run_runner(
    runner_path: str, pte: str, ptd: str, input_file: str, out_base: str
) -> subprocess.CompletedProcess:
    cmd = [
        runner_path,
        f"--model_path={pte}",
        f"--data_path={ptd}",
        f"--inputs={input_file}",
        f"--output_file={out_base}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())


def _count_offload_log_lines(stderr: str) -> int:
    """Count only the success-path log lines (``[ET_WEIGHT_OFFLOAD]``
    followed by a space). Error lines use ``[ET_WEIGHT_OFFLOAD][ERROR]``
    and must NOT count, so the hard-fail test can assert "success log
    absent + error log present" without ambiguity."""
    return sum(1 for line in stderr.splitlines() if "[ET_WEIGHT_OFFLOAD] " in line)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for runtime")
class TestPayloadTransportRuntime(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(RUNNER_PATH):
            self.skipTest(
                f"executor_runner not found at {RUNNER_PATH}; build with "
                "cmake --build cmake-out --target executor_runner"
            )

    def _run_with(self, specs):
        """Run the model and return the raw subprocess result. Callers
        decide whether rc==0 or rc!=0 is expected."""
        model = _TwoWeightModel().to("cuda").eval()
        x = torch.randn(4, 64, device="cuda")
        et_program = _et_program_from_specs(model, (x,), specs)

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = os.path.join(tmp, "export")
            os.makedirs(export_dir)
            pte, ptd = _write_artifact(et_program, export_dir)
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            in_path = os.path.join(run_dir, "0.bin")
            _save_tensor(x, in_path)
            out_base = os.path.join(run_dir, "out")
            return _run_runner(RUNNER_PATH, pte, ptd, in_path, out_base)

    def _run_and_get_stderr(self, specs) -> str:
        result = self._run_with(specs)
        self.assertEqual(
            result.returncode,
            0,
            f"executor_runner failed (rc={result.returncode}):\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}",
        )
        return result.stderr

    def test_runtime_logs_payload_when_opt_in_set(self):
        stderr = self._run_and_get_stderr(_opt_in_specs("forward"))
        self.assertEqual(
            _count_offload_log_lines(stderr),
            1,
            f"expected one [ET_WEIGHT_OFFLOAD] log line, stderr:\n{stderr}",
        )
        # Spot-check echoed fields are visible in the log
        offload_line = next(
            line
            for line in stderr.splitlines()
            if line.startswith("[ET_WEIGHT_OFFLOAD] ")
        )
        self.assertIn("method=forward", offload_line)
        self.assertIn("version=2", offload_line)
        self.assertIn("pinned=0", offload_line)
        # Success summary includes installed=N and the no-eager-load trace.
        self.assertIn("installed=", offload_line)
        self.assertIn(
            "[ET_WEIGHT_OFFLOAD_SKIP] skipped update_constants_from_blob "
            "method=forward",
            stderr,
        )

    def test_runtime_accepts_nonempty_pin_fqns(self):
        """Commit 9a lands pinning. A method with ``pin_fqns=["w1"]``
        now loads successfully through the runtime: the pinned weight
        is allocated once at Session::create and served through the
        resident fast path (the streaming pool only sees w2).

        Asserts loud-and-correct behavior at the transport layer.
        Detailed accounting (pinned_bytes / streaming_budget stats,
        prefetch interaction) lives in the pool-side tests."""
        stderr = self._run_and_get_stderr(_opt_in_specs("forward", pin_fqns=["w1"]))
        # Success summary emitted = init went all the way through.
        self.assertEqual(
            _count_offload_log_lines(stderr),
            1,
            f"expected one [ET_WEIGHT_OFFLOAD] success log, stderr:\n{stderr}",
        )
        summary = next(
            line
            for line in stderr.splitlines()
            if line.startswith("[ET_WEIGHT_OFFLOAD] ")
        )
        # The success summary reports pinned=1 even when pinning is
        # actually exercised — the field counts payload.pin_fqns
        # length, which existed pre-9a too.
        self.assertIn("pinned=1", summary)

    def test_runtime_silent_without_opt_in(self):
        specs = [CudaBackend.generate_method_name_compile_spec("forward")]
        stderr = self._run_and_get_stderr(specs)
        self.assertEqual(
            _count_offload_log_lines(stderr),
            0,
            f"expected no [ET_WEIGHT_OFFLOAD] log line, stderr:\n{stderr}",
        )

    def test_runtime_hard_fails_on_corrupted_payload(self):
        """Locks in the runtime parser's hard-fail contract: a payload
        whose magic header has been clobbered must cause ``init`` to
        fail loudly, not silently degrade. Mutates the .pte after
        export by zeroing the four magic bytes (the longer
        ``ETWO`` + version uint32 needle avoids matching any
        coincidental occurrence elsewhere in the compiled .so)."""
        model = _TwoWeightModel().to("cuda").eval()
        x = torch.randn(4, 64, device="cuda")
        et_program = _et_program_from_specs(model, (x,), _opt_in_specs("forward"))

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = os.path.join(tmp, "export")
            os.makedirs(export_dir)
            pte, ptd = _write_artifact(et_program, export_dir)

            # Magic (``ETWO``) immediately followed by schema_version
            # ``= 2`` (little-endian uint32 ``\x02\x00\x00\x00``) is
            # vanishingly unlikely to appear by chance in the .so or
            # weights blob; safer than searching for the four-byte
            # magic alone.
            needle = b"ETWO\x02\x00\x00\x00"
            with open(pte, "rb") as f:
                data = bytearray(f.read())
            idx = data.find(needle)
            self.assertGreaterEqual(idx, 0, "expected to find offload magic in .pte")
            second = data.find(needle, idx + 1)
            self.assertLess(
                second,
                0,
                "needle appeared twice — corruption target is ambiguous",
            )
            data[idx : idx + 4] = b"JUNK"
            with open(pte, "wb") as f:
                f.write(data)

            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            in_path = os.path.join(run_dir, "0.bin")
            _save_tensor(x, in_path)
            out_base = os.path.join(run_dir, "out")
            result = _run_runner(RUNNER_PATH, pte, ptd, in_path, out_base)

        # Non-zero exit because init aborted. The stderr should
        # mention "weight offload" and "parse" so a future debugger
        # has a fighting chance.
        self.assertNotEqual(
            result.returncode,
            0,
            f"executor_runner unexpectedly succeeded with corrupted payload; "
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        # Loud-at-init contract: the error path must produce stderr
        # output regardless of ET_LOG_ENABLED so a debugger can find
        # the cause.
        self.assertIn("[ET_WEIGHT_OFFLOAD][ERROR]", result.stderr)
        self.assertIn("payload parse failed", result.stderr)
        self.assertEqual(
            _count_offload_log_lines(result.stderr),
            0,
            "the offload log line should NOT appear for a corrupted payload "
            "(parse must hard-fail before the log emits)",
        )


if __name__ == "__main__":
    unittest.main()
