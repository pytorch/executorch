# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A model whose weights blob is absent must fail to load, not fail later.

The CUDA backend hands a model's constants to its generated library through a
sidecar blob. When that blob is missing the container keeps null constant
pointers, and the load used to succeed: the failure surfaced from the first
execute as an illegal memory access inside a generated kernel, with nothing in
the message naming the blob.

Two payload shapes reach that code. A current export carries per-name weight
metadata and goes through the weight cache, which reports a missing blob itself.
A library built before external weights carries only the two blob keys, newline
separated, and goes through the legacy path. That legacy path is the one the
check was added to, so it is the one this file exercises, by rewriting the
payload of a real export into the older shape in place.
"""

import os
import tempfile
import unittest

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.cuda_weight_collector import CUDA_WEIGHT_CACHE_MAGIC
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from torch.export import export


class ModuleWithConstants(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("a", torch.randn(4, 64))
        self.register_buffer("b", torch.randn(64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.a
        y = y + self.b
        y = torch.relu(y)
        return y * self.a


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class TestMissingWeightsBlob(unittest.TestCase):
    def _lower(self, outdir: str) -> str:
        torch.manual_seed(0)
        module = ModuleWithConstants().eval().cuda()
        inputs = (torch.randn(4, 64, device="cuda"),)
        exported = export(module, inputs)
        spec = CudaBackend.generate_method_name_compile_spec("forward")
        lowered = to_edge_transform_and_lower(
            exported, partitioner=[CudaPartitioner([spec])]
        )
        program = lowered.to_executorch()
        path = os.path.join(outdir, "model.pte")
        with open(path, "wb") as f:
            program.write_to_file(f)
        program.write_tensor_data_to_file(outdir)
        return path

    def _rewrite_payload_as_legacy(self, path: str) -> None:
        """Replace the weight metadata payload with the older two-key payload.

        Rewriting in place keeps every offset in the file valid, and modelling the
        older shape this way avoids needing an old toolchain to produce one. The
        payload is read from the program rather than located by scanning, because a
        program with more than one delegate holds more than one payload and a scan
        cannot tell where one ends.
        """
        with open(path, "rb") as f:
            raw = f.read()
        program = deserialize_pte_binary(raw).program

        payloads = []
        for plan in program.execution_plan:
            for delegate in plan.delegates:
                if delegate.id != "CudaBackend":
                    continue
                payloads.append(
                    bytes(program.backend_delegate_data[delegate.processed.index].data)
                )
        self.assertEqual(len(payloads), 1, "expected one CUDA delegate")
        payload = payloads[0]
        self.assertTrue(
            payload.startswith(CUDA_WEIGHT_CACHE_MAGIC),
            "expected the weight metadata payload this rewrite consumes",
        )

        # The payload carries a content hash, so it occurs once.
        offset = raw.find(payload)
        self.assertGreaterEqual(offset, 0, "payload not found in the file")
        self.assertEqual(
            raw.find(payload, offset + 1), -1, "payload is not unique in the file"
        )

        # Derived from the shared library key, so it carries the library's hash
        # rather than the blob's and would not resolve even if a sidecar were
        # supplied. That is fine here: the point is that an unresolvable key now
        # fails the load rather than binding nothing.
        marker = b"_so_blob"
        end = payload.index(marker) + len(marker)
        so_key = payload[:end].rsplit(b"\x00", 1)[-1].decode("utf-8")
        weights_key = so_key.replace("_so_blob", "_weights_blob")

        # The two keys, then zeros to keep the payload its original length. The
        # runtime reads the blob key as a C string, so the filler is not part of the
        # key it looks up. Nothing resolves either way, since no sidecar is supplied.
        legacy = (so_key + "\n" + weights_key).encode("utf-8")
        self.assertLessEqual(len(legacy), len(payload), "legacy payload does not fit")
        legacy += b"\x00" * (len(payload) - len(legacy))

        blob = bytearray(raw)
        blob[offset : offset + len(payload)] = legacy
        with open(path, "wb") as f:
            f.write(bytes(blob))

    def test_load_reports_not_found_when_blob_is_absent(self) -> None:
        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as outdir:
            path = self._lower(outdir)

            blobs = [f for f in os.listdir(outdir) if f.endswith(".ptd")]
            self.assertTrue(blobs, "expected an externalized weights blob")
            # A blob holding no constants is a bare header, and a missing blob
            # really is harmless then, so the model has to carry real data for
            # this to be testing anything. The largest is also the one used below,
            # because os.listdir does not promise an order.
            sidecar = max((os.path.join(outdir, f) for f in blobs), key=os.path.getsize)
            self.assertGreater(
                os.path.getsize(sidecar), 256, "expected non-empty constants"
            )

            # A positive control first: the same program loads when its blob is
            # supplied, so a later failure is about the missing blob rather than the
            # program or the backend registration. The rewrite is covered by the
            # payload check below it, which runs after the rewrite has happened.
            Runtime.get().load_program(path, data_path=sidecar).load_method("forward")

            self._rewrite_payload_as_legacy(path)

            # Without this the test would still pass if the rewrite stopped
            # working, by exercising the weight cache path instead, which reports
            # the same error number for the same program.
            with open(path, "rb") as f:
                self.assertNotIn(
                    CUDA_WEIGHT_CACHE_MAGIC,
                    f.read(),
                    "the rewrite left the metadata payload in place",
                )

            # The blob is never supplied, so the load must fail. The runtime's
            # exception carries only the method name and the error number, so the
            # cause is asserted through the rewrite check above rather than here.
            program = Runtime.get().load_program(path)
            with self.assertRaisesRegex(
                RuntimeError, r"Failed to load method forward, error: 0x:?20"
            ):
                program.load_method("forward")

    def test_load_succeeds_when_the_model_has_no_constants(self) -> None:
        """A model with nothing to bind still loads without its weights blob.

        The refusal must not fire on a model that has no constants, and that branch
        has no other coverage. A model with no parameters or buffers already emits
        the two-key payload this loader handles, so no rewrite is needed here.
        """
        with tempfile.TemporaryDirectory() as outdir:

            class NoConstants(torch.nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.relu(x) * 2.0

            exported = export(NoConstants().eval(), (torch.randn(4, 4, device="cuda"),))
            lowered = to_edge_transform_and_lower(
                exported,
                partitioner=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
            )
            path = os.path.join(outdir, "no_constants.pte")
            with open(path, "wb") as f:
                lowered.to_executorch().write_to_file(f)

            from executorch.runtime import Runtime

            method = Runtime.get().load_program(path).load_method("forward")
            # Host tensors: the runtime moves them to the device itself.
            out = method.execute([torch.ones(4, 4)])
            torch.testing.assert_close(
                (out[0] if isinstance(out, (list, tuple)) else out).cpu(),
                torch.full((4, 4), 2.0),
            )


if __name__ == "__main__":
    unittest.main()
