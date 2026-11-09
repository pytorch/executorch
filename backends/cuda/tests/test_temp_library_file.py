# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The CUDA backend loads its compiled code from a file in the temp directory.

Loading a method writes that library to a new file in the system temporary
directory and opens it from there. The file has to be a new one, readable only
by its owner, and it has to be removed once the program is freed.
"""

import os
import stat
import tempfile
import unittest
from unittest.mock import patch

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
@unittest.skipIf(os.name == "nt", "checks POSIX file permissions")
class TestTempLibraryFile(unittest.TestCase):
    def _lower(self, outdir: str) -> str:
        exported = export(AddOne().eval(), (torch.randn(4, device="cuda"),))
        lowered = to_edge_transform_and_lower(
            exported,
            partitioner=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
        )
        path = os.path.join(outdir, "add_one.pte")
        with open(path, "wb") as f:
            lowered.to_executorch().write_to_file(f)
        return path

    def test_library_file_is_new_and_owner_only(self) -> None:
        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as outdir, tempfile.TemporaryDirectory() as tmpdir:
            path = self._lower(outdir)

            # A permissive umask, so the mode check below does not pass by luck.
            old_umask = os.umask(0o022)
            try:
                with patch.dict(os.environ, {"TMPDIR": tmpdir}):
                    program = Runtime.get().load_program(path)
                    method = program.load_method("forward")
            finally:
                os.umask(old_umask)

            created = os.listdir(tmpdir)
            self.assertEqual(len(created), 1, created)
            mode = os.lstat(os.path.join(tmpdir, created[0])).st_mode
            self.assertTrue(stat.S_ISREG(mode))
            self.assertEqual(stat.S_IMODE(mode), 0o600)

            out = method.execute([torch.zeros(4)])
            torch.testing.assert_close(
                (out[0] if isinstance(out, (list, tuple)) else out).cpu(),
                torch.ones(4),
            )

            del method, program
            self.assertEqual(os.listdir(tmpdir), [])

    def test_missing_temp_dir_is_a_load_error(self) -> None:
        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as outdir:
            path = self._lower(outdir)
            missing = os.path.join(outdir, "missing")
            with patch.dict(os.environ, {"TMPDIR": missing}):
                program = Runtime.get().load_program(path)
                with self.assertRaisesRegex(
                    RuntimeError, r"Failed to load method forward, error: 0x:?22"
                ):
                    program.load_method("forward")


if __name__ == "__main__":
    unittest.main()
