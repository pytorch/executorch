# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import filecmp
import os
import re
import unittest


class TestSchema(unittest.TestCase):
    def test_schema_sync(self) -> None:
        """Test that all copies of scalar_type.fbs are in sync.

        Flatbuffers expects all included files to be in the same directory.
        For example, program.fbs includes scalar_type.fbs, and must be in the
        same directory. As most of the schema files in executorch include
        scalar_type.fbs, it is copied in several places across the executorch
        repo. This test ensures they all remain in sync with the canonical
        version in schema/scalar_type.fbs.

        See https://github.com/pytorch/executorch/issues/11572
        """
        # make the test work in both internal and oss.
        prefix = (
            "executorch/" if os.path.exists("executorch/schema/scalar_type.fbs") else ""
        )

        # The canonical source of truth
        canonical_path = prefix + "schema/scalar_type.fbs"

        # All copies that must stay in sync with the canonical version
        copies = [
            prefix + "devtools/bundled_program/schema/scalar_type.fbs",
            prefix + "devtools/etdump/scalar_type.fbs",
            prefix + "extension/flat_tensor/serialize/scalar_type.fbs",
        ]

        for copy_path in copies:
            with self.subTest(copy=copy_path):
                self.assertTrue(
                    filecmp.cmp(canonical_path, copy_path, shallow=False),
                    f"scalar_type.fbs is out of sync: {copy_path} differs from {canonical_path}. "
                    f"Please sync the schema by copying from {canonical_path}.",
                )

    def test_schema_version_constants_in_sync(self) -> None:
        """The schema version constants must not silently drift apart.

        A PTE/PTD file's schema version is stamped by a writer and gated by a
        runtime reader. The writer value and the reader ceiling live in separate
        files, in two languages, and today are kept together only by comments.
        If a writer is bumped without its reader (or the two PTD writers
        disagree), a runtime will either reject a file it should read or read a
        file it should reject. This test turns those comments into a check.

        The rule (see schema/README.md for the compatibility policy):
          * A reader ceiling may be >= its writer version: a runtime is allowed
            to support a version before any writer emits it. It must never be
            lower, which would refuse files the matching writer produces.
          * The two PTD writers (Python and C++) stamp the same field of the
            same file, so they must be exactly equal.
          * The PTE and PTD families are independent; no relation between them.

        The values are parsed as text rather than imported: importing the
        Python writers pulls in torch, and the C++ constants have no Python
        binding. Parsing keys on the file path, not the symbol name, because
        the two reader ceilings share the name kMaxSupportedSchemaVersion.
        """
        prefix = (
            "executorch/" if os.path.exists("executorch/schema/scalar_type.fbs") else ""
        )

        def read_constant(path: str, pattern: str) -> int:
            full_path = prefix + path
            with open(full_path) as f:
                contents = f.read()
            match = re.search(pattern, contents)
            self.assertIsNotNone(
                match,
                f"Could not find the schema version constant in {full_path} "
                f"(pattern {pattern!r}). If the declaration moved or was "
                f"reformatted, update this test so the sync check keeps working.",
            )
            # match is not None: asserted above.
            return int(match.group(1))  # pyre-ignore[16]

        # PTE (program) family.
        pte_writer = read_constant(
            "exir/version.py",
            r"EXECUTORCH_SCHEMA_VERSION\s*=\s*(\d+)",
        )
        pte_reader = read_constant(
            "runtime/executor/program.h",
            r"kMaxSupportedSchemaVersion\s*=\s*(\d+)",
        )

        # PTD (data) family. Two writers stamp the same field; one reader gate.
        ptd_writer_py = read_constant(
            "extension/flat_tensor/serialize/serialize.py",
            r"_FLAT_TENSOR_VERSION\s*:\s*int\s*=\s*(\d+)",
        )
        ptd_writer_cpp = read_constant(
            "extension/flat_tensor/serialize/serialize.h",
            r"kSchemaVersion\s*=\s*(\d+)",
        )
        ptd_reader = read_constant(
            "extension/flat_tensor/flat_tensor_data_map.h",
            r"kMaxSupportedSchemaVersion\s*=\s*(\d+)",
        )

        # A writer must never stamp a version its own runtime reader refuses.
        self.assertLessEqual(
            pte_writer,
            pte_reader,
            "EXECUTORCH_SCHEMA_VERSION (exir/version.py) is higher than "
            "Program::kMaxSupportedSchemaVersion (runtime/executor/program.h). "
            "The exporter would stamp PTE files this runtime refuses. Raise the "
            "runtime ceiling before (or with) the exporter version.",
        )

        # The two PTD writers stamp the same field of the same file.
        self.assertEqual(
            ptd_writer_py,
            ptd_writer_cpp,
            "_FLAT_TENSOR_VERSION (serialize.py) and kSchemaVersion "
            "(serialize.h) disagree. Both writers stamp the same version into "
            "every PTD file and must be equal.",
        )
        self.assertLessEqual(
            ptd_writer_cpp,
            ptd_reader,
            "The PTD writers are higher than "
            "FlatTensorDataMap::kMaxSupportedSchemaVersion "
            "(flat_tensor_data_map.h). The writers would stamp PTD files this "
            "runtime refuses. Raise the runtime ceiling before (or with) the "
            "writers.",
        )


if __name__ == "__main__":
    unittest.main()
