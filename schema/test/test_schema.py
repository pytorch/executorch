# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import filecmp
import os
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
                    filecmp.cmp(canonical_path, copy_path),
                    f"scalar_type.fbs is out of sync: {copy_path} differs from {canonical_path}. "
                    f"Please sync the schema by copying from {canonical_path}.",
                )


if __name__ == "__main__":
    unittest.main()
