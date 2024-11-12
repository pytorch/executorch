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
        # make the test work in both internal and oss.
        prefix = "executorch/" if os.path.exists("executorch/schema/common.fbs") else ""

        self.assertTrue(
            filecmp.cmp(
                prefix + "devtools/bundled_program/schema/common.fbs",
                prefix + "schema/common.fbs",
            ),
            'Please run "hg cp fbcode//executorch/schema/common.fbs fbcode//executorch/devtools/bundled_program/schema/common.fbs" to sync schema changes.',
        )
