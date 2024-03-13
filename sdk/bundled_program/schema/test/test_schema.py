# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import filecmp
import unittest


class TestSchema(unittest.TestCase):
    def test_schema_sync(self) -> None:
        self.assertTrue(
            filecmp.cmp(
                "executorch/sdk/bundled_program/schema/scalar_type.fbs",
                "executorch/schema/scalar_type.fbs",
            ),
            'Please run "hg cp fbcode//executorch/schema/scalar_type.fbs fbcode//executorch/sdk/bundled_program/schema/scalar_type.fbs" to sync schema changes.',
        )
