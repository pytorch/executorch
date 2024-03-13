# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

from executorch.exir.error import ExportError, ExportErrorType


class TestError(unittest.TestCase):
    def test_export_error_message(self) -> None:
        def throws_err() -> None:
            raise ExportError(ExportErrorType.NOT_SUPPORTED, "error message")

        with self.assertRaisesRegex(ExportError, "[ExportErrorType.NOT_SUPPORTED]"):
            throws_err()
