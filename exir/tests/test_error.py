# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from unittest.mock import MagicMock

from executorch.exir.error import (
    ExportError,
    ExportErrorType,
    internal_assert,
    InternalError,
)


class TestError(unittest.TestCase):
    def test_internal_assert_with_string_message(self) -> None:
        internal_assert(True, "error message")
        with self.assertRaisesRegex(InternalError, "^error message$"):
            internal_assert(False, "error message")

    def test_internal_assert_defers_formatting_on_success(self) -> None:
        value = MagicMock()
        value.__str__.side_effect = AssertionError("value should not be formatted")
        internal_assert(True, "invalid value %s: %d", value, 42)
        value.__str__.assert_not_called()

    def test_internal_assert_formats_message_once_on_failure(self) -> None:
        value = MagicMock()
        value.__str__.return_value = "example"
        with self.assertRaisesRegex(InternalError, "^invalid value example: 42$"):
            internal_assert(False, "invalid value %s: %d", value, 42)
        value.__str__.assert_called_once_with()

    def test_internal_assert_preserves_percent_without_arguments(self) -> None:
        with self.assertRaisesRegex(InternalError, "^100% complete$"):
            internal_assert(False, "100% complete")

    def test_export_error_message(self) -> None:
        def throws_err() -> None:
            raise ExportError(ExportErrorType.NOT_SUPPORTED, "error message")

        with self.assertRaisesRegex(ExportError, "[ExportErrorType.NOT_SUPPORTED]"):
            throws_err()
