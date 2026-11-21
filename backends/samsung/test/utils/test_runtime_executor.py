# Copyright 2025 (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import subprocess
import unittest
from unittest.mock import patch

from executorch.backends.samsung.test.utils.runtime_executor import EDBTestManager


class TestEDBTestManager(unittest.TestCase):
    def setUp(self):
        self.test_manager = EDBTestManager("program.pte", "/remote/work", [])

    @patch("executorch.backends.samsung.test.utils.runtime_executor.time.sleep")
    @patch("executorch.backends.samsung.test.utils.runtime_executor.subprocess.run")
    def test_upload_retries_and_succeeds(self, run, sleep):
        run.side_effect = [
            subprocess.CompletedProcess("devicefarm-cli", 1, b"", b"temporary failure"),
            subprocess.CompletedProcess("devicefarm-cli", 0, b"uploaded", b""),
        ]

        self.test_manager._edb(["-U", "program.pte", "/remote/work"])

        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("executorch.backends.samsung.test.utils.runtime_executor.time.sleep")
    @patch("executorch.backends.samsung.test.utils.runtime_executor.subprocess.run")
    def test_upload_fails_after_max_attempts(self, run, sleep):
        run.return_value = subprocess.CompletedProcess(
            "devicefarm-cli", 1, b"", b"persistent failure"
        )

        with self.assertRaisesRegex(RuntimeError, "command execute failed"):
            self.test_manager._edb(["-U", "program.pte", "/remote/work"])

        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    @patch("executorch.backends.samsung.test.utils.runtime_executor.time.sleep")
    @patch("executorch.backends.samsung.test.utils.runtime_executor.subprocess.run")
    def test_non_upload_command_is_not_retried(self, run, sleep):
        run.return_value = subprocess.CompletedProcess(
            "devicefarm-cli", 1, b"", b"persistent failure"
        )

        with self.assertRaisesRegex(RuntimeError, "command execute failed"):
            self.test_manager._edb(["-E", "command"])

        run.assert_called_once()
        sleep.assert_not_called()
