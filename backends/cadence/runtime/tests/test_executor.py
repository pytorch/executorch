# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest import mock

from executorch.backends.cadence.runtime import executor
from executorch.backends.cadence.runtime.executor import execute, Executor


class ExecuteTest(unittest.TestCase):
    def setUp(self) -> None:
        # execute() drops PYTHONPATH from the environment.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_failure_message_leads_with_the_logged_errors(self) -> None:
        # printf keeps the logged lines out of the command.
        command = (
            "printf 'I %s\\nE %s\\n' 'Running method forward' "
            "'Missing operator: [0] aten::abs.out' >&2; exit 1"
        )
        with self.assertRaises(RuntimeError) as context:
            execute(["sh", "-c", command])

        self.assertEqual(
            str(context.exception),
            "E Missing operator: [0] aten::abs.out\nsh failed with exit code 1.",
        )

    def test_failure_message_keeps_stderr_written_after_stdout_closes(self) -> None:
        command = "exec 1>&-; sleep 0.2; printf 'E %s\\n' late-error >&2; exit 1"
        with self.assertRaises(RuntimeError) as context:
            execute(["sh", "-c", command])

        self.assertEqual(
            str(context.exception), "E late-error\nsh failed with exit code 1."
        )

    def test_failure_message_without_stderr(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            execute(["sh", "-c", "exit 3"])

        self.assertEqual(str(context.exception), "sh failed with exit code 3.")

    def test_failure_message_names_the_signal(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            execute(["sh", "-c", "kill -ABRT $$"])

        self.assertEqual(str(context.exception), "sh was killed by signal 6.")

    def test_decodes_a_character_split_across_reads(self) -> None:
        # U+2019 is 3 bytes in UTF-8; the child writes its first byte alone.
        command = "printf '\\342' >&2; sleep 0.2; printf '\\200\\231\\n' >&2; exit 1"
        with self.assertRaises(RuntimeError) as context:
            execute(["sh", "-c", command])

        self.assertEqual(str(context.exception), "’\nsh failed with exit code 1.")


class ExecutorTest(unittest.TestCase):
    def test_passes_the_debug_buffer_size_to_the_runner(self) -> None:
        with mock.patch.object(
            executor.os.path, "isfile", return_value=True
        ), mock.patch.object(executor, "execute") as run:
            Executor("work", debug_buffer_size=4096)()

        run.assert_called_once()
        self.assertIn("--debug_buffer_size=4096", run.call_args.args[0])

    def test_builds_a_missing_runner(self) -> None:
        runner = Executor("work", debug_buffer_size=4096)
        with mock.patch.object(
            executor.os.path, "isfile", return_value=False
        ), mock.patch.object(executor, "execute") as run:
            runner()

        self.assertEqual(
            [call.args[0][0] for call in run.call_args_list],
            [runner.executor_builder, runner.execute_runner],
        )
