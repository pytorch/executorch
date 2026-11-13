# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from executorch.backends.samsung.test.utils.runtime_executor import EDBTestManager


class TestEDBTestManager(unittest.TestCase):
    def test_bundles_inputs_and_uses_four_device_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "program.pte",
                "enn_executor_runner",
                "input_0.bin",
                "input_1.bin",
            ):
                (root / name).write_bytes(name.encode())
            with patch(
                "executorch.backends.samsung.test.utils.runtime_executor.get_runner_path",
                return_value=root / "enn_executor_runner",
            ), patch(
                "executorch.backends.samsung.test.utils.runtime_executor.subprocess.Popen"
            ) as popen:
                popen.return_value.communicate.return_value = (b"", b"")
                popen.return_value.returncode = 0
                manager = EDBTestManager(
                    str(root / "program.pte"),
                    "/data/local/tmp/enn-executorch-test",
                    ["input_0.bin", "input_1.bin"],
                )
                manager.push()
                manager.execute()
                manager.pull(str(root / "output"))

            self.assertEqual(popen.call_count, 4)
            for call in popen.call_args_list:
                self.assertTrue(call.kwargs["start_new_session"])
            self.assertEqual(
                popen.return_value.communicate.call_args.kwargs, {"timeout": 300}
            )
            uploads = [
                call.args[0] for call in popen.call_args_list if "-U" in call.args[0]
            ]
            self.assertEqual(len(uploads), 1)
            with tarfile.open(uploads[0][2]) as bundle:
                self.assertEqual(
                    set(bundle.getnames()),
                    {
                        "program.pte",
                        "enn_executor_runner",
                        "input_0.bin",
                        "input_1.bin",
                    },
                )
                self.assertEqual(
                    bundle.extractfile("input_1.bin").read(), b"input_1.bin"
                )
            command = popen.call_args_list[2].args[0][-1]
            self.assertIn("tar -xf", command)
            self.assertIn('--input "input_0.bin input_1.bin"', command)

    def test_missing_input_fails_before_device_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "program.pte").write_bytes(b"model")
            (root / "enn_executor_runner").write_bytes(b"runner")
            with patch(
                "executorch.backends.samsung.test.utils.runtime_executor.get_runner_path",
                return_value=root / "enn_executor_runner",
            ), patch(
                "executorch.backends.samsung.test.utils.runtime_executor.subprocess.Popen",
            ) as popen:
                manager = EDBTestManager(
                    str(root / "program.pte"),
                    "/data/local/tmp/enn-executorch-test",
                    ["missing.bin"],
                )
                with self.assertRaises(FileNotFoundError):
                    manager.push()
                popen.assert_not_called()

    @unittest.skipUnless(
        sys.platform != "win32", "Devicefarm uses POSIX process groups"
    )
    def test_timeout_kills_children_holding_output_pipes(self):
        with patch(
            "executorch.backends.samsung.test.utils.runtime_executor.get_runner_path",
            return_value=Path("enn_executor_runner"),
        ), patch.object(EDBTestManager, "COMMAND_TIMEOUT_SECONDS", 0.2, create=True):
            manager = EDBTestManager("program.pte", "/data/local/tmp/enn-test", [])
            manager.devicefarm = sys.executable
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                manager._edb(
                    [
                        "-c",
                        "import subprocess, sys, time; "
                        "subprocess.Popen([sys.executable, '-c', "
                        "'import time; time.sleep(2)']); time.sleep(2)",
                    ]
                )
            self.assertLess(time.monotonic() - started, 1.5)

    def test_device_suites_fail_fast(self):
        root = Path(__file__).resolve().parents[4]
        script = (root / ".ci/scripts/test-samsung-models.sh").read_text()
        self.assertIn("python -m unittest -fv", script)
        self.assertIn("python -m unittest discover -fv", script)

    def test_runner_includes_portable_quantization_kernels(self):
        root = Path(__file__).resolve().parents[4]
        build = (root / "backends/samsung/build.sh").read_text()
        runner = (root / "examples/samsung/CMakeLists.txt").read_text()
        self.assertIn("-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON", build)
        self.assertIn("quantized_ops_lib", runner)
        host_build = build.split("function build_android()", 1)[0]
        self.assertIn("-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON", host_build)
        self.assertIn("-DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON", host_build)
        self.assertIn("/kernels/quantized/libquantized_ops_aot_lib.so", host_build)


if __name__ == "__main__":
    unittest.main()
