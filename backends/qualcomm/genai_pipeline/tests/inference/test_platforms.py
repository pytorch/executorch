# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.inference.platforms import (
    DevicePlatform,
    get_runner_platform,
    RunnerPlatform,
    X86Platform,
)

_DEVICE = "executorch.backends.qualcomm.export_utils.Device"
_RUNNER = "examples/qualcomm/oss_scripts/llama/qnn_llama_runner"


def _make_qnn_config(build_folder="/tmp/build"):
    """Create a stand-in QnnConfig carrying the build folder."""
    return SimpleNamespace(build_folder=build_folder)


class TestGetRunnerPlatform(unittest.TestCase):
    def test_selects_x86_when_enabled(self):
        """The x86 flag selects the host platform."""
        platform = get_runner_platform(_make_qnn_config(), enable_x86_64=True)
        self.assertIsInstance(platform, X86Platform)

    def test_selects_device_by_default(self):
        """Without the x86 flag the device platform is selected."""
        platform = get_runner_platform(_make_qnn_config(), enable_x86_64=False)
        self.assertIsInstance(platform, DevicePlatform)

    def test_both_platforms_satisfy_the_protocol(self):
        """Both platforms implement RunnerPlatform."""
        for platform in (
            DevicePlatform(_make_qnn_config()),
            X86Platform(_make_qnn_config()),
        ):
            with self.subTest(platform=type(platform).__name__):
                self.assertIsInstance(platform, RunnerPlatform)


class TestDevicePlatform(unittest.TestCase):
    def setUp(self):
        patcher = patch(_DEVICE)
        self.mock_device_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.device = self.mock_device_cls.return_value
        self.device.output_folder = "/data/local/tmp/ws/outputs"
        self.qnn_config = _make_qnn_config()

    def _setup_platform(self):
        """Create a DevicePlatform whose session holds a PTE and a tokenizer."""
        platform = DevicePlatform(self.qnn_config, workspace="/data/local/tmp/ws")
        platform.setup(
            _RUNNER, [Path("/host/m.pte"), Path("/host/tok.json")], Path("/host/out")
        )
        return platform

    def test_target_facts(self):
        """Device runs always pass --shared_buffer and do not limit generation."""
        self.assertEqual(DevicePlatform.target_flags, ("--shared_buffer",))
        self.assertFalse(DevicePlatform.limits_generation)

    def test_default_workspace_is_per_user(self):
        """The default workspace is a per-user folder."""
        platform = DevicePlatform(self.qnn_config)
        platform.setup(_RUNNER, [], Path("/host/out"))
        self.assertEqual(
            self.mock_device_cls.call_args.kwargs["workspace"],
            f"/data/local/tmp/{getpass.getuser()}/genai_pipeline",
        )

    def test_setup_pushes_files_without_pte_path(self):
        """setup pushes the files itself, so Device gets no pte_path to re-push."""
        self._setup_platform()
        self.mock_device_cls.assert_called_once_with(
            qnn_config=self.qnn_config,
            pte_path=[],
            workspace="/data/local/tmp/ws",
            runner=_RUNNER,
        )
        self.device.push.assert_called_once_with(
            inputs=None, files=["/host/m.pte", "/host/tok.json"]
        )

    def test_add_file_pushes_without_reinitializing(self):
        """add_file pushes one file without clearing the workspace."""
        platform = self._setup_platform()
        platform.add_file(Path("/host/input_tokens.raw"))
        self.device.push.assert_called_with(
            inputs=[], files=["/host/input_tokens.raw"], init_env=False
        )

    def test_runner_paths_are_bare_names(self):
        """Files and input lists are referred to by name in the workspace."""
        platform = DevicePlatform(self.qnn_config)
        self.assertEqual(platform.runner_path(Path("/host/dir/m.pte")), "m.pte")
        self.assertEqual(
            platform.runner_input_list(Path("/host/dir/list.txt")), "list.txt"
        )

    def test_output_dir_is_the_session_output_folder(self):
        """The runner writes to the Device output folder."""
        self.assertEqual(
            self._setup_platform().output_dir(), "/data/local/tmp/ws/outputs"
        )

    def test_execute_runs_from_the_workspace(self):
        """execute runs the runner by name from the workspace."""
        self._setup_platform().execute(["--a 1", "--b"])
        self.device.execute.assert_called_once_with(
            custom_runner_cmd="cd /data/local/tmp/ws && ./qnn_llama_runner --a 1 --b"
        )

    def test_pull_copies_from_the_output_folder(self):
        """pull copies one output file to the given host path."""
        self._setup_platform().pull("outputs.txt", Path("/host/out/outputs.txt"))
        self.device.pull.assert_called_once_with(
            host_output_path="/host/out/outputs.txt",
            device_output_path="/data/local/tmp/ws/outputs/outputs.txt",
        )

    def test_session_methods_raise_before_setup(self):
        """Methods that need the device session raise RuntimeError before setup."""
        platform = DevicePlatform(self.qnn_config)
        for call in (
            lambda: platform.add_file(Path("/host/f")),
            platform.output_dir,
            lambda: platform.execute([]),
            lambda: platform.pull("f", Path("/host/f")),
        ):
            with self.assertRaisesRegex(RuntimeError, "setup"):
                call()


class TestX86Platform(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.build_folder = self.tmp_dir / "build-x86"
        self.host_dir = self.tmp_dir / "outputs"
        self.host_dir.mkdir()
        self.platform = X86Platform(_make_qnn_config(str(self.build_folder)))

    def tearDown(self):
        self._tmp.cleanup()

    def _setup(self):
        """Start the platform's session with a fake QNN SDK root."""
        with patch.dict(os.environ, {"QNN_SDK_ROOT": "/opt/qnn"}):
            self.platform.setup(_RUNNER, [], self.host_dir)

    def test_target_facts(self):
        """The emulator passes no target flags and limits generation."""
        self.assertEqual(X86Platform.target_flags, ())
        self.assertTrue(X86Platform.limits_generation)

    def test_setup_requires_qnn_sdk_root(self):
        """setup raises when QNN_SDK_ROOT is unset."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "QNN_SDK_ROOT"):
                self.platform.setup(_RUNNER, [], self.host_dir)

    def test_runner_path_is_absolute(self):
        """The runner reads host files by absolute path."""
        relative = Path(os.path.relpath(self.tmp_dir / "m.pte"))
        self.assertEqual(
            self.platform.runner_path(relative), str((self.tmp_dir / "m.pte").resolve())
        )

    def test_runner_input_list_rewrites_entries_to_absolute_paths(self):
        """Each relative entry is prefixed with the list's own directory."""
        input_list = self.tmp_dir / "list.txt"
        input_list.write_text("a_0_0.raw a_0_1.raw\nb_1_0.raw\n")
        result = self.platform.runner_input_list(input_list)
        list_dir = self.tmp_dir.resolve()
        self.assertEqual(result, str(input_list.resolve()))
        self.assertEqual(
            input_list.read_text(),
            f"{list_dir}/a_0_0.raw {list_dir}/a_0_1.raw\n{list_dir}/b_1_0.raw\n",
        )

    def test_runner_input_list_is_idempotent(self):
        """Rewriting an already rewritten list changes nothing."""
        input_list = self.tmp_dir / "list.txt"
        input_list.write_text("a.raw\n")
        self.platform.runner_input_list(input_list)
        once = input_list.read_text()
        self.platform.runner_input_list(input_list)
        self.assertEqual(input_list.read_text(), once)

    def test_add_file_does_nothing(self):
        """The runner reads files in place, so add_file transfers nothing."""
        self.assertIsNone(self.platform.add_file(self.tmp_dir / "absent.raw"))

    def test_output_dir_is_host_dir(self):
        """The runner writes its outputs to the session's host_dir."""
        self._setup()
        self.assertEqual(self.platform.output_dir(), str(self.host_dir))

    @patch(
        "executorch.backends.qualcomm.genai_pipeline.inference.platforms.x86.subprocess.run"
    )
    def test_execute_runs_runner_with_qnn_libraries(self, mock_run):
        """execute launches the build-folder runner with the QNN x86 libraries."""
        mock_run.return_value = MagicMock(returncode=0)
        self._setup()
        self.platform.execute(["--a 1"])
        command = mock_run.call_args.args[0]
        self.assertIn(
            "export LD_LIBRARY_PATH=/opt/qnn/lib/x86_64-linux-clang/:", command
        )
        self.assertIn(f"{self.build_folder.resolve()}/{_RUNNER} --a 1", command)

    @patch(
        "executorch.backends.qualcomm.genai_pipeline.inference.platforms.x86.subprocess.run"
    )
    def test_execute_raises_on_non_zero_exit(self, mock_run):
        """A failing runner raises RuntimeError carrying its output."""
        mock_run.return_value = MagicMock(returncode=1, stdout="boom")
        self._setup()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.platform.execute([])

    def test_pull_moves_output_to_host_path(self):
        """pull moves an output file out of host_dir."""
        self._setup()
        (self.host_dir / "outputs.txt").write_text("text")
        destination = self.tmp_dir / "collected.txt"
        self.platform.pull("outputs.txt", destination)
        self.assertEqual(destination.read_text(), "text")
        self.assertFalse((self.host_dir / "outputs.txt").exists())

    def test_session_methods_raise_before_setup(self):
        """Methods that need the session raise RuntimeError before setup."""
        for call in (
            self.platform.output_dir,
            lambda: self.platform.execute([]),
            lambda: self.platform.pull("f", self.tmp_dir / "f"),
        ):
            with self.assertRaisesRegex(RuntimeError, "setup"):
                call()


if __name__ == "__main__":
    unittest.main()
