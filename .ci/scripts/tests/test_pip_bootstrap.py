# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import call, mock_open, patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "install_utils", ROOT / "install_utils.py"
)
assert SPEC is not None and SPEC.loader is not None
INSTALL_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL_UTILS)


class TestPipBootstrap(unittest.TestCase):
    def test_existing_pip_skips_bootstrap(self):
        with (
            patch.object(
                INSTALL_UTILS.importlib.util, "find_spec", return_value=object()
            ) as find_spec,
            patch.object(INSTALL_UTILS.shutil, "which") as which,
            patch.object(INSTALL_UTILS.subprocess, "run") as run,
        ):
            INSTALL_UTILS.ensure_pip()
            find_spec.assert_called_once_with("pip")
            which.assert_not_called()
            run.assert_not_called()

    def test_uv_installs_pip_for_running_python(self):
        with (
            patch.object(INSTALL_UTILS.importlib.util, "find_spec", return_value=None),
            patch.object(
                INSTALL_UTILS.shutil, "which", return_value="/test/bin/uv"
            ) as which,
            patch.object(INSTALL_UTILS.subprocess, "run") as run,
        ):
            INSTALL_UTILS.ensure_pip()
            which.assert_called_once_with("uv")
            run.assert_called_once_with(
                ["/test/bin/uv", "pip", "install", "--python", sys.executable, "pip"],
                check=True,
            )

    def test_missing_installers_reports_setup_instructions(self):
        with (
            patch.object(INSTALL_UTILS.importlib.util, "find_spec", return_value=None),
            patch.object(INSTALL_UTILS.shutil, "which", return_value=None),
            patch.object(INSTALL_UTILS.subprocess, "run") as run,
            self.assertRaisesRegex(
                RuntimeError, "Neither pip nor uv is available"
            ) as error,
        ):
            INSTALL_UTILS.ensure_pip()
        run.assert_not_called()
        message = str(error.exception)
        self.assertIn(f"{sys.executable} -m ensurepip --upgrade", message)
        self.assertIn(
            "https://docs.astral.sh/uv/getting-started/installation/", message
        )

    def test_uv_failure_is_propagated(self):
        with (
            patch.object(INSTALL_UTILS.importlib.util, "find_spec", return_value=None),
            patch.object(INSTALL_UTILS.shutil, "which", return_value="/test/bin/uv"),
            patch.object(
                INSTALL_UTILS.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "uv"),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            INSTALL_UTILS.ensure_pip()

    def test_bootstrap_is_skipped_after_pip_is_installed(self):
        with (
            patch.object(
                INSTALL_UTILS.importlib.util,
                "find_spec",
                side_effect=[None, object()],
            ),
            patch.object(INSTALL_UTILS.shutil, "which", return_value="/test/bin/uv"),
            patch.object(INSTALL_UTILS.subprocess, "run") as run,
        ):
            INSTALL_UTILS.ensure_pip()
            INSTALL_UTILS.ensure_pip()
            run.assert_called_once()

    def test_bootstraps_pip_before_packaging(self):
        original_import = builtins.__import__

        def import_without_packaging(name, *args, **kwargs):
            if name == "packaging":
                raise ModuleNotFoundError("No module named 'packaging'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=import_without_packaging),
            patch(
                "builtins.open",
                mock_open(read_data='requires-python = ">=3.11,<3.16"\n'),
            ),
            patch.object(
                INSTALL_UTILS.platform, "python_version", return_value="3.12.0"
            ),
            patch.object(INSTALL_UTILS.importlib.util, "find_spec", return_value=None),
            patch.object(INSTALL_UTILS.shutil, "which", return_value="/test/bin/uv"),
            patch.object(INSTALL_UTILS.subprocess, "run") as run,
        ):
            self.assertTrue(INSTALL_UTILS.python_is_compatible())
            self.assertEqual(
                run.call_args_list,
                [
                    call(
                        [
                            "/test/bin/uv",
                            "pip",
                            "install",
                            "--python",
                            sys.executable,
                            "pip",
                        ],
                        check=True,
                    ),
                    call(
                        [sys.executable, "-m", "pip", "install", "packaging"],
                        check=True,
                    ),
                ],
            )

    def test_bootstraps_pip_when_packaging_is_already_installed(self):
        with (
            patch(
                "builtins.open",
                mock_open(read_data='requires-python = ">=3.11,<3.16"\n'),
            ),
            patch.object(
                INSTALL_UTILS.platform, "python_version", return_value="3.12.0"
            ),
            patch.object(INSTALL_UTILS.importlib.util, "find_spec", return_value=None),
            patch.object(INSTALL_UTILS.shutil, "which", return_value="/test/bin/uv"),
            patch.object(INSTALL_UTILS.subprocess, "run") as run,
        ):
            self.assertTrue(INSTALL_UTILS.python_is_compatible())
            run.assert_called_once_with(
                ["/test/bin/uv", "pip", "install", "--python", sys.executable, "pip"],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
