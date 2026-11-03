# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "install_utils", ROOT / "install_utils.py"
)
assert SPEC is not None and SPEC.loader is not None
INSTALL_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL_UTILS)


class TestPackageInstallCommand(unittest.TestCase):
    def setUp(self):
        INSTALL_UTILS.get_package_install_command.cache_clear()
        self.addCleanup(INSTALL_UTILS.get_package_install_command.cache_clear)

    def test_prefers_pip_when_available(self):
        for uv in (None, "/test/bin/uv"):
            with self.subTest(uv=uv):
                INSTALL_UTILS.get_package_install_command.cache_clear()
                with (
                    patch.object(
                        INSTALL_UTILS.importlib.util, "find_spec", return_value=object()
                    ) as find_spec,
                    patch.object(INSTALL_UTILS.shutil, "which", return_value=uv),
                ):
                    self.assertEqual(
                        INSTALL_UTILS.get_package_install_command(),
                        (sys.executable, "-m", "pip", "install"),
                    )
                    find_spec.assert_called_once_with("pip")

    def test_uv_targets_running_python(self):
        with (
            patch.object(INSTALL_UTILS.importlib.util, "find_spec", return_value=None),
            patch.object(
                INSTALL_UTILS.shutil, "which", return_value="/test/bin/uv"
            ) as which,
        ):
            self.assertEqual(
                INSTALL_UTILS.get_package_install_command(),
                (
                    "/test/bin/uv",
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--index-strategy",
                    "unsafe-best-match",
                ),
            )
            which.assert_called_once_with("uv")

    def test_missing_installers_reports_setup_instructions(self):
        with (
            patch.object(INSTALL_UTILS.importlib.util, "find_spec", return_value=None),
            patch.object(INSTALL_UTILS.shutil, "which", return_value=None),
            self.assertRaisesRegex(
                RuntimeError, "Neither pip nor uv is available"
            ) as error,
        ):
            INSTALL_UTILS.get_package_install_command()
        message = str(error.exception)
        self.assertIn(f"{sys.executable} -m ensurepip --upgrade", message)
        self.assertIn(
            "https://docs.astral.sh/uv/getting-started/installation/", message
        )

    def test_keeps_uv_after_pip_becomes_available(self):
        with (
            patch.object(
                INSTALL_UTILS.importlib.util, "find_spec", return_value=None
            ) as find_spec,
            patch.object(
                INSTALL_UTILS.shutil, "which", return_value="/test/bin/uv"
            ) as which,
        ):
            command = INSTALL_UTILS.get_package_install_command()
            self.assertEqual(command[0], "/test/bin/uv")
            find_spec.return_value = object()
            self.assertEqual(INSTALL_UTILS.get_package_install_command(), command)
            find_spec.assert_called_once_with("pip")
            which.assert_called_once_with("uv")


if __name__ == "__main__":
    unittest.main()
