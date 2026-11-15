# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import call, Mock, patch

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

    def test_entrypoints_bootstrap_pip_before_python_check(self):
        for script in ("install_executorch.py", "install_requirements.py"):
            with self.subTest(script=script):
                path = str(ROOT / script)
                steps = Mock()
                steps.python_is_compatible.return_value = False
                with (
                    patch.dict(sys.modules, {"install_utils": INSTALL_UTILS}),
                    patch.object(sys, "path", [str(ROOT), *sys.path]),
                    patch.object(sys, "argv", [path]),
                    patch.object(os, "chdir"),
                    patch.object(INSTALL_UTILS, "ensure_pip", steps.ensure_pip),
                    patch.object(
                        INSTALL_UTILS,
                        "python_is_compatible",
                        steps.python_is_compatible,
                    ),
                    self.assertRaises(SystemExit),
                ):
                    runpy.run_path(path, run_name="__main__")
                self.assertEqual(
                    steps.mock_calls, [call.ensure_pip(), call.python_is_compatible()]
                )


if __name__ == "__main__":
    unittest.main()
