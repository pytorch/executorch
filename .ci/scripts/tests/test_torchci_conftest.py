# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
PLUGIN = "torch.testing._internal.torchci.plugin"
REPORT = "torch.testing._internal.torchci.report"
OPTED = {
    "RUNNER_TEST_RUN_REPORTS_DIR": "/tmp/torchci reports",
    "JOB_ID": "51234567890",
    "GITHUB_REPOSITORY": "pytorch/executorch",
}
WARNING = "::warning::torchci: test-run reports disabled"


def stub_modules(pytest_version="7.3.0", schema="1.0", reporter=True):
    pytest = ModuleType("pytest")
    pytest.__version__ = pytest_version

    def hookimpl(**options):
        def mark(hook):
            hook.hookimpl_options = options
            return hook

        return mark

    pytest.hookimpl = hookimpl
    modules = {"pytest": pytest, "torch": ModuleType("torch")}
    # None blocks an import even where the real reporter is installed.
    modules[PLUGIN] = ModuleType(PLUGIN) if reporter else None
    modules[REPORT] = SimpleNamespace(SCHEMA_VERSION=schema) if reporter else None
    return modules


class FakeConfig:
    def __init__(self, prefix=None, import_error=None, terminal=True):
        self.option = SimpleNamespace(torchci_report_prefix=prefix)
        self.imported = []
        self.lines = []
        self.import_error = import_error
        self.terminal = SimpleNamespace(write_line=self.lines.append)
        self.pluginmanager = SimpleNamespace(
            import_plugin=self.import_plugin,
            get_plugin=lambda name: self.terminal if terminal else None,
        )

    def getoption(self, name, default=None):
        return getattr(self.option, name, default)

    def import_plugin(self, name):
        if self.import_error:
            raise self.import_error
        self.imported.append(name)


def configure(env, config=None, **stubs):
    """Run the root conftest.py's pytest_configure with stub torch and pytest."""
    config = config or FakeConfig()
    spec = importlib.util.spec_from_file_location(
        "executorch_root_conftest", ROOT / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(sys.modules, stub_modules(**stubs)))
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        spec.loader.exec_module(module)
        module.pytest_configure(config)
    return module, config


class TorchciConftestTest(unittest.TestCase):
    def test_conftest_does_not_define_pytest_plugins(self):
        module, _ = configure(OPTED)
        self.assertFalse(hasattr(module, "pytest_plugins"))

    def test_configure_runs_after_the_terminal_reporter_is_registered(self):
        module, _ = configure({})
        self.assertEqual(module.pytest_configure.hookimpl_options, {"trylast": True})

    def test_a_missing_terminal_reporter_cannot_fail_the_run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _, config = configure({**OPTED, "JOB_ID": "0"}, FakeConfig(terminal=False))
        self.assertEqual(config.imported, [])
        self.assertIn(WARNING, out.getvalue())
        self.assertIn("positive JOB_ID", out.getvalue())

    def test_unopted_runs_change_nothing(self):
        for env in ({}, {"RUNNER_TEST_RUN_REPORTS_DIR": ""}):
            with self.subTest(env=env):
                _, config = configure(env, reporter=False)
                self.assertEqual(config.imported, [])
                self.assertEqual(config.lines, [])
                self.assertIsNone(config.option.torchci_report_prefix)

    def test_opted_runs_register_the_reporter_with_a_prefix(self):
        _, config = configure(OPTED)
        self.assertEqual(config.imported, [PLUGIN])
        self.assertEqual(config.lines, [])
        reports = OPTED["RUNNER_TEST_RUN_REPORTS_DIR"]
        self.assertEqual(
            config.option.torchci_report_prefix, os.path.join(reports, "pytest")
        )
        _, config = configure({**OPTED, "TORCHCI_REPORT_NAME": "unittest"})
        self.assertEqual(
            config.option.torchci_report_prefix, os.path.join(reports, "unittest")
        )

    def test_an_explicit_prefix_is_kept(self):
        _, config = configure(OPTED, FakeConfig(prefix="/explicit/p"))
        self.assertEqual(config.imported, [PLUGIN])
        self.assertEqual(config.option.torchci_report_prefix, "/explicit/p")

    def test_a_failed_preflight_warns_and_registers_nothing(self):
        cases = (
            ({}, {"reporter": False}, "torch"),
            ({}, {"schema": "2.0"}, "schema 1.0"),
            ({}, {"pytest_version": "7.2.0"}, "pytest >= 7.3"),
            ({"JOB_ID": ""}, {}, "positive JOB_ID"),
            ({"JOB_ID": "0"}, {}, "positive JOB_ID"),
            ({"JOB_ID": "-1"}, {}, "positive JOB_ID"),
            ({"JOB_ID": "abc"}, {}, "positive JOB_ID"),
            ({"GITHUB_REPOSITORY": ""}, {}, "GITHUB_REPOSITORY"),
        )
        for extra, stubs, message in cases:
            with self.subTest(extra=extra, stubs=stubs):
                _, config = configure({**OPTED, **extra}, **stubs)
                self.assertEqual(config.imported, [])
                self.assertEqual(len(config.lines), 1)
                self.assertTrue(config.lines[0].startswith(WARNING))
                self.assertIn(message, config.lines[0])

    def test_a_failing_terminal_reporter_cannot_fail_the_run(self):
        config = FakeConfig()
        config.terminal.write_line = mock.Mock(side_effect=OSError("closed"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            configure({**OPTED, "JOB_ID": "0"}, config)
        self.assertIn(WARNING, out.getvalue())

    def test_a_failed_registration_warns_instead_of_raising(self):
        config = FakeConfig(import_error=ImportError("plugin import failed"))
        _, config = configure(OPTED, config)
        self.assertEqual(len(config.lines), 1)
        self.assertIn("plugin import failed", config.lines[0])

    def test_xdist_workers_register_without_repeating_the_warning(self):
        _, config = configure({**OPTED, "PYTEST_XDIST_WORKER": "gw3"})
        self.assertEqual(config.imported, [PLUGIN])
        _, config = configure({**OPTED, "JOB_ID": "0", "PYTEST_XDIST_WORKER": "gw3"})
        self.assertEqual(config.imported, [])
        self.assertEqual(config.lines, [])


if __name__ == "__main__":
    unittest.main()
