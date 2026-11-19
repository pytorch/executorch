import hashlib
import importlib
import os
import platform
import sys

import pytest
import torch

collect_ignore_glob: list[str] = []

# Skip Apple tests on Windows. Note that some Core ML tests can run on Linux, as the AOT flow
# is available. Tests will manage this internally. However, the coremltools import is not available
# on Windows and causes collection to fail. The easiest way to manage this seems to be to just
# skip collection for this subdirectory on unsupported platforms.
if sys.platform == "win32":
    collect_ignore_glob += [
        "backends/apple/**",
    ]

# Every test file under backends/apple/coreml that defines tests imports coremltools at module
# scope, so collection fails wherever the wheel does not declare it. Windows is already covered
# above; what is left is any Linux that is not x86_64, since coremltools publishes no build for
# those, and Python 3.14, for which it publishes no build on any platform. Keep this condition in
# sync with the coremltools marker in setup.py; .ci/scripts/tests/test_coreml_markers.py checks
# that the two agree.
_coremltools_is_declared = (
    sys.platform == "darwin"
    or (sys.platform.startswith("linux") and platform.machine() == "x86_64")
) and sys.version_info < (3, 14)

if not _coremltools_is_declared:
    collect_ignore_glob += [
        "backends/apple/coreml/**",
    ]


_TORCHCI_PLUGIN = "torch.testing._internal.torchci.plugin"


def _torchci_preflight(env):
    """Why PyTorch's test-run reporter cannot run in this CI job, or None.

    linux_job_v3 sets RUNNER_TEST_RUN_REPORTS_DIR only for jobs that opt in.
    """
    try:
        importlib.import_module(_TORCHCI_PLUGIN)
        report = importlib.import_module("torch.testing._internal.torchci.report")
        if report.SCHEMA_VERSION != "1.0":
            raise RuntimeError("requires report schema 1.0")
        if tuple(map(int, pytest.__version__.split(".")[:2])) < (7, 3):
            raise RuntimeError("requires pytest >= 7.3")
        job_id = env.get("JOB_ID", "")
        if not job_id.isdigit() or int(job_id) <= 0:
            raise RuntimeError("requires a positive JOB_ID")
        if not env.get("GITHUB_REPOSITORY"):
            raise RuntimeError("requires GITHUB_REPOSITORY")
    except Exception as error:
        return error
    return None


def _torchci_warn(config, error):
    line = f"::warning::torchci: test-run reports disabled: {error}"
    try:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(line)
            return
    except Exception:
        pass  # Fall back to stdout below; a warning must not fail the run.
    print(line)


# trylast: pytest's terminal reporter registers in its own pytest_configure.
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # Registered here rather than with pytest_plugins, which pytest rejects in a
    # conftest.py it loads after startup. Reporting is best-effort, as in
    # PyTorch: any failure only disables the report and never fails the tests.
    env = os.environ
    if not env.get("RUNNER_TEST_RUN_REPORTS_DIR"):
        return
    error = _torchci_preflight(env)
    try:
        if error is None:
            if not config.getoption("torchci_report_prefix", None):
                config.option.torchci_report_prefix = os.path.join(
                    env["RUNNER_TEST_RUN_REPORTS_DIR"],
                    env.get("TORCHCI_REPORT_NAME", "pytest"),
                )
            config.pluginmanager.import_plugin(_TORCHCI_PLUGIN)
        elif not env.get("PYTEST_XDIST_WORKER"):
            _torchci_warn(config, error)
    except Exception as e:
        if not env.get("PYTEST_XDIST_WORKER"):
            _torchci_warn(config, e)


def pytest_runtest_setup(item):
    # Set a stable seed for each test based on a hash of the test name.
    seed = int(hashlib.sha256(item.nodeid.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
