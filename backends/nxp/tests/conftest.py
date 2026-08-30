# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib
import shutil

from executorch.backends.nxp.tests.outputs_dir_importer import outputs_dir


def pytest_addoption(parser):
    parser.addoption(
        "--nxp_runner_path",
        action="store",
        default=None,
        help="Path to the nxp_executor_runner executable",
    )


def pytest_configure(config):
    nxp_runner_path = config.getoption("--nxp_runner_path")
    if nxp_runner_path:
        os.environ["NXP_RUNNER_PATH"] = nxp_runner_path

    os.environ["PROJECT_DIR"] = str(pathlib.Path(__file__).parent.parent.parent.parent)


# noinspection SpellCheckingInspection
def pytest_sessionstart(session):
    import executorch.extension.pybindings.portable_lib
    import executorch.kernels.quantized  # noqa F401

    # Remove all cached test files.
    #
    # Guarded because OUTPUTS_DIR is derived from the working directory, so a session started
    # somewhere unexpected would point this at a directory these tests never created. This
    # conftest is reachable from an installed package, where pytest collects it for any session
    # run under site-packages, so the guard decides whether a delete happens at all rather than
    # merely tidying up.
    #
    # Only the xdist controller clears it. Every worker also runs this hook, and they run
    # concurrently, so letting all of them delete and recreate one directory means a worker can
    # remove what another just made. The controller has no workerinput attribute; a worker does.
    if hasattr(session.config, "workerinput"):
        return

    outputs = outputs_dir.OUTPUTS_DIR
    if not _is_created_by_these_tests(outputs):
        raise RuntimeError(
            f"{outputs} exists but was not created by these tests, so it is not being "
            f"removed. Run the NXP tests from a directory that does not already contain a "
            f"'.outputs' directory, or delete it yourself if it is stale."
        )
    shutil.rmtree(outputs, ignore_errors=True)
    os.makedirs(outputs, exist_ok=True)
    (outputs / _MARKER_NAME).touch(exist_ok=True)


_MARKER_NAME = ".created-by-nxp-tests"


def _is_created_by_these_tests(directory: pathlib.Path) -> bool:
    """Whether this directory is one these tests made, or does not exist yet.

    Three ways it can be ours, in decreasing confidence:

    - the marker file, written by every run since this check was added
    - empty, which is what a run producing no artifacts leaves
    - nothing in it but directories, which is the only shape these tests write: one
      subdirectory per test name, and never a file at the top level

    The third case exists so that a directory left by a run from before the marker, or by a
    checkout that does not write one, is still recognised rather than stopping the session. A
    directory holding a file at the top level is not something these tests produce, so it is
    left alone.
    """
    if not directory.is_dir():
        return True
    if (directory / _MARKER_NAME).exists():
        return True
    entries = list(directory.iterdir())
    if not entries:
        return True
    return all(entry.is_dir() for entry in entries)
