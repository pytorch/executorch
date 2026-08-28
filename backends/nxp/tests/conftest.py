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
    # conftest is reachable from an installed package, where pytest collects it for any
    # session run under site-packages, so the guard decides whether a delete happens at all
    # rather than merely tidying up.
    outputs = outputs_dir.OUTPUTS_DIR
    if outputs.is_dir() and not _is_created_by_these_tests(outputs):
        raise RuntimeError(
            f"{outputs} exists but was not created by these tests, so it is not being "
            f"removed. Run the NXP tests from a directory that does not already contain a "
            f"'.outputs' directory, or delete it yourself if it is stale."
        )
    shutil.rmtree(outputs, ignore_errors=True)
    os.makedirs(outputs, exist_ok=True)
    (outputs / _MARKER_NAME).touch()


_MARKER_NAME = ".created-by-nxp-tests"


def _is_created_by_these_tests(directory: pathlib.Path) -> bool:
    """Whether this directory is one a previous run of these tests made.

    An empty directory counts, since that is what a run that produced no artifacts leaves.
    """
    if (directory / _MARKER_NAME).exists():
        return True
    return not any(directory.iterdir())
