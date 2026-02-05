# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib
import shutil
from executorch.backends.nxp.tests_models.outputs_dir_importer import outputs_dir

def pytest_addoption(parser):
    parser.addoption(
        "--nxp_runner_path",
        action="store",
        default=None,
        help="Path to the nxp_executor_runner executable"
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

    # Remove all cached test files
    shutil.rmtree(outputs_dir.OUTPUTS_DIR, ignore_errors=True)
    os.mkdir(outputs_dir.OUTPUTS_DIR)





