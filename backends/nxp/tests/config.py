# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import shutil

# The PROJECT_DIR env variable is set by the conftest.py in backends.nxp.tests_models.conftest.
# It is supposed to point at ExecuTorch Project directory (not install folder) to derive path to artefacts (config files,
# dataset, model weight) located in the project directory structure, but not installed.
PROJECT_DIR = os.environ.get("PROJECT_DIR")
if not PROJECT_DIR:
    # Auto-detect: The PROJECT_DIR env variable is set by the conftest.py in backends.nxp.tests.conftest. But unittests
    #               don't use the conftest, so this variable is not set -> set it manually here in that case.
    PROJECT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
assert os.path.exists(
    PROJECT_DIR
), f"Invalid PROJECT_DIR env variable: `{PROJECT_DIR}`."

OUTPUTS_DIR = pathlib.Path(os.getcwd()) / ".outputs"


# The four values below need the profiler and the Neutron SDK, which only the machine running the
# hardware tests has. Resolved on call rather than at import, so importing this module does not
# require them: the wheel ships this file but not the SDK, so a module level lookup made every
# importer of the surrounding test helpers fail.
def nsys_path() -> pathlib.Path:
    found = shutil.which("nsys")
    assert found, "nsys not found on PATH."
    return pathlib.Path(found)


def nsys_config_path() -> str:
    return os.path.join(PROJECT_DIR, "backends", "nxp", "tests", "neutron-imxrt700.ini")


def nsys_firmware_path() -> str:
    import eiq_neutron_sdk

    return os.path.join(
        os.path.dirname(eiq_neutron_sdk.__file__),
        "target",
        "imxrt700",
        "cmodel",
        "NeutronFirmware.elf",
    )


def neutron_test_path() -> pathlib.Path:
    # The NXP_RUNNER_PATH env variable is either defined by pytest when using the CLI argument
    # --nxp_executor_path or a standard environment variable.
    from_env = os.environ.get("NXP_RUNNER_PATH")
    path = (
        pathlib.Path(from_env)
        if from_env
        else pathlib.Path(PROJECT_DIR)
        / "examples"
        / "nxp"
        / "executor_runner"
        / "build"
        / "nxp_executor_runner"
    )
    assert os.path.exists(path), f"Invalid NXP_RUNNER_PATH env variable: `{path}`."
    return path
