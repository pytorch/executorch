# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import shutil

import eiq_neutron_sdk

# The PROJECT_DIR env variable is set by the conftest.py in backends.nxp.tests_models.conftest.
# It is supposed to point at ExecuTorch Project directory (not install folder) to derive path to artefacts (config files,
# dataset, model weight) located in the project directory structure, but not installed.
# TODO(Robert Kalmar) In accordance with the "TODO(dbort): Prune /test[s]/ dirs, /third-party/ dirs" in pyproject.toml,
#  once the test folders are not installed we can derive the path from current file location: `pathlib.Path(__file__)`
PROJECT_DIR = os.environ.get("PROJECT_DIR")
assert PROJECT_DIR and os.path.exists(PROJECT_DIR)

OUTPUTS_DIR = pathlib.Path(os.getcwd()) / ".outputs"

NSYS_PATH = pathlib.Path(shutil.which("nsys"))
NSYS_CONFIG_PATH = os.path.join(
    PROJECT_DIR, "backends", "nxp", "tests_models", "neutron-imxrt700.ini"
)
NSYS_FIRMWARE_PATH = os.path.join(
    os.path.dirname(eiq_neutron_sdk.__file__),
    "target",
    "imxrt700",
    "cmodel",
    "NeutronFirmware.elf",
)

# The NXP_RUNNER_PATH env variable is either defined by pytest when using the CLI argument --nxp_executor_path or
# a standard environment variable.
NEUTRON_TEST_PATH = os.environ.get("NXP_RUNNER_PATH")
assert NEUTRON_TEST_PATH and os.path.exists(NEUTRON_TEST_PATH)
