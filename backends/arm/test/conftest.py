# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from typing import Any

import pytest
import torch

"""
This file contains the pytest hooks, fixtures etc. for the Arm test suite.
"""


# ==== Pytest hooks ====


def pytest_configure(config):
    pytest._test_options = {}

    if config.option.arm_quantize_io:
        _load_libquantized_ops_aot_lib()
        pytest._test_options["quantize_io"] = True
    if config.option.arm_run_corstoneFVP:
        corstone300_exists = shutil.which("FVP_Corstone_SSE-300_Ethos-U55")
        corstone320_exists = shutil.which("FVP_Corstone_SSE-320")
        if not (corstone300_exists and corstone320_exists):
            raise RuntimeError(
                "Tests are run with --arm_run_corstoneFVP but corstone FVP is not installed."
            )
        pytest._test_options["corstone_fvp"] = True
    pytest._test_options["fast_fvp"] = config.option.fast_fvp
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def pytest_collection_modifyitems(config, items):
    """
    Skip all tests that require run on Ethos-U if the option arm_quantize_io is
    not set.
    """
    if not config.option.arm_quantize_io:
        skip_if_aot_lib_not_loaded = pytest.mark.skip(
            "Ethos-U tests can only run on FVP with quantize_io=True."
        )

        for item in items:
            if re.search(r"u55|u65|u85", item.name, re.IGNORECASE):
                item.add_marker(skip_if_aot_lib_not_loaded)


def pytest_addoption(parser):
    parser.addoption("--arm_quantize_io", action="store_true")
    parser.addoption("--arm_run_corstoneFVP", action="store_true")
    parser.addoption("--fast_fvp", action="store_true")


def pytest_sessionstart(session):
    pass


def pytest_sessionfinish(session, exitstatus):
    pass


# ==== End of Pytest hooks =====


# ==== Pytest fixtures =====


@pytest.fixture(autouse=True)
def set_random_seed():
    """
    Control random numbers in Arm test suite. Default behavior is random seed,
    which is set before each test. Use the env variable ARM_TEST_SEED to set the
    seed you want to use to overrride the default behavior. Or set it to RANDOM
    if you want to be explicit.

    Examples:
    As default use random seed for each test
        ARM_TEST_SEED=RANDOM pytest --config-file=/dev/null --verbose -s --color=yes  backends/arm/test/ops/test_avg_pool.py -k <TESTCASE>
    Rerun with a specific seed found under a random seed test
        ARM_TEST_SEED=3478246 pytest --config-file=/dev/null --verbose -s --color=yes  backends/arm/test/ops/test_avg_pool.py -k <TESTCASE>
    """
    if os.environ.get("ARM_TEST_SEED", "RANDOM") == "RANDOM":
        random.seed()  # reset seed, in case any other test has fiddled with it
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
    else:
        seed_str = os.environ.get("ARM_TEST_SEED", "0")
        if str.isdigit(seed_str):
            seed = int(seed_str)
            random.seed(seed)
            torch.manual_seed(seed)
        else:
            raise TypeError(
                "ARM_TEST_SEED env variable must be integers or the string RANDOM"
            )

    print(f" ARM_TEST_SEED={seed} ", end=" ")


# ==== End of Pytest fixtures =====


# ==== Custom Pytest decorators =====


def expectedFailureOnFVP(test_item):
    if is_option_enabled("corstone_fvp"):
        test_item.__unittest_expecting_failure__ = True
    return test_item


# ==== End of Custom Pytest decorators =====


def is_option_enabled(option: str, fail_if_not_enabled: bool = False) -> bool:
    """
    Returns whether an option is successfully enabled, i.e. if the flag was
    given to pytest and the necessary requirements are available.
    Implemented options are:
        - corstone_fvp.
        - quantize_io.

    The optional parameter 'fail_if_not_enabled' makes the function raise
      a RuntimeError instead of returning False.
    """

    if option in pytest._test_options and pytest._test_options[option]:
        return True
    else:
        if fail_if_not_enabled:
            raise RuntimeError(f"Required option '{option}' for test is not enabled")
        else:
            return False


def get_option(option: str) -> Any | None:
    """
    Returns the value of an pytest option if it is set, otherwise None.

    Args:
        option (str): The option to check for.
    """
    if option in pytest._test_options:
        return pytest._test_options[option]
    return None


def _load_libquantized_ops_aot_lib():
    """
    Load the libquantized_ops_aot_lib shared library. It's required when
    arm_quantize_io is set.
    """
    so_ext = {
        "Darwin": "dylib",
        "Linux": "so",
        "Windows": "dll",
    }.get(platform.system(), None)

    find_lib_cmd = [
        "find",
        "cmake-out-aot-lib",
        "-name",
        f"libquantized_ops_aot_lib.{so_ext}",
    ]

    res = subprocess.run(find_lib_cmd, capture_output=True)
    if res.returncode == 0:
        library_path = res.stdout.decode().strip()
        torch.ops.load_library(library_path)
    else:
        raise RuntimeError(
            f"Failed to load libquantized_ops_aot_lib.{so_ext}. Did you build it?"
        )
