# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import shutil
import sys
from typing import Any

import pytest

"""
This file contains the pytest hooks, fixtures etc. for the Arm test suite.
"""


# ==== Pytest hooks ====


def pytest_configure(config):
    pytest._test_options = {}  # type: ignore[attr-defined]
    pytest._test_options["corstone_fvp"] = False  # type: ignore[attr-defined]

    if (
        getattr(config.option, "arm_run_corstoneFVP", False)
        and config.option.arm_run_corstoneFVP
    ):
        corstone300_exists = shutil.which("FVP_Corstone_SSE-300_Ethos-U55")
        corstone320_exists = shutil.which("FVP_Corstone_SSE-320")
        if not (corstone300_exists and corstone320_exists):
            raise RuntimeError(
                "Tests are run with --arm_run_corstoneFVP but corstone FVP is not installed."
            )
        # Only enable if we also have the TOSA reference model available.
        pytest._test_options["corstone_fvp"] = True  # type: ignore[attr-defined]

    if getattr(config.option, "llama_inputs", False) and config.option.llama_inputs:
        pytest._test_options["llama_inputs"] = config.option.llama_inputs  # type: ignore[attr-defined]

    pytest._test_options["fast_fvp"] = False  # type: ignore[attr-defined]
    if getattr(config.option, "fast_fvp", False):
        pytest._test_options["fast_fvp"] = config.option.fast_fvp  # type: ignore[attr-defined]

    pytest._test_options["tosa_version"] = "0.80"  # type: ignore[attr-defined]
    if config.option.arm_run_tosa_version:
        pytest._test_options["tosa_version"] = config.option.arm_run_tosa_version

    # Not all deployments of ET have the TOSA reference model available.
    # Make sure we don't try to use it if it's not available.
    try:
        if pytest._test_options["tosa_version"] == "0.80":
            import tosa_tools.v0_80.tosa_reference_model as tosa_reference_model
        else:
            import tosa_tools.tosa_ref_model as tosa_reference_model
    except ImportError:
        pytest._test_options["tosa_ref_model"] = False  # type: ignore[attr-defined]
        tosa_reference_model = None  # noqa

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def pytest_collection_modifyitems(config, items):
    pass


def pytest_addoption(parser):
    def try_addoption(*args, **kwargs):
        try:
            parser.addoption(*args, **kwargs)
        except Exception:
            pass

    try_addoption("--arm_quantize_io", action="store_true", help="Deprecated.")
    try_addoption("--arm_run_corstoneFVP", action="store_true", help="Deprecated.")
    try_addoption("--fast_fvp", action="store_true")
    try_addoption(
        "--llama_inputs",
        nargs="+",
        help="List of two files. Firstly .pt file. Secondly .json",
    )
    try_addoption("--arm_run_tosa_version", action="store", default="0.80")


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
    import torch

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

    The optional parameter 'fail_if_not_enabled' makes the function raise
      a RuntimeError instead of returning False.
    """

    if option in pytest._test_options and pytest._test_options[option]:  # type: ignore[attr-defined]
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
    if option in pytest._test_options:  # type: ignore[attr-defined]
        return pytest._test_options[option]  # type: ignore[attr-defined]
    return None
