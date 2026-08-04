# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import pytest
from executorch.backends.cortex_m.target_config import CortexMTargetConfig

_DEFAULT_TARGET = "cortex-m55"


def pytest_addoption(parser):
    parser.addoption(
        "--cortex-m-target",
        action="append",
        default=[],
        metavar="cortex-mXX",
        help=(
            "Cortex-M target to run the op tests against (repeatable; defaults "
            "to cortex-m55). Implementation tests additionally require a matching "
            "runner built via backends/cortex_m/test/build_test_runner.sh --target=<target>."
        ),
    )


def _selected_targets(config) -> list[str]:
    return config.getoption("--cortex-m-target") or [_DEFAULT_TARGET]


def pytest_report_header(config):
    return (
        f"cortex-m op-test targets: {', '.join(_selected_targets(config))} "
        f"{config._test_seed_label}"
    )


def pytest_generate_tests(metafunc):
    if "cortex_m_target" in metafunc.fixturenames:
        metafunc.parametrize(
            "cortex_m_target", _selected_targets(metafunc.config), indirect=True
        )


@pytest.fixture
def cortex_m_target(request) -> CortexMTargetConfig:
    """The Cortex-M target an op test runs against. Parametrized from
    ``--cortex-m-target`` so the target is explicit in the test id and selects
    the AoT target config (and, for implementation tests, the matching prebuilt
    FVP runner)."""
    return CortexMTargetConfig.from_target_string(request.param)


def pytest_configure(config):
    seed, seed_label = _setup_random_seed()
    config._test_seed = seed
    config._test_seed_label = seed_label

    if os.environ.get("TEST_RUNTIME_IS_NOT_OSS", "0") != "1":
        _set_random_seed(seed)


@pytest.fixture(autouse=True)
def set_random_seed(request):
    """Control random numbers in the Cortex-M test suite.

    By default this uses a fixed seed (0) for reproducible tests. Use
    TEST_SEED to set a custom session seed, or set it to RANDOM to choose a
    random session seed.
    """
    _set_random_seed(request.config._test_seed)


def _setup_random_seed():
    seed_env = os.environ.get("TEST_SEED", "0")
    if seed_env == "RANDOM":
        random.seed()  # reset seed, in case any other test has fiddled with it
        seed = random.randint(0, 2**32 - 1)  # nosec B311 - test seed
        seed_label = f"TEST_SEED=RANDOM using:{seed}"
    elif str.isdigit(seed_env):
        seed = int(seed_env)
        seed_label = f"TEST_SEED={seed}"
    else:
        raise TypeError("TEST_SEED env variable must be integers or the string RANDOM")

    return seed, seed_label


def _set_random_seed(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
