# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    return f"cortex-m op-test targets: {', '.join(_selected_targets(config))}"


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
