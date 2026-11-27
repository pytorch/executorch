# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selection of the Neutron target for the NXP backend pytest suite.

One pytest run targets exactly one Neutron target, chosen with `--neutron-target=<target>` or the
`NEUTRON_TARGET` env variable. Every target belongs to an architecture, and every architecture has
a pytest marker (see `TARGET_TO_MARKER`). Tests declare the architectures they are eligible for with
these markers. Tests that are not eligible for the selected target are deselected (not skipped).

The hooks are wired up in `backends/nxp/tests/conftest.py`. The logic lives in this module, which
depends only on `pytest`, so that it can be tested in isolation (see
`generic_tests/test_target_selection.py`).
"""

import os
import re

import pytest

NEUTRON_TARGET_OPTION = "--neutron-target"
NEUTRON_TARGET_ENV_VAR = "NEUTRON_TARGET"
DEFAULT_NEUTRON_TARGET = "imxrt700"

# Maps every supported target to the marker of its architecture. Several targets may share one
# marker. Adding a target means adding one entry here. Adding a new architecture also requires its
# description in `ARCHITECTURE_MARKER_DESCRIPTIONS`.
TARGET_TO_MARKER = {
    "imxrt700": "neutron_c",
    "imx943": "neutron_s_single",
    "imx95": "neutron_s_multi",
}

ARCHITECTURE_MARKER_DESCRIPTIONS = {
    "neutron_c": "the test is eligible for Neutron-C (MCU) targets.",
    "neutron_s_single": "the test is eligible for single-core Neutron-S targets.",
    "neutron_s_multi": "the test is eligible for multi-core Neutron-S targets.",
}

ARCHITECTURE_MARKERS = frozenset(TARGET_TO_MARKER.values())

assert (
    DEFAULT_NEUTRON_TARGET in TARGET_TO_MARKER
), "The default target must be supported."
assert ARCHITECTURE_MARKERS == set(
    ARCHITECTURE_MARKER_DESCRIPTIONS
), "Every architecture marker used in `TARGET_TO_MARKER` needs exactly one description."

# Words which must not appear in a `-m` expression. The target is selected with `--neutron-target`.
_FORBIDDEN_MARK_EXPRESSION_WORDS = frozenset(TARGET_TO_MARKER) | ARCHITECTURE_MARKERS

_TARGET_FIXTURE_NAME = "target"


def _usage() -> str:
    mapping = "\n".join(
        f"    {target_name} -> {marker}"
        for target_name, marker in TARGET_TO_MARKER.items()
    )
    return (
        f"Select the Neutron target with `{NEUTRON_TARGET_OPTION}=<target>` or the "
        f"`{NEUTRON_TARGET_ENV_VAR}` env variable (the option wins). The default is "
        f"`{DEFAULT_NEUTRON_TARGET}`.\n"
        f"Supported targets and the architecture markers they select:\n{mapping}\n"
        "Tests which are not eligible for the selected target are deselected automatically. "
        f"Tests without an architecture marker run only on `{DEFAULT_NEUTRON_TARGET}`."
    )


def add_neutron_target_option(parser):
    parser.addoption(
        NEUTRON_TARGET_OPTION,
        action="store",
        # An empty env variable counts as unset.
        default=os.environ.get(NEUTRON_TARGET_ENV_VAR) or DEFAULT_NEUTRON_TARGET,
        choices=list(TARGET_TO_MARKER),
        help=f"Neutron target to run the tests for. Defaults to the `{NEUTRON_TARGET_ENV_VAR}` "
        f"env variable, or `{DEFAULT_NEUTRON_TARGET}` if it is not set.",
    )


def configure_neutron_target(config):
    for marker, description in ARCHITECTURE_MARKER_DESCRIPTIONS.items():
        config.addinivalue_line("markers", f"{marker}: {description}")

    # argparse validates `choices` only for values given on the command line, not for the default
    # taken from the env variable.
    selected_target = config.getoption(NEUTRON_TARGET_OPTION)
    if selected_target not in TARGET_TO_MARKER:
        raise pytest.UsageError(
            f"Invalid Neutron target `{selected_target}` taken from the "
            f"`{NEUTRON_TARGET_ENV_VAR}` env variable.\n{_usage()}"
        )

    _check_mark_expression(config.getoption("markexpr", default=""))


def _check_mark_expression(mark_expression: str):
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", mark_expression or ""))
    misused = sorted(words & _FORBIDDEN_MARK_EXPRESSION_WORDS)
    if misused:
        raise pytest.UsageError(
            f"The `-m {mark_expression!r}` expression refers to "
            f"{', '.join(f'`{word}`' for word in misused)}. Neutron targets and architecture "
            f"markers cannot be selected with `-m`.\n{_usage()}"
        )


@pytest.fixture
def target(request) -> str:
    """Name of the Neutron target selected for this pytest run, e.g. `imx95`."""
    return request.config.getoption(NEUTRON_TARGET_OPTION)


def _get_architecture_markers(item) -> set[str]:
    return {marker.name for marker in item.iter_markers()} & ARCHITECTURE_MARKERS


def select_tests_for_neutron_target(config, items):
    """Validate the architecture markers and deselect the tests not eligible for the target."""
    item_markers = [(item, _get_architecture_markers(item)) for item in items]

    # A test with any architecture marker must take the `target` fixture. Otherwise it would
    # silently run with the default target and report a false pass. Only unmarked tests (which run
    # only on the default target) may omit it. Checked on every run, also with the default target,
    # so the mistake is caught early.
    offenders = [
        item.nodeid
        for item, markers in item_markers
        if markers and _TARGET_FIXTURE_NAME not in getattr(item, "fixturenames", ())
    ]
    if offenders:
        raise pytest.UsageError(
            "These tests have a Neutron architecture marker, but do not take the "
            f"`{_TARGET_FIXTURE_NAME}` fixture. They would run with the default target "
            f"`{DEFAULT_NEUTRON_TARGET}` instead. Add a `{_TARGET_FIXTURE_NAME}` argument and use "
            "it in the test:\n" + "\n".join(f"    {nodeid}" for nodeid in offenders)
        )

    selected_target = config.getoption(NEUTRON_TARGET_OPTION)
    selected_marker = TARGET_TO_MARKER[selected_target]

    kept, dropped = [], []
    for item, markers in item_markers:
        if markers:
            keep = selected_marker in markers
        else:
            # Interim policy: unmarked tests are exclusive to the default target.
            keep = selected_target == DEFAULT_NEUTRON_TARGET
        (kept if keep else dropped).append(item)

    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
