# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the Neutron target selection logic in `backends/nxp/tests/target_selection.py`.

Each test runs a separate pytest session in a temporary directory, with a small conftest that wires up the hooks
exactly like `backends/nxp/tests/conftest.py` does. The `target_selection` module is loaded from its file, so these
sessions do not need the rest of the NXP test infrastructure.
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

_TARGET_SELECTION_PATH = pathlib.Path(__file__).parent.parent / "target_selection.py"

_CONFTEST = textwrap.dedent(
    """
    import importlib.util
    import os

    _spec = importlib.util.spec_from_file_location(
        "_target_selection_under_test", os.environ["TARGET_SELECTION_PATH"]
    )
    _target_selection = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_target_selection)

    target = _target_selection.target


    def pytest_addoption(parser):
        _target_selection.add_neutron_target_option(parser)


    def pytest_configure(config):
        _target_selection.configure_neutron_target(config)


    def pytest_collection_modifyitems(config, items):
        _target_selection.select_tests_for_neutron_target(config, items)
    """
)

_TEST_MODULE = textwrap.dedent(
    """
    import os

    import pytest


    def test_unmarked():
        pass


    @pytest.mark.neutron_c
    def test_neutron_c_only(target):
        pass


    @pytest.mark.neutron_s_single
    def test_neutron_s_single_only(target):
        pass


    @pytest.mark.neutron_s_multi
    def test_neutron_s_multi_only(target):
        pass


    @pytest.mark.neutron_c
    @pytest.mark.neutron_s_single
    @pytest.mark.neutron_s_multi
    def test_all_architectures(target):
        assert target == os.environ["EXPECTED_TARGET"]
    """
)


def _run_pytest(
    tmp_path: pathlib.Path,
    *args: str,
    test_module: str = _TEST_MODULE,
    env_target: str | None = None,
    expected_target: str = "imxrt700",
) -> subprocess.CompletedProcess:
    (tmp_path / "conftest.py").write_text(_CONFTEST)
    (tmp_path / "test_module.py").write_text(test_module)

    env = os.environ.copy()
    # Isolate the inner session from the settings of the outer one.
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("NEUTRON_TARGET", None)
    if env_target is not None:
        env["NEUTRON_TARGET"] = env_target
    env["TARGET_SELECTION_PATH"] = str(_TARGET_SELECTION_PATH)
    env["EXPECTED_TARGET"] = expected_target

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            os.devnull,
            "--rootdir",
            str(tmp_path),
            "-p",
            "no:cacheprovider",
            "-v",
            *args,
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _passed_tests(result: subprocess.CompletedProcess) -> set[str]:
    return set(re.findall(r"::(\w+) PASSED", result.stdout))


def _output(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


def test_default_target_runs_unmarked_and_neutron_c_tests(tmp_path):
    result = _run_pytest(tmp_path, expected_target="imxrt700")

    assert result.returncode == 0, _output(result)
    assert _passed_tests(result) == {
        "test_unmarked",
        "test_neutron_c_only",
        "test_all_architectures",
    }


@pytest.mark.parametrize(
    "selected_target, expected_tests",
    [
        (
            "imxrt700",
            {"test_unmarked", "test_neutron_c_only", "test_all_architectures"},
        ),
        ("imx943", {"test_neutron_s_single_only", "test_all_architectures"}),
        ("imx95", {"test_neutron_s_multi_only", "test_all_architectures"}),
    ],
)
def test_cli_option_selects_target(tmp_path, selected_target, expected_tests):
    result = _run_pytest(
        tmp_path, f"--neutron-target={selected_target}", expected_target=selected_target
    )

    assert result.returncode == 0, _output(result)
    assert _passed_tests(result) == expected_tests


def test_env_variable_selects_target(tmp_path):
    result = _run_pytest(tmp_path, env_target="imx95", expected_target="imx95")

    assert result.returncode == 0, _output(result)
    assert _passed_tests(result) == {
        "test_neutron_s_multi_only",
        "test_all_architectures",
    }


def test_empty_env_variable_means_default_target(tmp_path):
    result = _run_pytest(tmp_path, env_target="", expected_target="imxrt700")

    assert result.returncode == 0, _output(result)
    assert "test_unmarked" in _passed_tests(result)


def test_cli_option_wins_over_env_variable(tmp_path):
    result = _run_pytest(
        tmp_path,
        "--neutron-target=imx943",
        env_target="imx95",
        expected_target="imx943",
    )

    assert result.returncode == 0, _output(result)
    assert _passed_tests(result) == {
        "test_neutron_s_single_only",
        "test_all_architectures",
    }


def test_invalid_cli_target_fails(tmp_path):
    result = _run_pytest(tmp_path, "--neutron-target=imx9")

    assert result.returncode != 0
    assert "invalid choice: 'imx9'" in _output(result)


def test_invalid_env_target_fails(tmp_path):
    result = _run_pytest(tmp_path, env_target="imx9")

    assert result.returncode != 0
    assert "Invalid Neutron target `imx9`" in _output(result)


def test_ineligible_tests_are_deselected_not_skipped(tmp_path):
    result = _run_pytest(tmp_path, "--neutron-target=imx95", expected_target="imx95")

    assert result.returncode == 0, _output(result)
    summary = result.stdout.strip().splitlines()[-1]
    assert "2 passed" in summary
    assert "3 deselected" in summary
    assert "skipped" not in summary


@pytest.mark.parametrize(
    "mark_expression",
    ["neutron_c", "not neutron_s_multi", "imx95", "slow and imxrt700"],
)
def test_neutron_target_or_marker_in_mark_expression_fails(tmp_path, mark_expression):
    result = _run_pytest(tmp_path, "-m", mark_expression)

    assert result.returncode != 0
    output = _output(result)
    assert "cannot be selected with `-m`" in output
    assert "--neutron-target=<target>" in output


def test_unrelated_mark_expression_still_works(tmp_path):
    result = _run_pytest(tmp_path, "-m", "not slow")

    assert result.returncode == 0, _output(result)
    assert _passed_tests(result) == {
        "test_unmarked",
        "test_neutron_c_only",
        "test_all_architectures",
    }


def test_marked_test_without_target_fixture_fails_on_default_target(tmp_path):
    test_module = textwrap.dedent(
        """
        import pytest


        @pytest.mark.neutron_c
        @pytest.mark.neutron_s_multi
        def test_forgets_target():
            pass


        @pytest.mark.neutron_c
        def test_neutron_c_forgets_target():
            pass


        def test_unmarked_without_target_is_fine():
            pass
        """
    )
    result = _run_pytest(tmp_path, test_module=test_module)

    assert result.returncode != 0
    output = _output(result)
    assert "do not take the `target` fixture" in output
    # Every test with an architecture marker must take the `target` fixture, including tests
    # marked only for the default architecture.
    assert "test_module.py::test_forgets_target" in output
    assert "test_module.py::test_neutron_c_forgets_target" in output
    # Only unmarked tests may omit the `target` fixture.
    assert "test_unmarked_without_target_is_fine" not in output


def test_markers_are_registered_for_strict_markers(tmp_path):
    result = _run_pytest(
        tmp_path,
        "--strict-markers",
        "--neutron-target=imx943",
        expected_target="imx943",
    )

    assert result.returncode == 0, _output(result)


def test_unregistered_marker_fails_with_strict_markers(tmp_path):
    test_module = textwrap.dedent(
        """
        import pytest


        @pytest.mark.neutron_s_mutli
        def test_with_typo(target):
            pass
        """
    )
    result = _run_pytest(tmp_path, "--strict-markers", test_module=test_module)

    assert result.returncode != 0
    assert "'neutron_s_mutli' not found in `markers` configuration option" in _output(
        result
    )
