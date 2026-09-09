# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import executorch.backends.arm.test.conftest as arm_conftest

import pytest
from executorch.backends.arm.test import common


def _report(nodeid, outcome, when, **attributes):
    return SimpleNamespace(nodeid=nodeid, outcome=outcome, when=when, **attributes)


def test_expected_xfail_marks_existing_artifact_directory(
    monkeypatch, tmp_path
) -> None:
    nodeid = "test_artifacts.py::test_case[1, 2]"
    artifact_dir = tmp_path / "test_case[1_2]"
    artifact_dir.mkdir()
    monkeypatch.setattr(pytest, "_test_options", {"dump_artifacts": str(tmp_path)})
    getattr(arm_conftest, "_expected_xfail_nodeids", set()).clear()

    arm_conftest.pytest_runtest_logreport(
        _report(nodeid, "skipped", "call", wasxfail="expected")
    )
    arm_conftest.pytest_runtest_logreport(_report(nodeid, "passed", "teardown"))

    assert (artifact_dir / "_xfailed_test").is_file()


def test_setup_expected_xfail_marks_existing_artifact_directory(
    monkeypatch, tmp_path
) -> None:
    nodeid = "test_artifacts.py::test_case[1, 2]"
    artifact_dir = tmp_path / "test_case[1_2]"
    artifact_dir.mkdir()
    monkeypatch.setattr(pytest, "_test_options", {"dump_artifacts": str(tmp_path)})
    getattr(arm_conftest, "_expected_xfail_nodeids", set()).clear()

    arm_conftest.pytest_runtest_logreport(
        _report(nodeid, "skipped", "setup", wasxfail="expected")
    )
    arm_conftest.pytest_runtest_logreport(_report(nodeid, "passed", "teardown"))

    assert (artifact_dir / "_xfailed_test").is_file()


def test_setup_notrun_xfail_does_not_mark(monkeypatch, tmp_path) -> None:
    nodeid = "test_artifacts.py::test_case"
    artifact_dir = tmp_path / "test_case"
    artifact_dir.mkdir()
    monkeypatch.setattr(pytest, "_test_options", {"dump_artifacts": str(tmp_path)})
    getattr(arm_conftest, "_expected_xfail_nodeids", set()).clear()

    arm_conftest.pytest_runtest_logreport(
        _report(nodeid, "skipped", "setup", wasxfail="[NOTRUN] expected")
    )
    arm_conftest.pytest_runtest_logreport(_report(nodeid, "passed", "teardown"))

    assert not (artifact_dir / "_xfailed_test").exists()


def test_expected_xfail_with_failed_teardown_does_not_mark(
    monkeypatch, tmp_path
) -> None:
    nodeid = "test_artifacts.py::test_case"
    artifact_dir = tmp_path / "test_case"
    artifact_dir.mkdir()
    monkeypatch.setattr(pytest, "_test_options", {"dump_artifacts": str(tmp_path)})
    getattr(arm_conftest, "_expected_xfail_nodeids", set()).clear()

    arm_conftest.pytest_runtest_logreport(
        _report(nodeid, "skipped", "call", wasxfail="expected")
    )
    arm_conftest.pytest_runtest_logreport(_report(nodeid, "failed", "teardown"))

    assert not (artifact_dir / "_xfailed_test").exists()


def test_xpass_does_not_mark(monkeypatch, tmp_path) -> None:
    nodeid = "test_artifacts.py::test_case"
    artifact_dir = tmp_path / "test_case"
    artifact_dir.mkdir()
    monkeypatch.setattr(pytest, "_test_options", {"dump_artifacts": str(tmp_path)})
    getattr(arm_conftest, "_expected_xfail_nodeids", set()).clear()

    arm_conftest.pytest_runtest_logreport(
        _report(nodeid, "passed", "call", wasxfail="expected")
    )
    arm_conftest.pytest_runtest_logreport(_report(nodeid, "passed", "teardown"))

    assert not (artifact_dir / "_xfailed_test").exists()


@pytest.mark.parametrize(
    "outcome, attributes", [("passed", {}), ("failed", {}), ("skipped", {})]
)
def test_ordinary_outcomes_do_not_mark(
    monkeypatch, tmp_path, outcome, attributes
) -> None:
    nodeid = "test_artifacts.py::test_case"
    artifact_dir = tmp_path / "test_case"
    artifact_dir.mkdir()
    monkeypatch.setattr(pytest, "_test_options", {"dump_artifacts": str(tmp_path)})
    getattr(arm_conftest, "_expected_xfail_nodeids", set()).clear()

    arm_conftest.pytest_runtest_logreport(
        _report(nodeid, outcome, "call", **attributes)
    )
    arm_conftest.pytest_runtest_logreport(_report(nodeid, "passed", "teardown"))

    assert not (artifact_dir / "_xfailed_test").exists()


def test_expected_xfail_does_not_create_missing_artifact_directory(
    monkeypatch, tmp_path
) -> None:
    nodeid = "test_artifacts.py::test_case"
    monkeypatch.setattr(pytest, "_test_options", {"dump_artifacts": str(tmp_path)})
    getattr(arm_conftest, "_expected_xfail_nodeids", set()).clear()

    arm_conftest.pytest_runtest_logreport(
        _report(nodeid, "skipped", "call", wasxfail="expected")
    )
    arm_conftest.pytest_runtest_logreport(_report(nodeid, "passed", "teardown"))

    assert not (tmp_path / "test_case").exists()


def test_dump_artifacts_uses_test_name(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        pytest,
        "_test_options",
        {"dump_artifacts": str(tmp_path)},
        raising=False,
    )
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "backends/arm/test/ops/test_add.py::test_add_tosa_INT[shape] (call)",
    )

    assert common.maybe_get_tosa_artifact_path() == str(
        tmp_path / "test_add_tosa_INT[shape]"
    )


def test_dump_artifacts_sanitizes_commas_and_spaces_in_test_name(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        pytest,
        "_test_options",
        {"dump_artifacts": str(tmp_path)},
        raising=False,
    )
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "backends/arm/test/ops/test_add.py::test_add_tosa_INT[1, 2, 3] (call)",
    )

    assert common.maybe_get_tosa_artifact_path() == str(
        tmp_path / "test_add_tosa_INT[1_2_3]"
    )


def test_custom_path_overrides_dump_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        pytest,
        "_test_options",
        {"dump_artifacts": str(tmp_path / "artifacts")},
        raising=False,
    )

    compile_spec = common.get_tosa_compile_spec(
        "TOSA-1.0+INT", custom_path=str(tmp_path / "custom")
    )

    assert compile_spec._get_intermediate_path() == str(tmp_path / "custom")
