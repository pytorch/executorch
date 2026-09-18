# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from types import SimpleNamespace

import executorch.backends.arm.test.conftest as arm_conftest

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester


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


@pytest.fixture(params=["collation", "dump_artifacts"])
def artifact_base(request, monkeypatch, tmp_path):
    monkeypatch.delenv("TOSA_TESTCASES_BASE_PATH", raising=False)
    options = {}
    monkeypatch.setattr(pytest, "_test_options", options, raising=False)
    if request.param == "collation":
        monkeypatch.setenv("TOSA_TESTCASES_BASE_PATH", str(tmp_path))
        return tmp_path / "tosa-int"
    options["dump_artifacts"] = str(tmp_path)
    return tmp_path


@pytest.fixture(
    params=[
        partial(common.get_tosa_compile_spec, "TOSA-1.0+FP"),
        common.get_u55_compile_spec,
        common.get_u65_compile_spec,
        common.get_u85_compile_spec,
        partial(common.get_vgf_compile_spec, "TOSA-1.0+FP"),
    ],
    ids=["tosa", "u55", "u65", "u85", "vgf"],
)
def compile_spec_factory(request):
    return request.param


def test_artifact_path_requires_explicit_deferral(
    monkeypatch, artifact_base, compile_spec_factory
):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    compile_spec = compile_spec_factory()

    assert compile_spec._get_intermediate_path() is None
    assert common.maybe_get_tosa_artifact_path(allow_unresolved=True) is None
    with pytest.raises(RuntimeError, match="current pytest test name"):
        common.maybe_get_tosa_artifact_path()
    with pytest.raises(RuntimeError, match="current pytest test name"):
        ArmTester(torch.nn.Identity(), (torch.ones(1),), compile_spec)


def test_shared_compile_spec_gets_each_tests_artifact_path(
    monkeypatch, artifact_base, compile_spec_factory
):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    compile_spec = compile_spec_factory()
    testers = []

    for name in ("test_first_INT[case]", "test_second_INT[case]"):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", f"test_artifacts.py::{name} (call)")
        tester = ArmTester(torch.nn.Identity(), (torch.ones(1),), compile_spec)
        testers.append(tester)
        expected = artifact_base / name
        assert tester.compile_spec._get_intermediate_path() == str(expected)
        serialized = {spec.key: spec.value for spec in tester.compile_spec._to_list()}
        assert serialized["debug_artifact_path"] == str(expected).encode()
        assert expected.is_dir()
        assert compile_spec._get_intermediate_path() is None

    assert testers[0].compile_spec._get_intermediate_path() == str(
        artifact_base / "test_first_INT[case]"
    )


def test_explicit_artifact_path_takes_precedence_without_current_test(
    monkeypatch, tmp_path, artifact_base, compile_spec_factory
):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    explicit_path = tmp_path / "custom"
    compile_spec = compile_spec_factory(custom_path=str(explicit_path))
    tester = ArmTester(torch.nn.Identity(), (torch.ones(1),), compile_spec)

    assert tester.compile_spec._get_intermediate_path() == str(explicit_path)


def test_deferred_artifact_path_writes_tosa_output(monkeypatch, artifact_base):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+FP")
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST", "test_artifacts.py::test_lower_INT (call)"
    )
    tester = ArmTester(torch.nn.ReLU(), (torch.randn(1, 2, 3),), compile_spec)
    tester.export().to_edge_transform_and_lower()

    assert list((artifact_base / "test_lower_INT").rglob("*.tosa"))
    assert compile_spec._get_intermediate_path() is None
