# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from executorch.backends.arm.test import common


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
