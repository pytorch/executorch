# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from executorch.backends.arm.test import common
from packaging.specifiers import InvalidSpecifier
from packaging.version import Version

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "version,specifier,expected",
    [
        ("0.9.0", "==0.10.0", False),
        ("0.10.0", "==0.10.0", True),
        ("0.10.1", "==0.10.0", False),
        ("0.10.1", ">=0.10,<0.11", True),
        ("0.11.0", ">=0.10,<0.11", False),
        ("0.10.0rc1", ">=0.10.0rc1,<0.11", True),
        (Version("0.10.0+local"), "==0.10.0", True),
        (None, "==0.10.0", False),
    ],
)
def test_version_specifier(version, specifier, expected):
    marker = common.xfail_if_version(version, specifier, reason="regression")

    assert marker.kwargs["condition"] is expected


@pytest.mark.parametrize("version", [None, "4.3.0-29-gd37febc"])
def test_invalid_specifier_rejected_even_with_unknown_version(version):
    with pytest.raises(InvalidSpecifier):
        common.xfail_if_version(version, "not a specifier", reason="regression")


@pytest.mark.parametrize("version", ["4.3.0-29-gd37febc", "unknown"])
def test_unparseable_version_warns_without_xfail(version):
    with pytest.warns(RuntimeWarning, match="Unparseable dependency version") as caught:
        marker = common.xfail_if_version(version, ">=4.3", reason="Vela regression")

    assert marker.kwargs["condition"] is False
    assert version in str(caught[0].message)
    assert "Vela regression" in str(caught[0].message)


@pytest.mark.parametrize("version,expected", [("5.0.0", False), ("5.1.0", True)])
def test_dependency_distribution_metadata(tmp_path, monkeypatch, version, expected):
    distribution = tmp_path / f"xfail_test_dependency-{version}.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: xfail-test-dependency\nVersion: {version}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    marker = common.xfail_if_dependency_version(
        "xfail-test-dependency",
        ">=5.1,<5.2",
        module="unused_fallback_module",
        reason="regression",
        strict=False,
        raises=ValueError,
    )

    assert marker.kwargs == {
        "condition": expected,
        "reason": "regression",
        "strict": False,
        "raises": ValueError,
    }


@pytest.mark.parametrize("module", [None, "missing_xfail_dependency.submodule"])
def test_missing_dependency_does_not_xfail(module):
    marker = common.xfail_if_dependency_version(
        "missing-xfail-dependency", ">=1", module=module, reason="regression"
    )

    assert marker.kwargs["condition"] is False


@pytest.mark.parametrize(
    "version,expected,warning_count",
    [
        ("5.0.0", False, 0),
        ("5.1.0", True, 0),
        ("4.3.0-29-gd37febc", False, 1),
    ],
)
def test_vela_module_fallback(monkeypatch, recwarn, version, expected, warning_count):
    vela = pytest.importorskip("ethosu.vela")
    monkeypatch.setattr(vela, "__version__", version)

    def missing_metadata(dependency):
        raise common.metadata.PackageNotFoundError(dependency)

    monkeypatch.setattr(common.metadata, "version", missing_metadata)

    marker = common.xfail_if_dependency_version(
        "ethos-u-vela",
        "==5.1.0",
        module="ethosu.vela",
        reason="regression",
    )

    assert marker.kwargs["condition"] is expected
    assert len(recwarn) == warning_count
    if warning_count:
        warning = recwarn.pop(RuntimeWarning)
        assert "Unparseable dependency version" in str(warning.message)
        assert version in str(warning.message)


def test_suite_collects_without_vela(pytester, monkeypatch):
    monkeypatch.setitem(sys.modules, "ethosu.vela", None)
    test_file = pytester.makepyfile(test_without_vela=Path(__file__).read_text())

    result = pytester.runpytest_inprocess(
        "-q",
        f"{test_file}::test_version_specifier",
        f"{test_file}::test_vela_module_fallback",
    )

    result.assert_outcomes(passed=8, skipped=3)


def test_module_fallback_does_not_hide_broken_import(tmp_path, monkeypatch):
    (tmp_path / "broken_xfail_dependency.py").write_text(
        "import missing_transitive_xfail_dependency\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(
        ModuleNotFoundError, match="missing_transitive_xfail_dependency"
    ):
        common.xfail_if_dependency_version(
            "missing-xfail-dependency",
            ">=1",
            module="broken_xfail_dependency",
            reason="regression",
        )


@pytest.mark.parametrize(
    "version_text,expected",
    [
        ("model-converter 0.9.0", False),
        ("model-converter 0.10.0", True),
        ("model-converter 0.10.1", True),
        ("model-converter 0.11.0", True),
        ("model-converter d8c1b8e", False),
        ("model-converter 19d1d0f", True),
        ("model-converter unknown-build", False),
        (None, False),
    ],
)
def test_model_converter_version(monkeypatch, version_text, expected):
    monkeypatch.setattr(
        common, "get_model_converter_version_text", lambda: version_text
    )

    marker = common.xfail_if_model_converter_version(">=0.10.0", reason="regression")

    assert marker.kwargs["condition"] is expected


@pytest.mark.parametrize(
    "version,passes,outcomes",
    [
        ("0.9.0", True, {"passed": 1}),
        ("0.9.0", False, {"failed": 1}),
        ("0.10.0", False, {"xfailed": 1}),
        ("0.10.0", True, {"failed": 1}),
        (None, False, {"failed": 1}),
        ("4.3.0-29-gd37febc", True, {"passed": 1}),
        ("4.3.0-29-gd37febc", False, {"failed": 1}),
    ],
)
def test_decorator_outcomes(pytester, version, passes, outcomes):
    pytester.makepyfile(
        f"""
        from executorch.backends.arm.test import common

        @common.xfail_if_version({version!r}, "==0.10.0", reason="regression")
        def test_dependency():
            assert {passes!r}
        """
    )

    result = pytester.runpytest_inprocess("-q")

    result.assert_outcomes(**outcomes)


def test_parametrize_xfails_apply_only_to_selected_cases(pytester):
    pytester.makepyfile(
        """
        from executorch.backends.arm.test import common

        @common.parametrize(
            "passes",
            {"affected": False, "unaffected": False, "legacy": False, "tuple": False},
            xfails={
                "affected": common.xfail_if_version(
                    "0.10.0", "==0.10.0", reason="regression"
                ),
                "legacy": "known failure",
                "tuple": ("known assertion", AssertionError),
            },
        )
        def test_cases(passes):
            assert passes
        """
    )

    result = pytester.runpytest_inprocess("-q")

    result.assert_outcomes(xfailed=3, failed=1)


def test_parametrize_preserves_explicit_marker_options(pytester):
    pytester.makepyfile(
        """
        from executorch.backends.arm.test import common

        @common.parametrize(
            "passes",
            {"unexpected_pass": True, "wrong_exception": False},
            strict=False,
            xfails={
                name: common.xfail_if_version(
                    "0.10.0", "==0.10.0", reason="regression", raises=ValueError
                )
                for name in ("unexpected_pass", "wrong_exception")
            },
        )
        def test_cases(passes):
            assert passes
        """
    )

    result = pytester.runpytest_inprocess("-q")

    result.assert_outcomes(failed=2)
