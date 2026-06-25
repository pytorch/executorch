# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import executorch.backends.arm.scripts.public_api_manifest.validate_public_api_manifest as vpam

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

from executorch.backends.arm.scripts.public_api_manifest.validate_public_api_manifest import (
    format_validation_report,
    get_current_python_symbols,
    get_manifest_python_symbols,
    validate_symbols,
)

RUNNING_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "public_api_manifests"
    / "api_manifest_running.toml"
)
MOCK_STATIC_MANIFEST_PATH = Path("mock_api_manifest_static_VERSION.toml")


def test_public_api_manifest_exact_comparison_rejects_signature_expansion():
    manifest_entries = {"foo": {"kind": "function", "signature": "foo(x: int) -> int"}}
    current_entries = {
        "foo": {
            "kind": "function",
            "signature": "foo(x: int, y: int | None = None) -> int",
        }
    }

    issues = validate_symbols(manifest_entries, current_entries)

    assert len(issues) == 1
    assert issues[0][0] == "foo"
    assert issues[0][1] == "signature changed"


def test_get_manifest_python_symbols_flattens_nested_tables():
    manifest = tomllib.loads(
        """
        [python]

        [python.Foo]
        kind = "class"
        signature = "Foo()"

        [python.Foo.bar]
        kind = "function"
        signature = "Foo.bar() -> None"
        """
    )

    assert get_manifest_python_symbols(manifest) == {
        "Foo": {"kind": "class", "signature": "Foo()"},
        "Foo.bar": {"kind": "function", "signature": "Foo.bar() -> None"},
    }


def test_nested_python_manifest_entries_are_validated():
    manifest_symbols = get_manifest_python_symbols(
        tomllib.loads(
            """
            [python]

            [python.Foo]
            kind = "class"
            signature = "Foo()"

            [python.Foo.bar]
            kind = "function"
            signature = "Foo.bar(x: int) -> int"
            """
        )
    )

    issues = validate_symbols(
        manifest_symbols,
        {
            "Foo": {"kind": "class", "signature": "Foo()"},
        },
    )

    assert issues == [
        (
            "Foo.bar",
            "entry is present in the manifest but missing from the current API",
            "Foo.bar(x: int) -> int",
            None,
        )
    ]


def test_public_api_manifest_static_accepts_backward_compatible_signature_expansion():
    manifest_entries = {
        "foo": {"kind": "function", "signature": "foo(x: int, y: int = 0) -> int"}
    }
    current_entries = {
        "foo": {
            "kind": "function",
            "signature": "foo(x: int, y: int = 0, z: int | None = None) -> int",
        }
    }

    issues = validate_symbols(
        manifest_entries,
        current_entries,
        ignore_new_api_symbols=True,
        allow_backward_compatible_signature_changes=True,
    )

    assert issues == []


def test_public_api_manifest_static_rejects_new_required_parameter():
    manifest_entries = {"foo": {"kind": "function", "signature": "foo(x: int) -> int"}}
    current_entries = {
        "foo": {
            "kind": "function",
            "signature": "foo(x: int, y: int) -> int",
        }
    }

    issues = validate_symbols(
        manifest_entries,
        current_entries,
        ignore_new_api_symbols=True,
        allow_backward_compatible_signature_changes=True,
    )

    assert len(issues) == 1
    assert issues[0][0] == "foo"
    assert issues[0][1] == vpam.INCOMPATIBLE_SIGNATURE_REASON


def test_public_api_manifest_exact_comparison_rejects_additions():
    manifest_entries = {"foo": {"kind": "function", "signature": "foo(x: int) -> int"}}
    current_entries = {
        "bar": {"kind": "function", "signature": "bar() -> int"},
        "foo": {"kind": "function", "signature": "foo(x: int) -> int"},
    }

    issues = validate_symbols(manifest_entries, current_entries)

    assert len(issues) == 1
    assert issues[0][0] == "bar"
    assert "missing from the manifest" in issues[0][1]


def test_public_api_manifest_running_regeneration_reports_drift():
    manifest_entries = {"foo": {"kind": "function", "signature": "foo(x: int) -> int"}}
    current_entries = {
        "bar": {"kind": "function", "signature": "bar() -> int"},
        "foo": {
            "kind": "function",
            "signature": "foo(x: int, y: int | None = None) -> int",
        },
    }

    issues = validate_symbols(manifest_entries, current_entries)
    report = format_validation_report(RUNNING_MANIFEST_PATH, issues)

    assert len(issues) == 2
    assert {issue[0] for issue in issues} == {"foo", "bar"}
    assert "public API validation failed" in report
    assert "added a new API symbol" in report
    assert "manifest with:" in report
    assert (
        "manifest with:\n\n"
        "python backends/arm/scripts/public_api_manifest/generate_public_api_manifest.py\n\n"
        "and amend the manifest into your change."
    ) in report


def test_public_api_manifest_static_deprecation_reports_drift():
    manifest_entries = {"foo": {"kind": "function", "signature": "foo(x: int) -> int"}}
    current_entries = {
        "bar": {"kind": "function", "signature": "bar() -> int"},
        "foo": {"kind": "function", "signature": "foo(x: int, y: int = 0) -> int"},
    }

    issues = validate_symbols(
        manifest_entries,
        current_entries,
        ignore_new_api_symbols=True,
        allow_backward_compatible_signature_changes=True,
    )
    report = format_validation_report(MOCK_STATIC_MANIFEST_PATH, issues)

    assert issues == []
    assert "public API is up to date" in report


def test_public_api_manifest_static_reports_incompatible_signature_drift():
    manifest_entries = {
        "foo": {"kind": "function", "signature": "foo(x: int, y: int = 0) -> int"}
    }
    current_entries = {
        "foo": {
            "kind": "function",
            "signature": "foo(x: int, y: int, z: int | None = None) -> int",
        }
    }

    issues = validate_symbols(
        manifest_entries,
        current_entries,
        ignore_new_api_symbols=True,
        allow_backward_compatible_signature_changes=True,
    )
    report = format_validation_report(MOCK_STATIC_MANIFEST_PATH, issues)

    assert len(issues) == 1
    assert issues[0][0] == "foo"
    assert issues[0][1] == vpam.INCOMPATIBLE_SIGNATURE_REASON
    assert "deprecate the old symbol" in report


def test_public_api_manifest_static_ignores_additions():
    manifest_entries = {"foo": {"kind": "function", "signature": "foo(x: int) -> int"}}
    current_entries = {
        "bar": {"kind": "function", "signature": "bar() -> int"},
        "foo": {"kind": "function", "signature": "foo(x: int) -> int"},
    }

    issues = validate_symbols(
        manifest_entries,
        current_entries,
        ignore_new_api_symbols=True,
    )
    report = format_validation_report(MOCK_STATIC_MANIFEST_PATH, issues)

    assert issues == []
    assert "public API is up to date" in report


def test_get_current_python_symbols_can_include_deprecated(monkeypatch):
    def fake_generate_manifest_from_init(
        *,
        repo_path=None,
        include_deprecated: bool = False,
    ) -> str:
        del repo_path
        if include_deprecated:
            return '[python]\n\n[python.foo]\nkind = "function"\nsignature = "foo()"\n'
        return "[python]\n"

    monkeypatch.setattr(
        vpam.gpam, "generate_manifest_from_init", fake_generate_manifest_from_init
    )

    assert get_current_python_symbols() == {}
    assert get_current_python_symbols(include_deprecated=True) == {
        "foo": {"kind": "function", "signature": "foo()"}
    }
