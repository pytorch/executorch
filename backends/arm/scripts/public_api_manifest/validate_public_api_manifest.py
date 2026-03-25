#!/usr/bin/env python3

# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Validate one Arm public API manifest against the current API."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

try:
    import executorch.backends.arm.scripts.public_api_manifest.generate_public_api_manifest as gpam
except ModuleNotFoundError:
    generator_path = Path(__file__).resolve().parent / "generate_public_api_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "generate_public_api_manifest",
        generator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator script at {generator_path}")
    gpam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpam)

REPO_PATH = Path(__file__).resolve().parents[4]
MANIFEST_PATH = (
    REPO_PATH
    / "backends"
    / "arm"
    / "public_api_manifests"
    / "api_manifest_running.toml"
)
Issue = tuple[str, str, str | None, str | None]
NEW_API_SYMBOL_REASON = (
    "entry is present in the current API but missing from the manifest"
)
INCOMPATIBLE_SIGNATURE_REASON = "signature is not backward compatible"


ParameterDescriptor = tuple[str, str, str | None]
ParsedSignature = tuple[list[ParameterDescriptor], str | None]


def read_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "rb") as manifest_file:
        return tomllib.load(manifest_file)


def get_manifest_python_symbols(manifest: dict) -> dict[str, dict[str, str]]:
    python_manifest = manifest.get("python")
    if not isinstance(python_manifest, dict):
        raise ValueError("Manifest is missing [python] section")

    symbols: dict[str, dict[str, str]] = {}
    for name, entry in python_manifest.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Entry [python.{name}] must be a table")
        kind = entry.get("kind")
        signature = entry.get("signature")
        if not isinstance(kind, str) or not isinstance(signature, str):
            raise ValueError(
                f"Entry [python.{name}] must define `kind` and `signature`"
            )
        symbols[name] = {"kind": kind, "signature": signature}
    return symbols


def get_current_python_symbols(
    *,
    include_deprecated: bool = False,
) -> dict[str, dict[str, str]]:
    generated_manifest = tomllib.loads(
        gpam.generate_manifest_from_init(
            repo_path=REPO_PATH,
            include_deprecated=include_deprecated,
        )
    )
    return get_manifest_python_symbols(generated_manifest)


def _parse_signature(signature: str) -> ParsedSignature:
    suffix_start = signature.find("(")
    if suffix_start == -1:
        raise ValueError(f"Malformed signature: {signature}")
    signature_suffix = signature[suffix_start:]
    function_definition = ast.parse(
        f"def _manifest_stub{signature_suffix}:\n    pass\n"
    )
    function_node = function_definition.body[0]
    if not isinstance(function_node, ast.FunctionDef):
        raise ValueError(f"Unable to parse signature: {signature}")

    parameters: list[ParameterDescriptor] = []
    positional_args = list(function_node.args.posonlyargs) + list(
        function_node.args.args
    )
    positional_defaults = [None] * (
        len(positional_args) - len(function_node.args.defaults)
    ) + list(function_node.args.defaults)

    for argument, default in zip(
        function_node.args.posonlyargs,
        positional_defaults[: len(function_node.args.posonlyargs)],
    ):
        parameters.append(
            (
                argument.arg,
                inspect.Parameter.POSITIONAL_ONLY.name,
                None if default is None else ast.unparse(default),
            )
        )

    for argument, default in zip(
        function_node.args.args,
        positional_defaults[len(function_node.args.posonlyargs) :],
    ):
        parameters.append(
            (
                argument.arg,
                inspect.Parameter.POSITIONAL_OR_KEYWORD.name,
                None if default is None else ast.unparse(default),
            )
        )

    if function_node.args.vararg is not None:
        parameters.append(
            (
                function_node.args.vararg.arg,
                inspect.Parameter.VAR_POSITIONAL.name,
                None,
            )
        )

    for argument, default in zip(
        function_node.args.kwonlyargs,
        function_node.args.kw_defaults,
    ):
        parameters.append(
            (
                argument.arg,
                inspect.Parameter.KEYWORD_ONLY.name,
                None if default is None else ast.unparse(default),
            )
        )

    if function_node.args.kwarg is not None:
        parameters.append(
            (
                function_node.args.kwarg.arg,
                inspect.Parameter.VAR_KEYWORD.name,
                None,
            )
        )

    return_annotation = (
        None if function_node.returns is None else ast.unparse(function_node.returns)
    )
    return parameters, return_annotation


def is_signature_backward_compatible(
    manifest_signature: str,
    current_signature: str,
) -> bool:
    try:
        manifest_parameters, manifest_return = _parse_signature(manifest_signature)
        current_parameters, current_return = _parse_signature(current_signature)
    except (SyntaxError, ValueError):
        return False

    if manifest_return != current_return:
        return False

    if len(current_parameters) < len(manifest_parameters):
        return False

    for expected, actual in zip(manifest_parameters, current_parameters):
        if actual != expected:
            return False

    for _, kind, default in current_parameters[len(manifest_parameters) :]:
        if (
            kind
            not in (
                inspect.Parameter.VAR_POSITIONAL.name,
                inspect.Parameter.VAR_KEYWORD.name,
            )
            and default is None
        ):
            return False

    return True


def validate_symbols(
    manifest_symbols: dict[str, dict[str, str]],
    current_symbols: dict[str, dict[str, str]],
    *,
    ignore_new_api_symbols: bool = False,
    allow_backward_compatible_signature_changes: bool = False,
) -> list[Issue]:
    issues: list[Issue] = []
    manifest_keys = set(manifest_symbols)
    current_keys = set(current_symbols)

    for name in sorted(manifest_keys - current_keys):
        issues.append(
            (
                name,
                "entry is present in the manifest but missing from the current API",
                manifest_symbols[name]["signature"],
                None,
            )
        )

    if not ignore_new_api_symbols:
        for name in sorted(current_keys - manifest_keys):
            issues.append(
                (
                    name,
                    NEW_API_SYMBOL_REASON,
                    None,
                    current_symbols[name]["signature"],
                )
            )

    for name in sorted(manifest_keys & current_keys):
        expected = manifest_symbols[name]
        actual = current_symbols[name]
        if actual["kind"] != expected["kind"]:
            issues.append(
                (
                    name,
                    f"kind changed from '{expected['kind']}' to '{actual['kind']}'",
                    expected["signature"],
                    actual["signature"],
                )
            )
        elif actual["signature"] != expected["signature"] and (
            not allow_backward_compatible_signature_changes
            or not is_signature_backward_compatible(
                expected["signature"],
                actual["signature"],
            )
        ):
            issues.append(
                (
                    name,
                    (
                        INCOMPATIBLE_SIGNATURE_REASON
                        if allow_backward_compatible_signature_changes
                        else "signature changed"
                    ),
                    expected["signature"],
                    actual["signature"],
                )
            )
    return issues


def format_manifest_guidance(manifest_path: Path) -> str:
    if manifest_path.name == "api_manifest_running.toml":
        return (
            f"If this change is intentional, regenerate {manifest_path.name} and amend "
            "it into your change."
        )
    return (
        "If this change is intentional, deprecate the old symbol instead of "
        "changing or removing it directly."
    )


def format_validation_report(manifest_path: Path, issues: list[Issue]) -> str:
    if not issues:
        return f"{manifest_path.name}: public API is up to date."

    lines = [f"{manifest_path.name}: public API validation failed."]
    for name, reason, expected, actual in issues:
        lines.append(f"- {name}: {reason}")
        if expected is not None:
            lines.append(f"  manifest: {expected}")
        if actual is not None:
            lines.append(f"  current:  {actual}")
    if manifest_path.name == MANIFEST_PATH.name and any(
        reason == NEW_API_SYMBOL_REASON for _, reason, _, _ in issues
    ):
        lines.append(
            "If you intentionally added a new API symbol, update the running "
            "manifest with:"
        )
        lines.append("")
        lines.append(
            "python backends/arm/scripts/public_api_manifest/generate_public_api_manifest.py"
        )
        lines.append("")
        lines.append("and amend the manifest into your change.")
    else:
        lines.append(format_manifest_guidance(manifest_path))
    return "\n".join(lines)


def validate_manifest(manifest_path: Path) -> list[Issue]:
    return validate_symbols(
        get_manifest_python_symbols(read_manifest(manifest_path)),
        get_current_python_symbols(
            include_deprecated=manifest_path.name != MANIFEST_PATH.name,
        ),
        ignore_new_api_symbols=manifest_path.name != MANIFEST_PATH.name,
        allow_backward_compatible_signature_changes=(
            manifest_path.name != MANIFEST_PATH.name
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to the public API manifest TOML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues = validate_manifest(args.manifest)
    print(format_validation_report(args.manifest, issues))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
