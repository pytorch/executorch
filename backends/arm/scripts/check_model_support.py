# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fast pre-backend operator-list check for Arm model support.

This tool exports a PyTorch model to ATen and compares operators visible in the
pre-backend ``torch.export`` graph with the generated operator-support table
committed in the current ExecuTorch checkout.

The comparison is intentionally lightweight. Normal ExecuTorch/Arm lowering may
decompose, rewrite, canonicalize, or remove an exported operator before backend
support checking, so an unmatched raw-export operator is reported as
``INCONCLUSIVE`` rather than as definitive backend rejection.

The model file follows the same convention as ``aot_arm_compiler.py``:

    ModelUnderTest = MyModel()
    ModelInputs = (example_input, ...)

An optional ``ModelKwargs`` dictionary is also accepted.

Examples:
    python backends/arm/scripts/check_model_support.py \\
        --backend vgf \\
        --model examples/arm/example_modules/add.py

    python backends/arm/scripts/check_model_support.py \\
        --backend u85 \\
        --model backends/arm/scripts/examples/fast_model_support_demo.py

Exit codes:
    0: No operator-list gap was found in the pre-backend export graph.
    1: At least one pre-backend export operator is absent from the support table;
       the result is inconclusive because later lowering may transform it.
    2: The checker could not load/export/check the model.

"""

from __future__ import annotations

import argparse
import importlib.util
import operator
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable

import torch


BACKEND_SUPPORT_TABLES = {
    "vgf": Path("docs/source/backends/arm-vgf/VGF_op_support.md"),
    "u55": Path("docs/source/backends/arm-ethos-u/U55_op_support.md"),
    "u85": Path("docs/source/backends/arm-ethos-u/U85_op_support.md"),
}

_DOCGEN_PATH = Path("backends/arm/scripts/docgen/generate_op_support.py")
_REPO_MARKER = Path("backends/arm/README.md")
_IGNORED_CALL_FUNCTION_TARGETS = frozenset({operator.getitem})

# Operators that may be visible in torch.export but are not backend support
# requirements. Keep them in the collected export-op list for diagnostics, then
# filter them at the support-decision layer so --verbose can explain why they
# were ignored.
NON_BACKEND_EXPORT_OPS = {
    "torch.ops.aten._assert_tensor_metadata.default": (
        "ExecuTorch removes tensor-metadata assertions before backend lowering"
    ),
}

DISCLAIMER = (
    "PASS is only a pre-backend operator-name check; it does not prove that the "
    "model will quantize, partition, lower, compile, or run on the selected "
    "backend. INCONCLUSIVE means one or more operators visible in the "
    "pre-backend torch.export graph are absent from the generated support table; "
    "it does not prove that the backend rejects the model because normal "
    "ExecuTorch/Arm lowering may decompose, rewrite, canonicalize, or remove "
    "those operators before backend support checking."
)


@dataclass(frozen=True)
class UnmatchedExportOperator:
    exported_op: str
    public_apis: tuple[str, ...]

    @property
    def display_name(self) -> str:
        for api in self.public_apis:
            if api.startswith("torch."):
                return api
        return self.exported_op


# Compatibility alias for helper scripts written against the initial checker.
# New code should use UnmatchedExportOperator because a missing raw-export op is
# not a definitive backend support failure.
UnsupportedOperator = UnmatchedExportOperator


def _find_repo_root(start: Path | None = None) -> Path:
    """Find the ExecuTorch checkout containing the support lists to inspect."""

    start = (start or Path.cwd()).resolve()
    candidates = (start, *start.parents)
    for candidate in candidates:
        if (candidate / _REPO_MARKER).is_file() and (
            candidate / _DOCGEN_PATH
        ).is_file():
            return candidate
    raise RuntimeError(
        "Could not find an ExecuTorch repository from the current directory. "
        "Run this tool from inside the checkout whose support lists you want "
        "to inspect."
    )


def _load_python_file(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # dataclasses and some import-time helpers expect the module to exist in
    # sys.modules while it is executing.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_model_definition(
    model_path: Path,
) -> tuple[torch.nn.Module, tuple[object, ...], dict[str, object]]:
    if not model_path.is_file():
        raise RuntimeError(f"Model file does not exist: {model_path}")

    module = _load_python_file(model_path, "_arm_fast_support_model")

    if not hasattr(module, "ModelUnderTest"):
        raise RuntimeError(
            f"{model_path} must define ModelUnderTest, matching the Arm "
            "aot_arm_compiler.py example-module convention."
        )
    if not hasattr(module, "ModelInputs"):
        raise RuntimeError(
            f"{model_path} must define ModelInputs as the example inputs used "
            "for torch.export.export."
        )

    model = module.ModelUnderTest
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"ModelUnderTest in {model_path} must be a torch.nn.Module instance."
        )

    inputs = module.ModelInputs
    if isinstance(inputs, tuple):
        example_inputs = inputs
    elif isinstance(inputs, list):
        example_inputs = tuple(inputs)
    else:
        example_inputs = (inputs,)

    kwargs = getattr(module, "ModelKwargs", {})
    if not isinstance(kwargs, dict):
        raise RuntimeError("ModelKwargs must be a dict when provided.")

    return model.eval(), example_inputs, kwargs


def _canonical_exported_op(target: object) -> str | None:
    """Convert torch.export call targets to docgen's canonical op spelling."""

    if target in _IGNORED_CALL_FUNCTION_TARGETS:
        return None

    if isinstance(target, torch._ops.OpOverload):
        return f"torch.ops.{target}"

    # Higher-order operators are not OpOverload instances. The two currently
    # represented in the generated Arm support tables are cond and while_loop.
    target_name = str(target)
    if target_name in {"cond", "while_loop"}:
        return f"torch.ops.higher_order.{target_name}"

    # Be conservative for unexpected torch call_function targets: surface them
    # as unsupported rather than silently claiming the model passes.
    module_name = getattr(target, "__module__", "")
    function_name = getattr(target, "__name__", "")
    if module_name.startswith("torch"):
        rendered = ".".join(part for part in (module_name, function_name) if part)
        return rendered or repr(target)

    return None


def _collect_exported_operators(
    model: torch.nn.Module,
    example_inputs: tuple[object, ...],
    model_kwargs: dict[str, object] | None = None,
) -> list[str]:
    """Export a model and return unique ATen/higher-order operator targets."""

    exported_program = torch.export.export(
        model,
        example_inputs,
        kwargs=model_kwargs or {},
        strict=True,
    )

    operators: set[str] = set()
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op != "call_function":
                continue
            op_name = _canonical_exported_op(node.target)
            if op_name is not None:
                operators.add(op_name)

    return sorted(operators)


def _split_markdown_row(line: str) -> list[str]:
    """Split a Markdown table row while preserving escaped pipe characters."""

    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def _read_supported_public_apis(table_path: Path) -> set[str]:
    """Read the PyTorch API column from a generated Arm support table."""

    if not table_path.is_file():
        raise RuntimeError(
            f"Support table is missing: {table_path}. Regenerate the Arm "
            "operator-support documentation for this checkout."
        )

    supported: set[str] = set()
    in_table = False

    for line in table_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| PyTorch API |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break

        cells = _split_markdown_row(line)
        if not cells or cells[0].startswith("---"):
            continue

        for code_span in re.findall(r"`([^`]+)`", cells[0]):
            if code_span.startswith("torch."):
                supported.add(code_span)

    if not supported:
        raise RuntimeError(
            f"No PyTorch API entries were found in generated support table "
            f"{table_path}."
        )
    return supported


def _load_api_alias_resolver(
    repo_root: Path,
) -> Callable[[str], tuple[str, ...]]:
    """Reuse the exact ATen -> public PyTorch API mapping used by docgen."""

    docgen = _load_python_file(
        repo_root / _DOCGEN_PATH,
        "_arm_fast_support_docgen",
    )
    resolver = getattr(docgen, "_pytorch_api_aliases", None)
    if resolver is None:
        raise RuntimeError(
            "Arm docgen no longer exposes _pytorch_api_aliases; update the fast "
            "checker together with generate_op_support.py."
        )

    def resolve(exported_op: str) -> tuple[str, ...]:
        aliases = resolver(exported_op)
        return tuple(str(alias) for alias in aliases)

    return resolve


def _find_unmatched_export_operators(
    exported_ops: Iterable[str],
    supported_public_apis: set[str],
    resolve_aliases: Callable[[str], tuple[str, ...]],
) -> list[UnmatchedExportOperator]:
    """Return pre-backend export ops that do not match the support table.

    This is deliberately not named "unsupported": the normal ExecuTorch/Arm
    lowering path may decompose, rewrite, canonicalize, or remove these ops
    before the selected backend performs support checking.

    """

    unmatched: list[UnmatchedExportOperator] = []

    for exported_op in sorted(set(exported_ops)):
        if exported_op in NON_BACKEND_EXPORT_OPS:
            continue

        public_apis = resolve_aliases(exported_op)
        if not any(api in supported_public_apis for api in public_apis):
            unmatched.append(
                UnmatchedExportOperator(
                    exported_op=exported_op,
                    public_apis=public_apis,
                )
            )

    return unmatched


def _find_unsupported_operators(
    exported_ops: Iterable[str],
    supported_public_apis: set[str],
    resolve_aliases: Callable[[str], tuple[str, ...]],
) -> list[UnmatchedExportOperator]:
    """Compatibility wrapper; use _find_unmatched_export_operators instead."""

    return _find_unmatched_export_operators(
        exported_ops,
        supported_public_apis,
        resolve_aliases,
    )


def check_model_support(
    *,
    backend: str,
    model: torch.nn.Module,
    example_inputs: tuple[object, ...],
    model_kwargs: dict[str, object] | None,
    repo_root: Path,
) -> tuple[list[str], list[UnmatchedExportOperator], Path]:
    """Return exported ops, unmatched pre-backend ops, and the support table."""

    support_table = repo_root / BACKEND_SUPPORT_TABLES[backend]
    supported_apis = _read_supported_public_apis(support_table)
    exported_ops = _collect_exported_operators(
        model,
        example_inputs,
        model_kwargs,
    )
    resolver = _load_api_alias_resolver(repo_root)
    unmatched = _find_unmatched_export_operators(
        exported_ops,
        supported_apis,
        resolver,
    )
    return exported_ops, unmatched, support_table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast pre-backend comparison of exported PyTorch operators against "
            "the Arm backend support table in the current checkout."
        )
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=tuple(BACKEND_SUPPORT_TABLES),
        help="Arm backend support table to check: vgf, u55, or u85.",
    )
    parser.add_argument(
        "--model",
        "--model_name",
        dest="model_path",
        required=True,
        help=(
            "Python model file defining ModelUnderTest and ModelInputs. "
            "--model_name is accepted for compatibility with aot_arm_compiler.py."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print every exported operator and its mapped PyTorch APIs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        repo_root = _find_repo_root()
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()

        model, example_inputs, model_kwargs = _load_model_definition(model_path)
        exported_ops, unmatched, support_table = check_model_support(
            backend=args.backend,
            model=model,
            example_inputs=example_inputs,
            model_kwargs=model_kwargs,
            repo_root=repo_root,
        )
        resolver = _load_api_alias_resolver(repo_root)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    ignored_ops = [op for op in exported_ops if op in NON_BACKEND_EXPORT_OPS]
    backend_ops = [op for op in exported_ops if op not in NON_BACKEND_EXPORT_OPS]

    print(f"Backend: {args.backend}")
    print(f"Support list: {support_table.relative_to(repo_root)}")
    print(f"Exported operators found: {len(exported_ops)}")
    print(f"Backend-relevant operators checked: {len(backend_ops)}")
    if ignored_ops:
        print(f"Non-backend export operators ignored: {len(ignored_ops)}")

    if args.verbose:
        unmatched_names = {item.exported_op for item in unmatched}
        print("\nExported operators:")
        for exported_op in exported_ops:
            if exported_op in NON_BACKEND_EXPORT_OPS:
                status = "IGNORED"
                rendered = NON_BACKEND_EXPORT_OPS[exported_op]
            else:
                aliases = resolver(exported_op)
                status = "UNMATCHED" if exported_op in unmatched_names else "MATCHED"
                rendered = ", ".join(aliases) if aliases else "<no API alias>"
            print(f"  [{status}] {exported_op} -> {rendered}")

    if unmatched:
        print(f"\nUnmatched pre-backend export operators ({len(unmatched)}):")
        for item in unmatched:
            print(f"  - {item.display_name} [{item.exported_op}]")
        print("\nResult: INCONCLUSIVE")
        print(f"\nImportant: {DISCLAIMER}")
        return 1

    print("\nUnmatched pre-backend export operators: none")
    print("Result: PASS")
    print(f"\nImportant: {DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
