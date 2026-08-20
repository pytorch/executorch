#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate RegisterAllKernels.cpp for the ExecuTorch CMSIS Pack.

This wraps ExecuTorch's own operator codegen pipeline (the same one that
produces RegisterCodegenUnboxedKernels.cpp for a normal lean build) and adds the
single thing the CMSIS Pack needs on top: a per-operator
``#ifdef RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>`` guard around every forward
declaration and kernel registration.

The pipeline derives every C++ signature, EValue unboxing body and registration
entry from the operator YAML schema -- nothing is parsed out of the compiled
.cpp sources, so the output stays correct as kernels evolve. The guard naming
and operator discovery are shared with generate_components.py via op_guards.py
so the registration #ifdef set and the component #define set cannot drift.

Usage:
    python generate_register_all_kernels.py \
        --source-dir /path/to/pack-staging \
        --output /path/to/pack-staging/src/registration/RegisterAllKernels.cpp

"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torchgen
from executorch.codegen.api.types import ExecutorchCppSignature
from executorch.codegen.gen import (
    ComputeCodegenUnboxedKernels,
    parse_yaml,
    parse_yaml_files,
)
from executorch.codegen.model import ETKernelIndex

from op_guards import (  # type: ignore[import-not-found]
    discover_components,
    kernel_source_index,
)
from torchgen.context import native_function_manager
from torchgen.selective_build.selector import SelectiveBuilder


# ===========================================================================
# Parsing the operator YAML into NativeFunctions + kernel index
# ===========================================================================


def _torchgen_yaml() -> tuple[Path, Path]:
    """Locate the canonical ATen native_functions.yaml + tags.yaml shipped with
    the installed torchgen package.

    They are needed to resolve the ``op:``
    references in kernels/portable/functions.yaml to full schemas.

    """
    native = Path(torchgen.__file__).parent / "packaged" / "ATen" / "native"
    return native / "native_functions.yaml", native / "tags.yaml"


def _find_yaml(source_dir: Path, rel: str) -> Path | None:
    """Resolve a backend YAML under the staged include/ layout, the staged src/
    layout, or a flat source tree.
    """
    for root in (source_dir / "include" / "executorch", source_dir / "src", source_dir):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def _parse_portable(
    functions_yaml: Path | None,
    custom_yaml: Path | None,
    aten_yaml: Path,
    tags_yaml: Path,
    selector: SelectiveBuilder,
) -> tuple[list, ETKernelIndex]:
    """Parse functions.yaml (``op:`` refs, translated against the ATen schema)
    together with custom_ops.yaml (full ``func:`` schemas).
    """
    combined, _ = parse_yaml_files(
        tags_yaml_path=str(tags_yaml),
        aten_yaml_path=str(aten_yaml),
        native_yaml_path=str(functions_yaml) if functions_yaml else None,
        custom_ops_yaml_path=str(custom_yaml) if custom_yaml else None,
        selector=selector,
        use_aten_lib=False,
    )
    return list(combined[0]), combined[1]


def _parse_custom(
    custom_yaml: Path,
    tags_yaml: Path,
    selector: SelectiveBuilder,
) -> tuple[list, ETKernelIndex]:
    """Parse a custom-ops YAML (quantized.yaml / cortex_m operators.yaml) that
    already carries full ``func:`` schemas, so no ATen translation is needed.
    """
    native_functions, index = parse_yaml(
        str(custom_yaml),
        str(tags_yaml),
        lambda f: selector.is_native_function_selected(f),
        skip_native_fns_gen=True,
    )
    if not isinstance(index, ETKernelIndex):
        index = ETKernelIndex.from_backend_indices(index)
    return list(native_functions), index


# ===========================================================================
# Rendering: codegen pipeline output + #ifdef guard layer
# ===========================================================================


@dataclass
class GuardGroup:
    """All forward declarations and registrations sharing one #ifdef guard (i.e.
    one op_*.cpp source file / CMSIS component).
    """

    guard: str
    forward_decls: list  # list[str]
    registrations: list  # list[str]


def render_category(
    category: str,
    native_functions: list,
    kernel_index: ETKernelIndex,
    source_index: dict,
    selector: SelectiveBuilder,
) -> tuple[list, list]:
    """Render forward declarations + registrations for one category, grouped by
    guard.

    The guard is found from the kernel's defining op_*.cpp via
    ``source_index``. Returns (groups, unmatched) where ``unmatched`` lists
    kernels whose symbol is not defined in any staged source (and are skipped).

    """
    compute = ComputeCodegenUnboxedKernels(selector, False, False)
    forward_decls: dict = defaultdict(list)
    registrations: dict = defaultdict(list)
    declared: set = set()  # kernel function names already declared in this category
    unmatched: list = []

    for native_function in native_functions:
        op_name = str(native_function.func.name)
        kernels = kernel_index.get_kernels(native_function)
        if not kernels:
            continue
        sig = ExecutorchCppSignature.from_native_function(native_function)
        with native_function_manager(native_function):
            for kernel_key, metadata in kernels.items():
                component = source_index.get((category, metadata.kernel))
                if component is None:
                    unmatched.append((category, op_name, metadata.kernel))
                    continue
                guard = component.guard
                registration = compute((native_function, (kernel_key, metadata)))
                if registration.strip():
                    registrations[guard].append(registration)
                if metadata.kernel not in declared:
                    declared.add(metadata.kernel)
                    forward_decls[guard].append(
                        sig.decl(name=metadata.kernel, include_context=True) + ";"
                    )

    groups = [
        GuardGroup(guard, forward_decls.get(guard, []), registrations.get(guard, []))
        for guard in sorted(set(forward_decls) | set(registrations))
    ]
    return groups, unmatched


# ===========================================================================
# File assembly
# ===========================================================================

_BANNER = "// ====================================================================="

_FILE_HEADER = """\
/*
 * CMSIS Pack Kernel Registration
 *
 * Auto-generated by generate_register_all_kernels.py
 *
 * This file contains all ExecuTorch kernel registrations, with each operator
 * guarded by its corresponding RTE_Components_h define.
 *
 * When a user selects an operator component in the CMSIS pack, that component
 * defines RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>, which enables the
 * registration of that operator in this file.
 *
 * Define naming convention:
 *   RTE_ML_EXECUTORCH_OP_PORTABLE_<NAME>   - Portable operators
 *   RTE_ML_EXECUTORCH_OP_QUANTIZED_<NAME>  - Quantized operators
 *   RTE_ML_EXECUTORCH_OP_CORTEX_M_<NAME>   - Cortex-M operators
 *
 * This approach allows:
 * 1. Only selected operators to be registered (minimizing code size)
 * 2. No per-operator registration files needed
 * 3. Single static initialization for all selected operators
 */

#include "RTE_Components.h"

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/platform/profiler.h>

using KernelSpan = ::executorch::runtime::Span<
    const ::executorch::runtime::Kernel>;

namespace torch {
namespace executor {

// Forward declarations for native functions (in torch::executor::native namespace)
namespace native {"""

_CORTEX_M_DECLS_HEADER = """\

namespace cortex_m {
namespace native {
// =====================================================================
// CORTEX-M OPERATOR FORWARD DECLARATIONS
// =====================================================================
// Bring kernel-API types into the cortex_m::native scope so the
// forward declarations below resolve. These mirror the file-scope
// using aliases in
// backends/cortex_m/ops/cortex_m_ops_common.h, but without pulling
// in arm_nn_types.h / arm_nnfunctions.h -- RegisterAllKernels.cpp
// only needs the type names, not the CMSIS-NN headers.
using Tensor = ::torch::executor::Tensor;
using ScalarType = ::executorch::aten::ScalarType;
using Int64ArrayRef = ::executorch::aten::ArrayRef<int64_t>;
using KernelRuntimeContext = ::torch::executor::KernelRuntimeContext;"""

_REG_ARRAY_HEADER = """\

// Use fully qualified namespace for kernel registration
namespace {

// Only the torch::executor umbrella is opened here: the codegen pipeline emits
// the tracer hooks unqualified (internal::EventTracerProfileOpScope,
// internal::event_tracer_log_evalue), and `internal` is a namespace under both
// torch::executor and executorch::runtime, so opening both makes that reference
// ambiguous. torch::executor re-exports the runtime symbols the array needs
// (Kernel, register_kernels, EValue, Span, Error, and internal::*).
using namespace torch::executor;

// =====================================================================
// KERNEL REGISTRATION ARRAY
// Each kernel registration is guarded by its operator define.
// =====================================================================

static Kernel kernels_to_register[] = {"""

_FOOTER = """\

// Sentinel to ensure the array has at least one element
// (Required because all operators might be #ifdef'd out)
    Kernel(nullptr, nullptr)
};

// Calculate the number of kernels (excluding the sentinel)
static constexpr size_t num_kernels = sizeof(kernels_to_register) / sizeof(Kernel) - 1;

// Explicitly convert to Span, so that the API can take an empty C array of Kernels.
static KernelSpan kernel_span(
    kernels_to_register,
    kernels_to_register + num_kernels);

// Return value not used. Keep the static variable assignment to register
// kernels in static initialization time.
static auto success_with_kernel_reg = (num_kernels > 0) ? register_kernels(kernel_span) : Error::Ok;

} // namespace"""


def _decls_body(groups: list) -> str:
    out: list = []
    for group in groups:
        out.append("")
        out.append(f"#ifdef {group.guard}")
        out.extend(group.forward_decls)
        out.append("#endif")
    return "\n".join(out)


def _decls_section(title: str, groups: list) -> str:
    return "\n".join(["", _BANNER, f"// {title}", _BANNER, _decls_body(groups)])


def _reg_section(title: str, groups: list) -> str:
    out: list = ["", _BANNER, f"// {title}", _BANNER]
    for group in groups:
        out.append("")
        out.append(f"#ifdef {group.guard}")
        out.extend(group.registrations)
        out.append(f"#endif // {group.guard}")
    return "\n".join(out)


def generate_register_all_kernels(
    portable_groups: list,
    quantized_groups: list,
    cortex_m_groups: list,
) -> str:
    """Assemble the complete RegisterAllKernels.cpp from the rendered groups."""
    sections = [_FILE_HEADER]

    # Forward declarations -- Portable + Quantized in torch::executor::native.
    sections.append(
        _decls_section("PORTABLE OPERATOR FORWARD DECLARATIONS", portable_groups)
    )
    sections.append(
        _decls_section("QUANTIZED OPERATOR FORWARD DECLARATIONS", quantized_groups)
    )
    sections.append(
        "\n} // namespace native\n} // namespace executor\n} // namespace torch"
    )

    # Forward declarations -- Cortex-M in cortex_m::native.
    if cortex_m_groups:
        sections.append(_CORTEX_M_DECLS_HEADER)
        sections.append(_decls_body(cortex_m_groups))
        sections.append("\n} // namespace native\n} // namespace cortex_m")

    # Registration array.
    sections.append(_REG_ARRAY_HEADER)
    sections.append(_reg_section("PORTABLE OPERATORS", portable_groups))
    sections.append(_reg_section("QUANTIZED OPERATORS", quantized_groups))
    if cortex_m_groups:
        sections.append(_reg_section("CORTEX-M OPERATORS", cortex_m_groups))
    sections.append(_FOOTER)

    return "\n".join(sections) + "\n"


# ===========================================================================
# Reconciliation: registration #ifdef set must match component #define set
# ===========================================================================


def _reconcile(
    unmatched_yaml: list, unmatched_source: list, allow_mismatch: bool
) -> None:
    problems: list = []
    if unmatched_yaml:
        problems.append(
            f"{len(unmatched_yaml)} YAML operator(s) with no staged op_*.cpp "
            "source (cannot be registered):"
        )
        for category, op_name, guard in unmatched_yaml:
            problems.append(f"    [{category}] {op_name} -> {guard}")
    if unmatched_source:
        problems.append(
            f"{len(unmatched_source)} staged source component(s) with no YAML "
            "registration (#define without #ifdef):"
        )
        for guard in unmatched_source:
            problems.append(f"    {guard}")
    if not problems:
        return
    message = "Operator/component reconciliation mismatch:\n" + "\n".join(problems)
    if allow_mismatch:
        print("WARNING: " + message, file=sys.stderr)
    else:
        raise SystemExit(
            message
            + "\n\nThe registration #ifdef set and the component #define set must "
            "match. Fix the YAML/source, or pass --allow-mismatch to downgrade "
            "this to a warning."
        )


# ===========================================================================
# Main
# ===========================================================================


def build_groups(source_dir: Path, allow_mismatch: bool = False) -> dict:
    """Parse, render and reconcile every category.

    Returns category -> groups.

    """
    aten_yaml, tags_yaml = _torchgen_yaml()
    selector = SelectiveBuilder.get_nop_selector()

    components = discover_components(source_dir)
    source_index = kernel_source_index(source_dir, components)

    cat_groups: dict = {}
    unmatched_yaml: list = []

    functions_yaml = _find_yaml(source_dir, "kernels/portable/functions.yaml")
    custom_yaml = _find_yaml(source_dir, "kernels/portable/custom_ops.yaml")
    if functions_yaml or custom_yaml:
        nfs, index = _parse_portable(
            functions_yaml, custom_yaml, aten_yaml, tags_yaml, selector
        )
        groups, unmatched = render_category(
            "Portable", nfs, index, source_index, selector
        )
        cat_groups["Portable"] = groups
        unmatched_yaml += unmatched

    for category, rel in (
        ("Quantized", "kernels/quantized/quantized.yaml"),
        ("Cortex-M", "backends/cortex_m/ops/operators.yaml"),
    ):
        yaml_path = _find_yaml(source_dir, rel)
        if not yaml_path:
            continue
        nfs, index = _parse_custom(yaml_path, tags_yaml, selector)
        groups, unmatched = render_category(
            category, nfs, index, source_index, selector
        )
        cat_groups[category] = groups
        unmatched_yaml += unmatched

    emitted = {g.guard for groups in cat_groups.values() for g in groups}
    unmatched_source = sorted({c.guard for c in components} - emitted)
    _reconcile(unmatched_yaml, unmatched_source, allow_mismatch)

    return cat_groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RegisterAllKernels.cpp for the ExecuTorch CMSIS Pack"
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        required=True,
        help="Pack staging dir containing kernels/ and backends/ sources + YAML",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output path for RegisterAllKernels.cpp"
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Downgrade registration/component reconciliation mismatches to warnings",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    cat_groups = build_groups(source_dir, allow_mismatch=args.allow_mismatch)

    output = generate_register_all_kernels(
        cat_groups.get("Portable", []),
        cat_groups.get("Quantized", []),
        cat_groups.get("Cortex-M", []),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output)

    num_regs = sum(
        len(g.registrations) for groups in cat_groups.values() for g in groups
    )
    num_guards = sum(len(groups) for groups in cat_groups.values())
    print(f"Wrote {output_path}: {num_regs} registrations across {num_guards} guards")


if __name__ == "__main__":
    main()
