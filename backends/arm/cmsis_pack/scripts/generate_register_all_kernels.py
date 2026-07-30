#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate RegisterAllKernels.cpp from ExecuTorch operator YAML definitions and
source files.

This script reads the operator YAML files (functions.yaml, custom_ops.yaml, quantized.yaml)
to discover all operators, then extracts their function signatures from the C++ source files,
and generates a complete RegisterAllKernels.cpp with #ifdef guards for each operator.

Each operator is guarded by RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME> which is defined
by the CMSIS Pack component system via RTE_Components.h.

Usage:
    python generate_register_all_kernels.py \
        --source-dir /path/to/output \
        --output /path/to/output/src/registration/RegisterAllKernels.cpp

"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# ===========================================================================
# Data model
# ===========================================================================


@dataclass
class Param:
    """One parameter of an operator function."""

    type: str  # e.g. "const Tensor&", "int64_t", "Scalar"
    name: str  # e.g. "self", "dim", "out"


@dataclass
class OpOverload:
    """One overload of an ATen / custom operator."""

    aten_name: str  # e.g. "aten::add.out"
    kernel_name: str  # e.g. "torch::executor::add_out"
    short_kernel: str  # e.g. "add_out" (just function name)
    params: list  # list[Param] excluding KernelRuntimeContext
    return_type: str  # "Tensor&", "void", "std::tuple<Tensor&, Tensor&>", etc.
    namespace: str  # "aten" or "dim_order_ops" or "quantized_decomposed" or "cortex_m"
    # C++ namespace where the kernel is implemented. Defaults to the
    # ExecuTorch ATen-shim "torch::executor::native"; cortex_m kernels
    # live in "cortex_m::native".
    kernel_namespace: str = "torch::executor::native"


@dataclass
class OpGroup:
    """All overloads that share one #ifdef guard (same source file /
    component).
    """

    guard_define: str  # e.g. "RTE_ML_EXECUTORCH_OP_PORTABLE_ADD"
    overloads: list  # list[OpOverload]
    category: str  # "Portable", "Quantized", "Cortex-M"


# ===========================================================================
# YAML parsing
# ===========================================================================


def parse_functions_yaml(yaml_path: Path) -> list[dict]:
    """Parse a functions.yaml or custom_ops.yaml or quantized.yaml file.

    Returns a list of dicts with keys: op_name, kernel_name, namespace, func_schema (optional).

    """
    with open(yaml_path) as f:
        entries = yaml.safe_load(f)

    if entries is None:
        return []

    results = []
    for entry in entries:
        if "op" in entry:
            # Standard ATen op: "- op: add.out" with "kernel_name: torch::executor::add_out"
            op_name = entry["op"]
            namespace = "aten"
            func_schema = None
        elif "func" in entry:
            # Custom op with full schema: "- func: quantized_decomposed::add.out(...) -> ..."
            func_str = entry["func"]
            # Parse "namespace::op_name(args) -> ret"
            func_schema = func_str
            match = re.match(r"(\w+(?:::\w+)*)\.?(\w*)\(", func_str)
            if match:
                full_name = match.group(0).rstrip("(")
                # Split namespace and op
                if "::" in full_name:
                    parts = full_name.rsplit("::", 1)
                    namespace = parts[0]
                    op_name = parts[1]
                else:
                    namespace = "aten"
                    op_name = full_name
            else:
                continue
        else:
            continue

        kernels = entry.get("kernels", [])
        if not kernels:
            continue

        kernel_name = kernels[0].get("kernel_name", "")
        if not kernel_name:
            continue

        results.append(
            {
                "op_name": op_name,
                "kernel_name": kernel_name,
                "namespace": namespace,
                "func_schema": func_schema,
            }
        )

    return results


# ===========================================================================
# C++ source parsing – extract function signatures
# ===========================================================================


def read_file_text(path: Path) -> str:
    """Read entire file as text."""
    with open(path, "r") as f:
        return f.read()


# Set of known underscore-prefixed kernel function names that are valid public APIs
ALLOWED_UNDERSCORE_FUNCS = {
    "_native_batch_norm_legit_out",
    "_native_batch_norm_legit_no_stats_out",
    "_native_batch_norm_legit_no_training_out",
    "_cdist_forward_out",
    "_pdist_forward_out",
    "_upsample_bilinear2d_aa_out",
    "_empty_dim_order_out",
    "_to_dim_order_copy_out",
    "_clone_dim_order_out",
}


def extract_function_signatures(
    source_dir: Path, subdir: str
) -> tuple[dict[str, list], dict[str, str]]:
    """Extract all exported function signatures from op_*.cpp files.

    Returns: tuple of:
      - dict mapping short function name -> list of (return_type, params_list)
        where params_list is list[Param].
      - dict mapping short function name -> source file base name
        e.g. 'quantize_per_tensor_out' -> 'quantize' (from op_quantize.cpp)

    """
    sigs: dict[str, list] = {}
    func_to_source: dict[str, str] = (
        {}
    )  # function_name -> source file base (e.g. 'quantize')
    op_dir = source_dir / subdir
    if not op_dir.exists():
        return sigs, func_to_source

    for cpp_file in sorted(op_dir.glob("op_*.cpp")):
        # Extract the source file base name: op_quantize.cpp -> 'quantize'
        source_base = re.match(r"op_(.+)\.cpp$", cpp_file.name)
        source_base = source_base.group(1) if source_base else cpp_file.stem

        text = read_file_text(cpp_file)
        # Remove single-line comments to avoid false matches
        text_clean = re.sub(r"//[^\n]*", "", text)
        # Remove multi-line comments
        text_clean = re.sub(r"/\*.*?\*/", "", text_clean, flags=re.DOTALL)

        # Detect unary ops defined via DEFINE_UNARY_UFUNC macros
        # These expand to: Tensor& op_name(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)
        macro_pattern = r"DEFINE_UNARY_UFUNC_\w+\s*\(\s*(\w+)\s*,"
        for m in re.finditer(macro_pattern, text_clean):
            func_name = m.group(1)
            params = [
                Param(type="const Tensor&", name="in"),
                Param(type="Tensor&", name="out"),
            ]
            if func_name not in sigs:
                sigs[func_name] = []
            sigs[func_name].append(("Tensor&", params))
            func_to_source[func_name] = source_base

        # Find function signatures in the torch::executor::native namespace
        # Pattern: Look for functions defined after entering namespace native
        # We look for return_type function_name( ... ) {
        # The return_type can be:
        #   Tensor&
        #   Tensor
        #   void
        #   std::tuple<Tensor&, Tensor&>
        #   std::tuple<Tensor&, Tensor&, Tensor&>
        #   ::std::tuple<Tensor&, Tensor&>  (note leading ::)

        # Multi-line regex: match function definitions
        # Allow return type and function name to be on separate lines
        pattern = r"""
            (?:^|\n)\s*                                          # start of line
            ((?:::?std::tuple<[^>]+>|std::tuple<[^>]+>|Tensor&?|void)\s*)  # return type (with optional leading ::)
            \n?\s*                                                # possible newline
            (\w+)\s*\(                                           # function name and opening paren
            ([^)]*(?:\([^)]*\)[^)]*)*?)                          # parameters (handling nested parens)
            \)\s*\{                                               # closing paren and opening brace
        """

        for m in re.finditer(pattern, text_clean, re.VERBOSE | re.MULTILINE):
            ret_type = m.group(1).strip()
            func_name = m.group(2).strip()
            params_str = m.group(3).strip()

            # Normalize return type (strip leading ::)
            if ret_type.startswith("::"):
                ret_type = ret_type[2:]

            # Skip internal/helper functions (not in the public API)
            # We only want functions that are registered as kernels
            if func_name.startswith("_") and func_name not in ALLOWED_UNDERSCORE_FUNCS:
                continue

            # Parse parameters
            params = parse_params(params_str)
            if params is None:
                continue

            # Skip overloads without KernelRuntimeContext or RuntimeContext (non-ctx wrappers)
            has_ctx = any("RuntimeContext" in p.type for p in params)
            if not has_ctx:
                continue

            # Remove the context parameter from params
            params = [p for p in params if "RuntimeContext" not in p.type]

            if func_name not in sigs:
                sigs[func_name] = []
            sigs[func_name].append((ret_type, params))
            func_to_source[func_name] = source_base

    return sigs, func_to_source


def parse_params(params_str: str) -> list:  # noqa: C901
    """Parse a parameter string into list of Param objects."""
    if not params_str.strip():
        return []

    params = []
    # Split on commas, but handle template types like std::optional<Tensor>
    # and nested types like ArrayRef<int64_t>
    depth = 0
    current = []
    for char in params_str:
        if char in ("<", "("):
            depth += 1
            current.append(char)
        elif char in (">", ")"):
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        params.append("".join(current).strip())

    result = []
    for p in params:
        p = p.strip()
        if not p:
            continue
        # Remove ET_UNUSED annotation
        p = p.replace("ET_UNUSED ", "").replace("ET_UNUSED\n", "")
        # Parse "type name" or "type name = default"
        # Remove default value
        p = re.sub(r"\s*=\s*[^,]*$", "", p)
        # Split into type and name - name is the last token
        tokens = p.rsplit(None, 1)
        if len(tokens) == 2:
            ptype = tokens[0].strip()
            pname = tokens[1].strip()
            # Handle cases like "const Tensor&" where & is on the name
            if pname.startswith("&"):
                ptype += "&"
                pname = pname[1:]
            result.append(Param(type=ptype, name=pname))
        elif len(tokens) == 1:
            # Single token - might be just a type (in template specialization etc)
            continue

    return result


# ===========================================================================
# Mapping operators to source files and #ifdef guards
# ===========================================================================


def op_name_to_source_file(op_name: str) -> str:
    """Map an op name like "add.out" or "add.Scalar_out" to a source file
    basename.

    The source file is typically op_<base_name>.cpp where base_name is derived
    from the op name before the first dot.

    """
    # Get the base operator name (before the overload suffix)
    base = op_name.split(".")[0] if "." in op_name else op_name
    # Special mappings
    special = {
        "_cdist_forward": "cdist_forward",
        "_log_softmax": "log_softmax",
        "_softmax": "softmax",
        "_to_copy": "to_copy",
        "_pdist_forward": "pdist_forward",
        "_native_batch_norm_legit": "native_batch_norm",
        "_native_batch_norm_legit_no_training": "native_batch_norm",
        "_upsample_bilinear2d_aa": "upsample_bilinear2d_aa",
    }
    if base in special:
        base = special[base]
    return base


def op_name_to_guard(base_name: str, category: str) -> str:
    """Map a base operator name (source file) to its #ifdef guard define.

    The guard follows the convention: RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>
    where NAME is the uppercased source file base (matches what generate_components.py creates).
    Display-form categories with hyphens or spaces (e.g. "Cortex-M") are
    normalized to identifier-safe tokens ("CORTEX_M") to keep the C macro
    valid.

    """
    # Strip leading underscores (matching sanitize_component_name in generate_components.py)
    clean = base_name.lstrip("_")
    cat = category.upper().replace("-", "_").replace(" ", "_")
    return f"RTE_ML_EXECUTORCH_OP_{cat}_{clean.upper()}"


def kernel_short_name(kernel_name: str) -> str:
    """Extract short function name from full kernel name.

    e.g. "torch::executor::add_out" -> "add_out"

    """
    return kernel_name.rsplit("::", 1)[-1]


# ===========================================================================
# EValue unpacking code generation
# ===========================================================================


def get_evalue_type(param_type: str) -> str:
    """Map a C++ parameter type to its EValue extraction expression."""
    # Normalize the type
    t = param_type.strip()
    # Remove const and reference
    t_base = t.replace("const ", "").replace("&", "").strip()

    # Handle common types
    if t_base in ("Tensor", "Tensor&"):
        if "const" in param_type:
            return "to<Tensor>"
        return "to<Tensor>"

    if t_base == "Scalar":
        return "to<Scalar>"

    if t_base == "int64_t":
        return "to<int64_t>"

    if t_base == "double":
        return "to<double>"

    if t_base == "float":
        return "to<double>"  # float comes as double from schema

    if t_base == "bool":
        return "to<bool>"

    if t_base == "ScalarType":
        return "to<ScalarType>"

    return ""  # Complex types needing special handling


def needs_special_unpacking(param_type: str) -> bool:
    """Check if a parameter type needs special unpacking (not simple EValue::to<T>())."""
    t = param_type.strip()
    special_patterns = [
        "IntArrayRef",
        "ArrayRef",
        "TensorList",
        "TensorOptList",
        "optional",
        "Optional",
        "string_view",
        "MemoryFormat",
        "Span",
    ]
    return any(p in t for p in special_patterns)


def generate_param_unpack(  # noqa: C901
    param: Param, stack_idx: int
) -> tuple[str, str]:
    """Generate code to unpack a parameter from the EValue stack.

    Returns (variable_declaration_code, variable_name_for_call).

    """
    t = param.type.strip()
    name = param.name

    # Remove const and & for base type checking
    t_clean = t.replace("const ", "").replace("&", "").strip()

    # --- Simple scalar types ---
    if t_clean in ("int64_t",):
        return (
            f"            int64_t {name}_base = {name}.to<int64_t>();",
            f"{name}_base",
        )

    if t_clean in ("double",):
        return (
            f"            double {name}_base = {name}.to<double>();",
            f"{name}_base",
        )

    if t_clean in ("float",):
        return (
            f"            double {name}_base = {name}.to<double>();",
            f"{name}_base",
        )

    if t_clean in ("bool",):
        return (f"            bool {name}_base = {name}.to<bool>();", f"{name}_base")

    if t_clean == "ScalarType":
        return (
            f"            ScalarType {name}_base = {name}.to<ScalarType>();",
            f"{name}_base",
        )

    # --- Tensor types ---
    if t_clean == "Tensor" and "const" in t:
        return (
            f"            const Tensor& {name}_base = {name}.to<Tensor>();",
            f"{name}_base",
        )

    if t_clean == "Tensor":
        # Non-const Tensor& (output tensor)
        return (
            f"            Tensor& {name}_base = {name}.to<Tensor>();",
            f"{name}_base",
        )

    if t_clean == "Scalar":
        return (
            f"            const Scalar& {name}_base = {name}.to<Scalar>();",
            f"{name}_base",
        )

    # --- Array types ---
    # --- Optional Array types (must come before non-optional to match first) ---
    if "OptionalIntArrayRef" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<IntArrayRef>();",
            f"{name}_opt_out",
        )

    if "OptIntArrayRef" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<IntArrayRef>();",
            f"{name}_opt_out",
        )

    # OptionalArrayRef<int64_t> — distinct upstream spelling from
    # OptionalIntArrayRef / OptIntArrayRef / optional<ArrayRef<int64_t>>.
    # Same wire type, so route through the same EValue unpack form.
    # Substring catches both the unqualified spelling and the
    # `executorch::aten::OptionalArrayRef<int64_t>` form.
    if "OptionalArrayRef<int64_t>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<IntArrayRef>();",
            f"{name}_opt_out",
        )

    # `Int64ArrayRef` is a using-alias for `ArrayRef<int64_t>` used in
    # the Cortex-M kernel headers; semantically identical to `IntArrayRef`.
    if (
        "IntArrayRef" in t
        or "Int64ArrayRef" in t
        or ("ArrayRef<int64_t>" in t and "Optional" not in t and "optional" not in t)
    ):
        return (
            f"            auto {name}_list_out = {name}.toIntList();",
            f"{name}_list_out",
        )

    if "TensorOptList" in t:
        return (
            f"            auto {name}_list_out = {name}.toTensorOptList();",
            f"{name}_list_out",
        )

    if "TensorList" in t and "Opt" not in t:
        return (
            f"            auto {name}_list_out = {name}.toTensorList();",
            f"{name}_list_out",
        )

    if "ArrayRef<Tensor>" in t and "Opt" not in t:
        return (
            f"            auto {name}_list_out = {name}.toTensorList();",
            f"{name}_list_out",
        )

    if "ArrayRef<bool>" in t:
        return (
            f"            auto {name}_list_out = {name}.toBoolList();",
            f"{name}_list_out",
        )

    if "OptionalArrayRef<double>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<ArrayRef<double>>();",
            f"{name}_opt_out",
        )

    if "ArrayRef<double>" in t and "Optional" not in t:
        return (
            f"            auto {name}_list_out = {name}.toDoubleList();",
            f"{name}_list_out",
        )

    # --- Optional types ---
    if "optional<Tensor>" in t or "std::optional<Tensor>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<Tensor>();",
            f"{name}_opt_out",
        )

    if "optional<ScalarType>" in t or "std::optional<ScalarType>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<ScalarType>();",
            f"{name}_opt_out",
        )

    if "optional<int64_t>" in t or "std::optional<int64_t>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<int64_t>();",
            f"{name}_opt_out",
        )

    if "optional<double>" in t or "std::optional<double>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<double>();",
            f"{name}_opt_out",
        )

    if "optional<Scalar>" in t or "std::optional<Scalar>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<Scalar>();",
            f"{name}_opt_out",
        )

    # `optional<bool>` — substring catches both the unqualified spelling
    # and `torch::executor::optional<bool>` (qualifier sits before).
    if "optional<bool>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<bool>();",
            f"{name}_opt_out",
        )

    # `optional<MemoryFormat>` — the third spelling
    # `optional<executorch::aten::MemoryFormat>` has the qualifier *inside*
    # the angle brackets, so we have to list it explicitly; substring
    # alone doesn't catch it.
    if (
        "optional<MemoryFormat>" in t
        or "std::optional<MemoryFormat>" in t
        or "optional<executorch::aten::MemoryFormat>" in t
    ):
        return (
            f"            auto {name}_opt_out = {name}.toOptional<MemoryFormat>();",
            f"{name}_opt_out",
        )

    if "optional<ArrayRef<int64_t>>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<IntArrayRef>();",
            f"{name}_opt_out",
        )

    if "OptionalArrayRef<double>" in t:
        return (
            f"            auto {name}_opt_out = {name}.toOptional<ArrayRef<double>>();",
            f"{name}_opt_out",
        )

    if "string_view" in t:
        return (f"            auto {name}_sv = {name}.toStringView();", f"{name}_sv")

    # Fallback: try direct to<T>
    return (
        f"            // WARNING: auto-generated unpack for {t}\n            auto {name}_base = {name}.to<Tensor>();",
        f"{name}_base",
    )


def count_stack_args(params: list) -> int:
    """Count EValue stack slots emitted by the AOT emitter.

    AOT writes ``KernelCall.args = [...params..., out_idx]`` for every
    out-variant kernel: one slot per parsed parameter plus a trailing
    return slot whose EValue index is the same as ``out``. Mirrors
    ``len(binding_list) + 1`` in upstream ``executorch/codegen/gen.py``.

    """
    return len(params) + 1


# ===========================================================================
# Code generation
# ===========================================================================


def generate_forward_decl(overload: OpOverload) -> str:
    """Generate a forward declaration for one kernel function."""
    params_str = ", ".join(
        f"{p.type} {p.name}"
        for p in [Param("KernelRuntimeContext&", "context")] + overload.params
    )
    return f"{overload.return_type} {overload.short_kernel}({params_str});"


def generate_kernel_registration(overload: OpOverload) -> str:
    """Generate a Kernel(...) registration entry for one overload."""
    ns = overload.namespace
    if ns == "aten":
        full_op_name = f"aten::{overload.aten_name}"
    else:
        full_op_name = f"{ns}::{overload.aten_name}"

    # Clean name for profiler scope
    prof_name = overload.aten_name.replace(".", ".")
    clean_name = f"native_call_{prof_name}"

    num_stack = count_stack_args(overload.params)
    # Stack layout: stack[0..N-1] = parsed params (inputs + `out`),
    # stack[N] = return slot (same EValue as `out` for out-variants).
    # Total: count_stack_args(params) == N + 1.

    lines = []
    lines.append("    Kernel(")
    lines.append(f'        "{full_op_name}",')
    lines.append(
        "        [](torch::executor::KernelRuntimeContext & context, Span<EValue*> stack) {"
    )
    lines.append(
        f'            ET_KERNEL_CHECK_MSG(context, stack.size() == {num_stack}, InvalidProgram, /*void*/, "Expected {num_stack} args");'
    )

    # Generate EValue extractions
    unpack_lines = []
    call_args = []

    for i, param in enumerate(overload.params):
        lines.append(f"            EValue& {param.name} = *stack[{i}];")

    # Generate type conversions
    for i, param in enumerate(overload.params):
        decl, var_name = generate_param_unpack(param, i)
        unpack_lines.append(decl)
        call_args.append(var_name)

    lines.extend(unpack_lines)

    # Generate the profiler scope and function call
    lines.append(
        f'            executorch::runtime::internal::EventTracerProfileOpScope event_tracer_op_scope(context.internal_event_tracer(), "{clean_name}");'
    )
    lines.append(f'            EXECUTORCH_SCOPE_PROF("{clean_name}");')

    call_str = ", ".join(call_args)
    lines.append(
        f"            {overload.kernel_namespace}::{overload.short_kernel}(context, {call_str});"
    )

    # Log output EVvalues (for non-void return types, log the last stack entry)
    # For multi-output ops, log all output entries
    # Determine which params are outputs (Tensor& not const, or TensorList at end)
    output_indices = []
    for i, param in enumerate(overload.params):
        t = param.type.strip()
        is_output = (
            ("Tensor&" in t or "Tensor &" in t)
            and "const" not in t
            and "optional" not in t.lower()
        ) or ("TensorList" in t and i == len(overload.params) - 1)
        if is_output:
            output_indices.append(i)

    for idx in output_indices:
        lines.append(
            f"            executorch::runtime::internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[{idx}]);"
        )

    lines.append("        }")
    lines.append("    ),")

    return "\n".join(lines)


def generate_register_all_kernels(  # noqa: C901
    portable_groups: list,
    quantized_groups: list,
    cortex_m_groups: Optional[list] = None,
) -> str:
    """Generate the complete RegisterAllKernels.cpp file."""
    cortex_m_groups = cortex_m_groups or []

    sections = []

    # File header
    sections.append(
        """/*
 * CMSIS Pack Kernel Registration
 * 
 * Auto-generated by generate_register_all_kernels.py
 * 
 * This file contains all ExecuTorch kernel registrations, with each operator
 * guarded by its corresponding RTE_Components_h define.
 * 
 * When a user selects an operator component in the CMSIS pack, that component
 * defines RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>, which enables the registration 
 * of that operator in this file.
 * 
 * Define naming convention:
 *   RTE_ML_EXECUTORCH_OP_PORTABLE_<NAME>   - Portable operators
 *   RTE_ML_EXECUTORCH_OP_QUANTIZED_<NAME>  - Quantized operators
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
    )

    # Forward declarations - Portable
    sections.append("")
    sections.append(
        "// ====================================================================="
    )
    sections.append("// PORTABLE OPERATOR FORWARD DECLARATIONS")
    sections.append(
        "// ====================================================================="
    )

    for group in portable_groups:
        sections.append("")
        sections.append(f"#ifdef {group.guard_define}")
        for overload in group.overloads:
            sections.append(generate_forward_decl(overload))
        sections.append("#endif")

    # Forward declarations - Quantized
    sections.append("")
    sections.append(
        "// ====================================================================="
    )
    sections.append("// QUANTIZED OPERATOR FORWARD DECLARATIONS")
    sections.append(
        "// ====================================================================="
    )

    for group in quantized_groups:
        sections.append("")
        sections.append(f"#ifdef {group.guard_define}")
        for overload in group.overloads:
            sections.append(generate_forward_decl(overload))
        sections.append("#endif")

    # Close torch::executor::native namespace
    sections.append(
        """
} // namespace native
} // namespace executor
} // namespace torch"""
    )

    # Forward declarations - Cortex-M (in cortex_m::native namespace)
    if cortex_m_groups:
        sections.append(
            """
namespace cortex_m {
namespace native {
// =====================================================================
// CORTEX-M OPERATOR FORWARD DECLARATIONS
// =====================================================================
// Bring kernel-API types into the cortex_m::native scope so the
// forward declarations below resolve. These mirror the file-scope
// using aliases in
// backends/cortex_m/ops/cortex_m_ops_common.h, but without pulling
// in arm_nn_types.h / arm_nnfunctions.h — RegisterAllKernels.cpp
// only needs the type names, not the CMSIS-NN headers.
using Tensor = ::torch::executor::Tensor;
using ScalarType = ::executorch::aten::ScalarType;
using Int64ArrayRef = ::executorch::aten::ArrayRef<int64_t>;
using KernelRuntimeContext = ::torch::executor::KernelRuntimeContext;"""
        )
        for group in cortex_m_groups:
            sections.append("")
            sections.append(f"#ifdef {group.guard_define}")
            for overload in group.overloads:
                sections.append(generate_forward_decl(overload))
            sections.append("#endif")
        sections.append(
            """
} // namespace native
} // namespace cortex_m"""
        )

    # Anonymous registration namespace
    sections.append(
        """
// Use fully qualified namespace for kernel registration
namespace {

using namespace torch::executor;
using namespace executorch::runtime;

// =====================================================================
// KERNEL REGISTRATION ARRAY
// Each kernel registration is guarded by its operator define.
// =====================================================================

static Kernel kernels_to_register[] = {"""
    )

    # Kernel registrations - Portable
    sections.append("")
    sections.append(
        "// ====================================================================="
    )
    sections.append("// PORTABLE OPERATORS")
    sections.append(
        "// ====================================================================="
    )

    for group in portable_groups:
        sections.append("")
        sections.append(f"#ifdef {group.guard_define}")
        for overload in group.overloads:
            sections.append(generate_kernel_registration(overload))
        sections.append(f"#endif // {group.guard_define}")

    # Kernel registrations - Quantized
    sections.append("")
    sections.append(
        "// ====================================================================="
    )
    sections.append("// QUANTIZED OPERATORS")
    sections.append(
        "// ====================================================================="
    )

    for group in quantized_groups:
        sections.append("")
        sections.append(f"#ifdef {group.guard_define}")
        for overload in group.overloads:
            sections.append(generate_kernel_registration(overload))
        sections.append(f"#endif // {group.guard_define}")

    # Kernel registrations - Cortex-M
    if cortex_m_groups:
        sections.append("")
        sections.append(
            "// ====================================================================="
        )
        sections.append("// CORTEX-M OPERATORS")
        sections.append(
            "// ====================================================================="
        )

        for group in cortex_m_groups:
            sections.append("")
            sections.append(f"#ifdef {group.guard_define}")
            for overload in group.overloads:
                sections.append(generate_kernel_registration(overload))
            sections.append(f"#endif // {group.guard_define}")

    # Footer
    sections.append(
        """
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

} // namespace
"""
    )

    return "\n".join(sections)


# ===========================================================================
# Main pipeline
# ===========================================================================


def build_op_groups(
    yaml_entries: list,
    signatures: dict,
    category: str,
    func_to_source: Optional[dict] = None,
    kernel_namespace: str = "torch::executor::native",
) -> list:
    """Build OpGroup list from YAML entries and extracted signatures.

    Groups overloads by their source file (= CMSIS component = #ifdef guard).
    Uses func_to_source mapping (from extract_function_signatures) to determine
    the actual source file for each function, falling back to op_name_to_source_file.

    `kernel_namespace` is the C++ namespace where the kernel functions are
    defined (e.g. "torch::executor::native" for ATen/quantized ops, or
    "cortex_m::native" for cortex_m ops).

    """
    if func_to_source is None:
        func_to_source = {}

    # Group yaml entries by source file base name
    file_groups: dict[str, list] = {}  # base_name -> list of yaml entries

    for entry in yaml_entries:
        op_name = entry["op_name"]
        kernel_name = entry["kernel_name"]
        short = kernel_short_name(kernel_name)

        # Use actual source file mapping if available, fall back to heuristic
        if short in func_to_source:
            base = func_to_source[short]
        else:
            base = op_name_to_source_file(op_name)

        if base not in file_groups:
            file_groups[base] = []
        file_groups[base].append(entry)

    result = []
    skipped = []

    for base_name, entries in sorted(file_groups.items()):
        guard = op_name_to_guard(base_name, category)
        overloads = []

        for entry in entries:
            op_name = entry["op_name"]
            kernel_name = entry["kernel_name"]
            short = kernel_short_name(kernel_name)
            namespace = entry["namespace"]

            # Try to find the signature in extracted signatures
            if short in signatures:
                sig_list = signatures[short]
                # Take the first signature (with KernelRuntimeContext)
                ret_type, params = sig_list[0]

                overloads.append(
                    OpOverload(
                        aten_name=op_name,
                        kernel_name=kernel_name,
                        short_kernel=short,
                        params=params,
                        return_type=ret_type,
                        namespace=namespace,
                        kernel_namespace=kernel_namespace,
                    )
                )
            else:
                skipped.append((op_name, kernel_name, short))

        if overloads:
            result.append(
                OpGroup(
                    guard_define=guard,
                    overloads=overloads,
                    category=category,
                )
            )

    if skipped:
        print(
            f"  WARNING: Skipped {len(skipped)} {category} ops (signature not found):"
        )
        for op_name, _kernel_name, short in skipped:
            print(f"    - {op_name} -> {short}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate RegisterAllKernels.cpp for ExecuTorch CMSIS Pack"
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        required=True,
        help="Path to the output/ directory containing kernels/ and include/",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output path for RegisterAllKernels.cpp"
    )

    args = parser.parse_args()
    source_dir = Path(args.source_dir)

    print("=== ExecuTorch RegisterAllKernels.cpp Generator ===\n")

    # 1. Parse YAML files to get op name -> kernel name mappings
    portable_yaml = (
        source_dir
        / "include"
        / "executorch"
        / "kernels"
        / "portable"
        / "functions.yaml"
    )
    custom_yaml = (
        source_dir
        / "include"
        / "executorch"
        / "kernels"
        / "portable"
        / "custom_ops.yaml"
    )
    quantized_yaml = (
        source_dir
        / "include"
        / "executorch"
        / "kernels"
        / "quantized"
        / "quantized.yaml"
    )
    cortex_m_yaml = (
        source_dir
        / "include"
        / "executorch"
        / "backends"
        / "cortex_m"
        / "ops"
        / "operators.yaml"
    )

    print("Parsing YAML files...")
    portable_entries = []
    if portable_yaml.exists():
        portable_entries.extend(parse_functions_yaml(portable_yaml))
        print(f"  functions.yaml: {len(portable_entries)} entries")

    custom_entries = []
    if custom_yaml.exists():
        custom_entries = parse_functions_yaml(custom_yaml)
        print(f"  custom_ops.yaml: {len(custom_entries)} entries")
        # Custom ops are registered as portable (same source directory)
        portable_entries.extend(custom_entries)

    quantized_entries = []
    if quantized_yaml.exists():
        quantized_entries = parse_functions_yaml(quantized_yaml)
        print(f"  quantized.yaml: {len(quantized_entries)} entries")

    cortex_m_entries = []
    if cortex_m_yaml.exists():
        cortex_m_entries = parse_functions_yaml(cortex_m_yaml)
        print(f"  cortex_m operators.yaml: {len(cortex_m_entries)} entries")

    # 2. Extract function signatures from C++ source files
    print("\nExtracting function signatures from source files...")

    # Try both directory layouts
    if (source_dir / "src" / "kernels").exists():
        portable_sigs, portable_func_src = extract_function_signatures(
            source_dir, "src/kernels/portable/cpu"
        )
        quantized_sigs, quantized_func_src = extract_function_signatures(
            source_dir, "src/kernels/quantized/cpu"
        )
        cortex_m_sigs, cortex_m_func_src = extract_function_signatures(
            source_dir, "src/backends/cortex_m/ops"
        )
    else:
        portable_sigs, portable_func_src = extract_function_signatures(
            source_dir, "kernels/portable/cpu"
        )
        quantized_sigs, quantized_func_src = extract_function_signatures(
            source_dir, "kernels/quantized/cpu"
        )
        cortex_m_sigs, cortex_m_func_src = extract_function_signatures(
            source_dir, "backends/cortex_m/ops"
        )

    print(f"  Portable signatures found: {len(portable_sigs)}")
    print(f"  Quantized signatures found: {len(quantized_sigs)}")
    print(f"  Cortex-M signatures found: {len(cortex_m_sigs)}")

    # 3. Build op groups
    print("\nBuilding operator groups...")
    portable_groups = build_op_groups(
        portable_entries, portable_sigs, "Portable", portable_func_src
    )
    quantized_groups = build_op_groups(
        quantized_entries, quantized_sigs, "Quantized", quantized_func_src
    )
    # Cortex-M kernels live in the cortex_m::native C++ namespace, not
    # torch::executor::native.
    cortex_m_groups = build_op_groups(
        cortex_m_entries,
        cortex_m_sigs,
        "Cortex-M",
        cortex_m_func_src,
        kernel_namespace="cortex_m::native",
    )

    total_portable_ops = sum(len(g.overloads) for g in portable_groups)
    total_quantized_ops = sum(len(g.overloads) for g in quantized_groups)
    total_cortex_m_ops = sum(len(g.overloads) for g in cortex_m_groups)
    print(
        f"\n  Portable: {len(portable_groups)} groups, {total_portable_ops} overloads"
    )
    print(
        f"  Quantized: {len(quantized_groups)} groups, {total_quantized_ops} overloads"
    )
    print(f"  Cortex-M: {len(cortex_m_groups)} groups, {total_cortex_m_ops} overloads")

    # 4. Generate output
    print("\nGenerating RegisterAllKernels.cpp...")
    output = generate_register_all_kernels(
        portable_groups, quantized_groups, cortex_m_groups
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output)

    print(f"\nOutput written to: {output_path}")
    print(
        f"Total registered operators: {total_portable_ops + total_quantized_ops + total_cortex_m_ops}"
    )


if __name__ == "__main__":
    main()
