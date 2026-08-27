#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate CMSIS Pack component definitions for ExecuTorch operators.

This script analyzes the ExecuTorch source tree and generates:
1. Individual component definitions for each operator
2. Conditions for operator dependencies
3. File lists for each component

Usage:
    python generate_components.py --source-dir /path/to/ai_layer/engine --output components.xml

"""

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from op_guards import (  # type: ignore[import-not-found]
    category_to_id,
    discover_components,
)


@dataclass
class OperatorComponent:
    """Represents a single operator component."""

    name: str  # e.g., "add", "mul", "conv2d"
    category: str  # e.g., "Portable", "Quantized"
    source_file: Path  # Path to the op_*.cpp file
    condition_id: str  # e.g., "op_add"
    guard: str  # e.g., "RTE_ML_EXECUTORCH_OP_PORTABLE_ADD"
    dependencies: list = field(default_factory=list)  # Other operators this depends on


def sanitize_condition_id(name: str, category: str = "") -> str:
    """Create a valid XML ID from operator name."""
    # Remove leading underscores and replace special chars
    name = name.lstrip("_")
    # Include category prefix to avoid duplicates between portable/quantized
    if category:
        return f"op_{category_to_id(category)}_{name}"
    return f"op_{name}"


def _op_uses_cmsis_nn(cpp_path: Path) -> bool:
    """True if the op_*.cpp source links against the ARM::CMSIS-NN pack.

    Detection signals (any one is sufficient):
      - direct #include of <arm_nnfunctions.h> / <arm_nn_types.h> /
        <cmsis_nn.h>
      - direct #include of "cortex_m_ops_common.h" (which itself pulls in
        the CMSIS-NN headers and is the canonical entry point for every
        op that uses arm_*** APIs)
      - direct call to any arm_nn* / arm_cmsis_nn_* / arm_softmax_s8 /
        arm_elementwise_* etc. symbol — catches sources that go through
        a different shared header in the future
    """
    try:
        text = cpp_path.read_text()
    except OSError:
        return False
    if re.search(
        r"#include[^\n]*(arm_nnfunctions|arm_nn_types|cmsis_nn|cortex_m_ops_common)",
        text,
    ):
        return True
    if re.search(
        r"\barm_(nn|cmsis_nn|elementwise|convolve|depthwise|avgpool|max_pool|softmax|fully_connected|batch_matmul|pad|transpose|minimum|maximum)\w*\s*\(",
        text,
    ):
        return True
    return False


def generate_operator_condition(op: OperatorComponent, relative_base: str = "") -> str:
    """Generate XML condition for an operator."""
    requires = ['<require condition="Kernel Utils"/>']
    for dep in op.dependencies:
        requires.append(f'<require condition="{dep}"/>')
    require_xml = "\n      ".join(requires)
    condition_xml = f"""    <condition id="{op.condition_id}">
      <description>Operator: {op.name}</description>
      {require_xml}
    </condition>"""
    return condition_xml


def generate_operator_component(
    op: OperatorComponent, version_placeholder: str, relative_base: str = ""
) -> str:
    """Generate XML component definition for an operator."""
    source_path = str(op.source_file).replace("\\", "/")

    define_name = op.guard

    component_xml = f"""    <component Cclass="Machine Learning" Cgroup="ExecuTorch Operators" Csub="{op.category} {op.name}" Cversion="{version_placeholder}" condition="{op.condition_id}">
      <description>ExecuTorch {op.category} Operator: {op.name}</description>
      <RTE_Components_h>
        #define {define_name}     /* ExecuTorch op_{op.name} */
      </RTE_Components_h>
      <files>
        <file category="sourceCpp" name="{source_path}"/>
      </files>
    </component>"""
    return component_xml


def generate_all_components(
    source_dir: Path, config: dict, version: str = "1.0.0"
) -> dict:
    """Generate all component definitions from source directory."""

    # Operator discovery + guard naming are shared with the registration
    # generator via op_guards.discover_components so the component #define set
    # and the registration #ifdef set cannot drift.
    ops_by_category: dict = {"Portable": [], "Quantized": [], "Cortex-M": []}
    for component in discover_components(source_dir):
        deps = []
        if component.category == "Cortex-M" and _op_uses_cmsis_nn(
            source_dir / component.source_file
        ):
            deps.append("CMSIS-NN")
        ops_by_category[component.category].append(
            OperatorComponent(
                name=component.name,
                category=component.category,
                source_file=component.source_file,
                condition_id=sanitize_condition_id(component.name, component.category),
                guard=component.guard,
                dependencies=deps,
            )
        )

    portable_ops = ops_by_category["Portable"]
    quantized_ops = ops_by_category["Quantized"]
    cortex_m_ops = ops_by_category["Cortex-M"]
    print(f"Found {len(portable_ops)} portable operators")
    print(f"Found {len(quantized_ops)} quantized operators")
    print(f"Found {len(cortex_m_ops)} cortex_m operators")

    # Generate conditions
    conditions = []
    for op in portable_ops + quantized_ops + cortex_m_ops:
        conditions.append(generate_operator_condition(op))

    # Generate components with actual version
    portable_components = [
        generate_operator_component(op, version) for op in portable_ops
    ]
    quantized_components = [
        generate_operator_component(op, version) for op in quantized_ops
    ]
    cortex_m_components = [
        generate_operator_component(op, version) for op in cortex_m_ops
    ]

    return {
        "conditions": "\n".join(conditions),
        "portable_components": "\n\n".join(portable_components),
        "quantized_components": "\n\n".join(quantized_components),
        "cortex_m_components": "\n\n".join(cortex_m_components),
        "portable_ops": portable_ops,
        "quantized_ops": quantized_ops,
        "cortex_m_ops": cortex_m_ops,
    }


def generate_runtime_files(source_dir: Path) -> str:
    """Generate file list for runtime component."""
    files = []

    # Try new structure first (src/runtime/...), fall back to old (runtime/...)
    if (source_dir / "src" / "runtime").exists():
        runtime_dirs = ["src/runtime"]
        schema_dir = source_dir / "src" / "schema"
    else:
        runtime_dirs = ["runtime"]
        schema_dir = source_dir / "schema"

    for runtime_dir in runtime_dirs:
        dir_path = source_dir / runtime_dir
        if dir_path.exists():
            for cpp_file in sorted(dir_path.rglob("*.cpp")):
                path_str = str(cpp_file)
                # Exclude test files and testing utilities
                if "/test/" in path_str or "_test.cpp" in path_str:
                    continue
                # Exclude testing_util directory (requires gmock/gtest)
                if "/testing_util/" in path_str:
                    continue
                # Exclude ATen-specific files (requires full PyTorch)
                # tensor_util_aten.cpp and tensor_parser_aten.cpp need full ATen
                # but tensor_parser_exec_aten.cpp is safe (exec_aten != ATen)
                basename = cpp_file.name
                if basename.endswith("_aten.cpp") and "exec_aten" not in basename:
                    continue
                # Exclude platform-specific files that don't work on bare-metal
                if "/platform/default/" in path_str:
                    filename = cpp_file.name
                    # Only include minimal.cpp from the default platform dir
                    # posix.cpp — needs std::chrono (fails on AC6 bare-metal)
                    # zephyr.cpp — Zephyr RTOS only
                    # android.cpp — Android only
                    # arm_zephyr.cpp — ARM Zephyr only
                    if filename not in ["minimal.cpp"]:
                        continue
                rel_path = cpp_file.relative_to(source_dir)
                files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')

    # Add schema files
    if schema_dir.exists():
        for cpp_file in sorted(schema_dir.rglob("*.cpp")):
            rel_path = cpp_file.relative_to(source_dir)
            files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')

    return "\n".join(files)


def generate_shared_kernel_files(source_dir: Path) -> str:
    """File list for shared kernel implementation sources that are required
    across operators but are not themselves op_*.cpp, so no operator component
    owns them.

    The pack models "one component == one op_*.cpp"; a kernel helper that lives
    at the top of a cpu/ dir (not in util/ or pattern/, which Kernel Utils
    already bundles) therefore falls through every net. e.g.
    kernels/quantized/cpu/embeddingxb.cpp holds the bit-width-generic
    quantized_embedding_xbit_out / _dtype_out that op_embedding2b.cpp and
    op_embedding4b.cpp delegate to; without it those components compile but fail
    to link with undefined references. These sources are bundled into the
    always-present Runtime component so the symbols are available whenever a
    dependent operator is selected.

    """
    if (source_dir / "src" / "kernels").exists():
        kernels_base = source_dir / "src" / "kernels"
    else:
        kernels_base = source_dir / "kernels"

    files = []
    for cpu_subdir in ("portable/cpu", "quantized/cpu"):
        cpu_dir = kernels_base / cpu_subdir
        if not cpu_dir.exists():
            continue
        for cpp_file in sorted(cpu_dir.glob("*.cpp")):
            if cpp_file.name.startswith("op_") or "_test.cpp" in cpp_file.name:
                continue
            rel_path = cpp_file.relative_to(source_dir)
            files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')

    return "\n".join(files)


def generate_kernel_utils_files(source_dir: Path) -> str:
    """Generate file list for kernel utils component.

    Includes shared implementation helpers transitively required by
    operator components via the "Kernel Utils" condition:

    - kernels/portable/cpu/util/*.cpp — broadcast/reduce/copy helpers
      shared by both portable and quantized operators.
    - kernels/portable/cpu/pattern/*.cpp — unary-ufunc pattern helpers
      used by many op_*.cpp via DEFINE_UNARY_UFUNC_* macros in
      pattern/pattern.h. Without these, consumers that select unary ops
      (acos, tanh, bitwise_*, logical_*, eq, le, etc.) get undefined
      references to `internal::unary_ufunc_*`.

    """
    files = []

    # Try new structure first (src/kernels/...), fall back to old (kernels/...)
    if (source_dir / "src" / "kernels").exists():
        cpu_dir = source_dir / "src" / "kernels" / "portable" / "cpu"
    else:
        cpu_dir = source_dir / "kernels" / "portable" / "cpu"

    for subdir_name in ("util", "pattern"):
        subdir = cpu_dir / subdir_name
        if not subdir.exists():
            continue
        for cpp_file in sorted(subdir.glob("*.cpp")):
            if "_test.cpp" in cpp_file.name:
                continue
            rel_path = cpp_file.relative_to(source_dir)
            files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')

    return "\n".join(files)


def generate_backend_files(
    source_dir: Path,
    backend: str,
    extra_exclude: Optional[list] = None,
) -> str:
    """Generate file list for backend component.

    `extra_exclude` is a list of filename substrings to exclude in addition
    to the backend's built-in exclusions. Used by the ethos_u backend to
    keep the Cortex-A/Linux host TU out of the Cortex-M host component;
    the two host TUs define the same symbols and cannot be linked
    together.

    """
    files = []
    extra_exclude = extra_exclude or []

    # Try new structure first (src/backends/...), fall back to old (backends/...)
    if (source_dir / "src" / "backends").exists():
        backends_base = source_dir / "src" / "backends"
    else:
        backends_base = source_dir / "backends"

    if backend == "ethos_u":
        backend_dir = backends_base / "arm" / "runtime"
        # Exclude VGF* files (Vulkan Graphics Framework) - not supported on bare metal
        exclude_patterns = ["VGF"]
    elif backend == "cortex_m":
        backend_dir = backends_base / "cortex_m"
        exclude_patterns = []
    else:
        return ""

    exclude_patterns = exclude_patterns + list(extra_exclude)

    if backend_dir.exists():
        for cpp_file in sorted(backend_dir.rglob("*.cpp")):
            # Skip excluded patterns
            if any(pattern in cpp_file.name for pattern in exclude_patterns):
                continue
            rel_path = cpp_file.relative_to(source_dir)
            files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')

    return "\n".join(files)


def generate_extension_tensor_files(source_dir: Path) -> str:
    """Generate the file list for the Tensor extension component.

    Ships extension/tensor's two sources (tensor_ptr.cpp, tensor_ptr_maker.cpp).
    The headers reach consumers through the Runtime component's blanket include/
    directory, so only the sources are listed here.

    """
    if (source_dir / "src" / "extension").exists():
        extension_base = source_dir / "src" / "extension"
    else:
        extension_base = source_dir / "extension"

    tensor_dir = extension_base / "tensor"
    files = []
    if tensor_dir.exists():
        for cpp_file in sorted(tensor_dir.rglob("*.cpp")):
            path_str = str(cpp_file)
            if "/test/" in path_str or "_test.cpp" in cpp_file.name:
                continue
            rel_path = cpp_file.relative_to(source_dir)
            files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')

    return "\n".join(files)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMSIS Pack components for ExecuTorch operators"
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        required=True,
        help="Path to ai_layer/engine source directory",
    )
    parser.add_argument("--config", "-c", help="Path to executorch_config.yml")
    parser.add_argument(
        "--output",
        "-o",
        default="components.xml",
        help="Output file for generated components",
    )
    parser.add_argument(
        "--template", "-t", help="Path to PDSC template file to populate"
    )
    parser.add_argument("--pdsc-output", help="Output path for populated PDSC file")
    parser.add_argument("--version", default="0.6.0", help="Pack version")
    parser.add_argument("--date", help="Release date (YYYY-MM-DD)")

    args = parser.parse_args()
    source_dir = Path(args.source_dir)

    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Generate components with version
    result = generate_all_components(source_dir, config, args.version)

    # Generate runtime files. Shared kernel implementation helpers that no
    # op_*.cpp component owns (e.g. kernels/quantized/cpu/embeddingxb.cpp) are
    # bundled into the always-present Runtime component so dependent operators
    # link.
    runtime_files = generate_runtime_files(source_dir)
    shared_kernel_files = generate_shared_kernel_files(source_dir)
    if shared_kernel_files:
        runtime_files = f"{runtime_files}\n{shared_kernel_files}"

    # Generate kernel utils files
    kernel_utils_files = generate_kernel_utils_files(source_dir)

    # Generate backend files.
    # Cortex-M host only. EthosUBackend_Cortex_A.cpp is filtered out
    # because the Cortex-A/Linux host variant is intentionally not
    # exposed in this pack.
    ethos_u_cortex_m_files = generate_backend_files(
        source_dir, "ethos_u", extra_exclude=["Cortex_A"]
    )

    # Generate extension files (opt-in components that only need the Runtime).
    tensor_extension_files = generate_extension_tensor_files(source_dir)

    # Output summary
    print("\nGenerated components:")
    print(f"  - {len(result['portable_ops'])} portable operators")
    print(f"  - {len(result['quantized_ops'])} quantized operators")
    print(f"  - {len(result['cortex_m_ops'])} cortex_m operators")

    # Write component definitions to output file
    with open(args.output, "w") as f:
        f.write("<!-- Operator Conditions -->\n")
        f.write(result["conditions"])
        f.write("\n\n<!-- Portable Operator Components -->\n")
        f.write(result["portable_components"])
        f.write("\n\n<!-- Quantized Operator Components -->\n")
        f.write(result["quantized_components"])
        f.write("\n\n<!-- Cortex-M Operator Components -->\n")
        f.write(result["cortex_m_components"])

    print(f"\nComponent definitions written to: {args.output}")

    # If template provided, populate it
    if args.template and args.pdsc_output:
        with open(args.template) as f:
            template = f.read()

        import datetime

        release_date = args.date or datetime.date.today().strftime("%Y-%m-%d")

        # Replace placeholders
        pdsc = template
        pdsc = pdsc.replace("%{RELEASE_VERSION}%", args.version)
        pdsc = pdsc.replace("%{RELEASE_DATE}%", release_date)
        pdsc = pdsc.replace("%{HISTORY}%", "")
        pdsc = pdsc.replace("%{OPERATOR_CONDITIONS}%", result["conditions"])
        pdsc = pdsc.replace("%{RUNTIME_FILES}%", runtime_files)
        pdsc = pdsc.replace("%{KERNEL_UTILS_FILES}%", kernel_utils_files)
        pdsc = pdsc.replace(
            "%{PORTABLE_OPERATOR_COMPONENTS}%", result["portable_components"]
        )
        pdsc = pdsc.replace(
            "%{QUANTIZED_OPERATOR_COMPONENTS}%", result["quantized_components"]
        )
        pdsc = pdsc.replace(
            "%{CORTEX_M_OPERATOR_COMPONENTS}%", result["cortex_m_components"]
        )
        pdsc = pdsc.replace(
            "%{ETHOS_U_BACKEND_CORTEX_M_FILES}%", ethos_u_cortex_m_files
        )
        pdsc = pdsc.replace("%{TENSOR_EXTENSION_FILES}%", tensor_extension_files)

        with open(args.pdsc_output, "w") as f:
            f.write(pdsc)

        print(f"PDSC file written to: {args.pdsc_output}")


if __name__ == "__main__":
    main()
