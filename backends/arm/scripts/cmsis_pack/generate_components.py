#!/usr/bin/env python3
"""
Generate CMSIS Pack component definitions for ExecuTorch operators.

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
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OperatorComponent:
    """Represents a single operator component."""
    name: str                    # e.g., "add", "mul", "conv2d"
    category: str               # e.g., "Portable", "Quantized"
    source_file: Path           # Path to the op_*.cpp file
    condition_id: str           # e.g., "op_add"
    dependencies: list = field(default_factory=list)  # Other operators this depends on
    

def extract_operator_name(filename: str) -> str:
    """Extract operator name from filename like op_add.cpp -> add."""
    match = re.match(r'op_(.+)\.cpp$', filename)
    if match:
        return match.group(1)
    return None


def sanitize_component_name(name: str) -> str:
    """Convert operator name to valid CMSIS component name."""
    # Replace underscores and special chars
    # e.g., "_to_dim_order_copy" -> "to_dim_order_copy"
    name = name.lstrip('_')
    # Convert to title case for display
    return name


def sanitize_condition_id(name: str, category: str = "") -> str:
    """Create a valid XML ID from operator name."""
    # Remove leading underscores and replace special chars
    name = name.lstrip('_')
    # Include category prefix to avoid duplicates between portable/quantized
    if category:
        return f"op_{category.lower()}_{name}"
    return f"op_{name}"


def scan_operators(source_dir: Path, subdir: str, category: str) -> list[OperatorComponent]:
    """Scan a directory for operator source files."""
    operators = []
    op_dir = source_dir / subdir
    
    if not op_dir.exists():
        print(f"Warning: Operator directory not found: {op_dir}")
        return operators
        
    for cpp_file in sorted(op_dir.glob("op_*.cpp")):
        op_name = extract_operator_name(cpp_file.name)
        if op_name:
            operators.append(OperatorComponent(
                name=sanitize_component_name(op_name),
                category=category,
                source_file=cpp_file.relative_to(source_dir),
                condition_id=sanitize_condition_id(op_name, category),
            ))
            
    return operators


def generate_operator_condition(op: OperatorComponent, relative_base: str = "") -> str:
    """Generate XML condition for an operator."""
    condition_xml = f'''    <condition id="{op.condition_id}">
      <description>Operator: {op.name}</description>
      <require condition="Kernel Utils"/>
    </condition>'''
    return condition_xml


def generate_operator_component(op: OperatorComponent, version_placeholder: str, 
                                 relative_base: str = "") -> str:
    """Generate XML component definition for an operator."""
    source_path = str(op.source_file).replace('\\', '/')
    
    # Include category in define to avoid conflicts between portable/quantized ops with same name
    define_name = f"RTE_ML_EXECUTORCH_OP_{op.category.upper()}_{op.name.upper()}"
    
    component_xml = f'''    <component Cclass="Machine Learning" Cgroup="ExecuTorch Operators" Csub="{op.category} {op.name}" Cversion="{version_placeholder}" condition="{op.condition_id}">
      <description>ExecuTorch {op.category} Operator: {op.name}</description>
      <RTE_Components_h>
        #define {define_name}     /* ExecuTorch op_{op.name} */
      </RTE_Components_h>
      <files>
        <file category="sourceCpp" name="{source_path}"/>
      </files>
    </component>'''
    return component_xml


def generate_all_components(source_dir: Path, config: dict, version: str = "1.0.0") -> dict:
    """Generate all component definitions from source directory."""
    
    # Try new structure first (src/kernels/...), fall back to old (kernels/...)
    if (source_dir / "src" / "kernels").exists():
        kernels_base = "src/kernels"
    else:
        kernels_base = "kernels"
    
    # Scan for portable operators
    portable_ops = scan_operators(source_dir, f"{kernels_base}/portable/cpu", "Portable")
    print(f"Found {len(portable_ops)} portable operators")
    
    # Scan for quantized operators  
    quantized_ops = scan_operators(source_dir, f"{kernels_base}/quantized/cpu", "Quantized")
    print(f"Found {len(quantized_ops)} quantized operators")
    
    # Generate conditions
    conditions = []
    for op in portable_ops + quantized_ops:
        conditions.append(generate_operator_condition(op))
    
    # Generate components with actual version
    portable_components = []
    for op in portable_ops:
        portable_components.append(generate_operator_component(op, version))
        
    quantized_components = []
    for op in quantized_ops:
        quantized_components.append(generate_operator_component(op, version))
    
    return {
        'conditions': '\n'.join(conditions),
        'portable_components': '\n\n'.join(portable_components),
        'quantized_components': '\n\n'.join(quantized_components),
        'portable_ops': portable_ops,
        'quantized_ops': quantized_ops,
    }


def generate_runtime_files(source_dir: Path) -> str:
    """Generate file list for runtime component."""
    files = []
    
    # Try new structure first (src/runtime/...), fall back to old (runtime/...)
    if (source_dir / "src" / "runtime").exists():
        runtime_dirs = ['src/runtime']
        schema_dir = source_dir / "src" / "schema"
    else:
        runtime_dirs = ['runtime']
        schema_dir = source_dir / "schema"
    
    for runtime_dir in runtime_dirs:
        dir_path = source_dir / runtime_dir
        if dir_path.exists():
            for cpp_file in sorted(dir_path.rglob("*.cpp")):
                path_str = str(cpp_file)
                # Exclude test files and testing utilities
                if '/test/' in path_str or '_test.cpp' in path_str:
                    continue
                # Exclude testing_util directory (requires gmock/gtest)
                if '/testing_util/' in path_str:
                    continue
                # Exclude ATen-specific files (requires full PyTorch)
                # tensor_util_aten.cpp and tensor_parser_aten.cpp need full ATen
                # but tensor_parser_exec_aten.cpp is safe (exec_aten != ATen)
                basename = cpp_file.name
                if basename.endswith('_aten.cpp') and 'exec_aten' not in basename:
                    continue
                # Exclude platform-specific files that don't work on bare-metal
                if '/platform/default/' in path_str:
                    filename = cpp_file.name
                    # Only include minimal.cpp from the default platform dir
                    # posix.cpp — needs std::chrono (fails on AC6 bare-metal)
                    # zephyr.cpp — Zephyr RTOS only
                    # android.cpp — Android only
                    # arm_zephyr.cpp — ARM Zephyr only
                    if filename not in ['minimal.cpp']:
                        continue
                rel_path = cpp_file.relative_to(source_dir)
                files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')
    
    # Add schema files
    if schema_dir.exists():
        for cpp_file in sorted(schema_dir.rglob("*.cpp")):
            rel_path = cpp_file.relative_to(source_dir)
            files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')
    
    return '\n'.join(files)


def generate_kernel_utils_files(source_dir: Path) -> str:
    """Generate file list for kernel utils component.
    
    Includes all .cpp files under kernels/portable/cpu/util/ which provide
    shared helper functions (reduce_util, broadcast_util, copy_ops_util, etc.)
    needed by both portable and quantized operators.
    """
    files = []
    
    # Try new structure first (src/kernels/...), fall back to old (kernels/...)
    if (source_dir / "src" / "kernels").exists():
        util_dir = source_dir / "src" / "kernels" / "portable" / "cpu" / "util"
    else:
        util_dir = source_dir / "kernels" / "portable" / "cpu" / "util"
    
    if util_dir.exists():
        for cpp_file in sorted(util_dir.glob("*.cpp")):
            # Skip test files
            if '_test.cpp' in cpp_file.name:
                continue
            rel_path = cpp_file.relative_to(source_dir)
            files.append(f'        <file category="sourceCpp" name="{rel_path}"/>')
    
    return '\n'.join(files)


def generate_backend_files(
    source_dir: Path,
    backend: str,
    extra_exclude: list = None,
) -> str:
    """Generate file list for backend component.

    `extra_exclude` is a list of filename substrings to exclude in addition
    to the backend's built-in exclusions. Used by the ethos_u backend to
    produce host-specific component variants (Cortex-M host vs Cortex-A
    host) — the two host TUs define the same symbols and cannot be shipped
    together in a single component.
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

    return '\n'.join(files)


def main():
    parser = argparse.ArgumentParser(description="Generate CMSIS Pack components for ExecuTorch operators")
    parser.add_argument("--source-dir", "-s", required=True, help="Path to ai_layer/engine source directory")
    parser.add_argument("--config", "-c", help="Path to executorch_config.yml")
    parser.add_argument("--output", "-o", default="components.xml", help="Output file for generated components")
    parser.add_argument("--template", "-t", help="Path to PDSC template file to populate")
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
    
    # Generate runtime files
    runtime_files = generate_runtime_files(source_dir)
    
    # Generate kernel utils files
    kernel_utils_files = generate_kernel_utils_files(source_dir)
    
    # Generate backend files.
    # EthosUBackend_Cortex_M.cpp and EthosUBackend_Cortex_A.cpp define the
    # same platform_* symbols, so each host variant ships in its own
    # component and the other host's TU is filtered out.
    ethos_u_cortex_m_files = generate_backend_files(
        source_dir, "ethos_u", extra_exclude=["Cortex_A"]
    )
    ethos_u_cortex_a_files = generate_backend_files(
        source_dir, "ethos_u", extra_exclude=["Cortex_M"]
    )
    cortex_m_files = generate_backend_files(source_dir, "cortex_m")
    
    # Output summary
    print(f"\nGenerated components:")
    print(f"  - {len(result['portable_ops'])} portable operators")
    print(f"  - {len(result['quantized_ops'])} quantized operators")
    
    # Write component definitions to output file
    with open(args.output, 'w') as f:
        f.write("<!-- Operator Conditions -->\n")
        f.write(result['conditions'])
        f.write("\n\n<!-- Portable Operator Components -->\n")
        f.write(result['portable_components'])
        f.write("\n\n<!-- Quantized Operator Components -->\n")
        f.write(result['quantized_components'])
    
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
        pdsc = pdsc.replace("%{OPERATOR_CONDITIONS}%", result['conditions'])
        pdsc = pdsc.replace("%{RUNTIME_FILES}%", runtime_files)
        pdsc = pdsc.replace("%{KERNEL_UTILS_FILES}%", kernel_utils_files)
        pdsc = pdsc.replace("%{PORTABLE_OPERATOR_COMPONENTS}%", result['portable_components'])
        pdsc = pdsc.replace("%{QUANTIZED_OPERATOR_COMPONENTS}%", result['quantized_components'])
        pdsc = pdsc.replace(
            "%{ETHOS_U_BACKEND_CORTEX_M_FILES}%", ethos_u_cortex_m_files
        )
        pdsc = pdsc.replace(
            "%{ETHOS_U_BACKEND_CORTEX_A_FILES}%", ethos_u_cortex_a_files
        )
        pdsc = pdsc.replace("%{CORTEX_M_BACKEND_FILES}%", cortex_m_files)
        
        with open(args.pdsc_output, 'w') as f:
            f.write(pdsc)
            
        print(f"PDSC file written to: {args.pdsc_output}")


if __name__ == "__main__":
    main()
