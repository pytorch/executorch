#!/usr/bin/env python3
"""
Generates .pte model, operator definitions, and header files
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True, capture_output=True, text=True
        )
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build ExecuTorch ARM Hello World model"
    )
    parser.add_argument(
        "--project-root",
        default="~/",
        help="Path to project root (should be zephry/../)",
        required=True,
    )
    parser.add_argument(
        "--model-name", default="add", help="Name of the model (default: add)"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean generated files before building"
    )

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    project_root = Path(args.project_root)
    executorch_root = project_root / "modules" / "lib" / "executorch"

    src_dir = script_dir / "src"

    model_name = args.model_name
    pte_file = f"{model_name}.pte"
    ops_def_file = "gen_ops_def.yml"
    header_file = "model_pte.h"

    print(f"Building ExecuTorch model: {model_name}")
    print(f"ExecuTorch root: {executorch_root}")
    print(f"Working directory: {script_dir}")

    # Clean previous build if requested
    if args.clean:
        files_to_clean = [pte_file, ops_def_file, src_dir / header_file]
        for file_path in files_to_clean:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"Cleaned: {file_path}")

    # Step 1: Generate the .pte model file
    export_script = (
        executorch_root / "extension" / "embedded" / "export_{model_name}.py"
    )
    if not export_script.exists():
        print(f"Error: Export script not found: {export_script}")
        sys.exit(1)

    try:
        run_command(
            [sys.executable, str(export_script)],
            cwd=script_dir,
            description="Generating .pte model file",
        )
    except SystemExit:
        print(
            "\n❌ Model generation failed. This is likely because PyTorch/ExecuTorch is not installed."
        )
        print("For now, using dummy model_pte.h for compilation testing.")
        print("To generate a real model, install PyTorch and ExecuTorch:")
        print("  pip install torch")
        print("  # Install ExecuTorch according to documentation")
        print("  python build_model.py")
        return

    if not Path(script_dir / pte_file).exists():
        print(f"Error: Model file {pte_file} was not generated")
        sys.exit(1)

    # Step 2: Generate operator definitions

    gen_ops_script = (
        "/home/zephyruser/optional/modules/lib/executorch/codegen/tools/gen_ops_def.py"
    )
    if not os.path.exists(gen_ops_script):
        print(f"Error: gen_ops_def.py not found at {gen_ops_script}")
        sys.exit(1)

    run_command(
        [
            sys.executable,
            str(gen_ops_script),
            "--output_path",
            ops_def_file,
            "--model_file_path",
            pte_file,
        ],
        cwd=script_dir,
        description="Generating operator definitions",
    )

    # Step 3: Convert .pte to header file
    # pte_to_header_script = executorch_root / "examples" / "arm" / "executor_runner" / "pte_to_header.py"
    pte_to_header_script = "/home/zephyruser/optional/modules/lib/executorch/examples/arm/executor_runner/pte_to_header.py"
    if not os.path.exists(pte_to_header_script):
        print(f"Error: pte_to_header.py not found at {pte_to_header_script}")
        sys.exit(1)

    run_command(
        [
            sys.executable,
            str(pte_to_header_script),
            "--pte",
            pte_file,
            "--outdir",
            "src",
        ],
        cwd=script_dir,
        description="Converting .pte to header file",
    )

    # Step 4: Make the generated array const and remove section attribute
    header_path = src_dir / header_file
    if header_path.exists():
        content = header_path.read_text()

        # Remove section attribute and replace with Zephyr alignment macro
        import re

        # Replace section+aligned pattern with Zephyr __ALIGN macro
        content = re.sub(
            r"__attribute__\s*\(\s*\(\s*section\s*\([^)]*\)\s*,\s*aligned\s*\(([^)]*)\)\s*\)\s*\)\s*",
            r"__ALIGN(\1) ",
            content,
        )
        # Remove any remaining section-only attributes
        content = re.sub(
            r"__attribute__\s*\(\s*\(\s*section\s*\([^)]*\)\s*\)\s*\)\s*", "", content
        )
        # Also replace any standalone __attribute__((aligned(n))) with __ALIGN(n)
        content = re.sub(
            r"__attribute__\s*\(\s*\(\s*aligned\s*\(([^)]*)\)\s*\)\s*\)\s*",
            r"__ALIGN(\1) ",
            content,
        )

        # Replace 'char model_pte_data[]' with 'const char model_pte_data[]'
        content = content.replace(
            "char model_pte_data[]", "const char model_pte_data[]"
        )
        # Also handle 'char model_pte[]' variant
        content = content.replace("char model_pte[]", "const char model_pte[]")

        header_path.write_text(content)
        print(
            f"✓ Made model data const and removed section attributes in {header_file}"
        )
    else:
        print(f"Warning: Header file {header_file} not found")

    print("\n=== Build Summary ===")
    print(f"✓ Generated: {pte_file}")
    print(f"✓ Generated: {ops_def_file}")
    print(f"✓ Generated: src/{header_file}")
    print("\nNext steps:")
    print("1. Review gen_ops_def.yml and customize if needed")
    print("2. Build the Zephyr application with west build")


if __name__ == "__main__":
    main()
