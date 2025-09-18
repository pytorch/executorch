#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for CUDA AOTI export functionality.
This script tests basic CUDA export functionality for a subset of models:
linear, conv2d, add, and resnet18.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional


def run_command(
    cmd: List[str], cwd: Optional[str] = None, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run a command with proper error handling and timeout."""
    print(f"Running command: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # We'll handle the return code ourselves
        )

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Command timed out after {timeout} seconds")
        raise e
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        raise e


def test_cuda_export(
    model_name: str, export_mode: str = "export_aoti_only", timeout: int = 300
) -> bool:
    """Test CUDA export for a specific model."""
    print(f"\n{'='*60}")
    print(f"Testing CUDA export for model: {model_name}")
    print(f"Export mode: {export_mode}")
    print(f"{'='*60}")

    try:
        # Run the export using export_aoti.py
        cmd = ["python", "export_aoti.py", model_name]
        if export_mode == "export_aoti_only":
            cmd.append("--aoti_only")

        result = run_command(cmd, timeout=timeout)

        if result.returncode == 0:
            print(f"SUCCESS: {model_name} export completed successfully")
            return True
        else:
            print(
                f"ERROR: {model_name} export failed with return code {result.returncode}"
            )
            return False

    except subprocess.TimeoutExpired:
        print(f"ERROR: {model_name} export timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"ERROR: {model_name} export failed with exception: {e}")
        return False


def cleanup_temp_files():
    """Clean up temporary files generated during export."""
    print("Cleaning up temporary files...")

    # List of file patterns to clean up
    cleanup_patterns = [
        "*.cubin",
        "*.pte",
        "*.so",
        "*kernel_metadata.json",
        "*kernel.cpp",
        "*wrapper_metadata.json",
        "*wrapper.cpp",
        "*wrapper.json",
        "aoti_intermediate_output.txt",
    ]

    # Remove files matching patterns
    for pattern in cleanup_patterns:
        try:
            import glob

            files = glob.glob(pattern)
            for file in files:
                if os.path.isfile(file):
                    os.remove(file)
                    print(f"Removed file: {file}")
        except Exception as e:
            print(f"Warning: Failed to remove {pattern}: {e}")

    # Remove temporary directories created by wrappers
    try:
        import glob

        for wrapper_file in glob.glob("*wrapper.cpp"):
            basename = wrapper_file.replace("wrapper.cpp", "")
            if os.path.isdir(basename):
                import shutil

                shutil.rmtree(basename)
                print(f"Removed directory: {basename}")
    except Exception as e:
        print(f"Warning: Failed to remove wrapper directories: {e}")

    print("Cleanup completed.")


def main():
    """Main function to test CUDA export for specified models."""
    parser = argparse.ArgumentParser(
        description="Test CUDA AOTI export functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "conv2d", "add", "resnet18"],
        help="List of models to test (default: linear, conv2d, add, resnet18)",
    )

    parser.add_argument(
        "--export-mode",
        choices=["export_aoti_only", "full"],
        default="export_aoti_only",
        help="Export mode: export_aoti_only (AOTI only) or full (full pipeline)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each model export in seconds (default: 300)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=True,
        help="Clean up temporary files after testing (default: True)",
    )

    args = parser.parse_args()

    print("CUDA AOTI Export Test")
    print("=" * 60)
    print(f"Models to test: {args.models}")
    print(f"Export mode: {args.export_mode}")
    print(f"Timeout per model: {args.timeout} seconds")
    print(f"Cleanup enabled: {args.cleanup}")
    print("=" * 60)

    # Check if we're in the correct directory (should have export_aoti.py)
    if not os.path.exists("export_aoti.py"):
        print("ERROR: export_aoti.py not found in current directory")
        print("Please run this script from the executorch root directory")
        sys.exit(1)

    # Test each model
    successful_models = []
    failed_models = []

    for model in args.models:
        # Clean up before each test
        if args.cleanup:
            cleanup_temp_files()

        success = test_cuda_export(model, args.export_mode, args.timeout)

        if success:
            successful_models.append(model)
        else:
            failed_models.append(model)

    # Final cleanup
    if args.cleanup:
        cleanup_temp_files()

    # Print summary
    print("\n" + "=" * 60)
    print("CUDA AOTI Export Test Summary")
    print("=" * 60)
    print(f"Total models tested: {len(args.models)}")
    print(f"Successful exports: {len(successful_models)}")
    print(f"Failed exports: {len(failed_models)}")

    if successful_models:
        print(f"Successful models: {', '.join(successful_models)}")

    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
        print("\nERROR: One or more model exports failed!")
        sys.exit(1)
    else:
        print("\nSUCCESS: All model exports completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
